import copy
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from .neuralwalker import NeuralWalker
from ..loss import get_loss_function, get_metric_function, setup_metric_function
from ..lr_schedulers import get_lr_schedule_cls


class NeuralWalker_pl(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.model = NeuralWalker(**OmegaConf.to_container(cfg.model, resolve=True))
        self.cfg = cfg

        self.criterion = get_loss_function(cfg.training.loss)
        self.metric, self.metric_mode = get_metric_function(cfg.dataset.metric)
        self.metric = setup_metric_function(
            self.metric,
            cfg.dataset.metric,
            cfg.dataset.name
        )
        if self.cfg.dataset.name == 'ogbg-code2':
            from ..data.ogbg_code2_utils import decode_arr_to_seq, idx2vocab
            self.arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)

        self.best_weights = None
        self.best_val_score = -float('inf')

        self.validation_step_outputs = []
        self.save_hyperparameters()

    def compute_loss(self, batch, return_y=False):
        y_pred = self.model(batch)
        y_true = batch.y_arr if self.cfg.dataset.name == 'ogbg-code2' else batch.y
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.squeeze(-1)
        if return_y:
            return self.criterion(y_pred, y_true), y_pred, batch.y
        return self.criterion(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return loss

    def val_test_step(self, batch, batch_idx, step='val'):
        if 'edge' in self.cfg.model.head:
            y_pred, stats = self.model(batch)
            loss = self.criterion(y_pred, batch.y)
            self.log(f"{step}_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.y))
            stats = {key: value for key, value in stats.items() if self.cfg.dataset.metric in key}
            mean_stats = {f'{step}_{key}': sum(value) / len(value) for key, value in stats.items()}
            self.log_dict(mean_stats, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
            self.validation_step_outputs.append(stats)
            return loss
        loss, y_pred, y_true = self.compute_loss(batch, return_y=True)
        self.log(f"{step}_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.y))
        if self.cfg.dataset.name == 'ogbg-code2':
            mat = []
            for i in range(len(y_pred)):
                mat.append(y_pred[i].argmax(dim=1).view(-1, 1))
            mat = torch.cat(mat, dim = 1)
            y_pred = [self.arr_to_seq(arr) for arr in mat]
            y_true = [y_true[i] for i in range(len(y_true))]
        self.validation_step_outputs.append({'y_pred': y_pred, 'y_true': y_true})
        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, 'test')

    def on_validation_epoch_end(self):
        if 'edge' in self.cfg.model.head:
            score = torch.cat(
                [torch.tensor(out[self.cfg.dataset.metric]) for out in self.validation_step_outputs]
            )
            score = torch.mean(score).item()
            self.log('val_score', score)
        elif self.cfg.dataset.name == 'ogbg-code2':
            y_pred = []
            y_true = []
            for out in self.validation_step_outputs:
                y_pred.extend(out['y_pred'])
                y_true.extend(out['y_true'])
            score = self.metric(y_pred, y_true).item()
            self.log('val_score', score)
        else:
            y_pred = torch.cat([out['y_pred'] for out in self.validation_step_outputs])
            y_true = torch.cat([out['y_true'] for out in self.validation_step_outputs])
            score = self.metric(y_pred, y_true).item()
            self.log('val_score', score)
        if self.metric_mode == 'min':
            score = -score
        if score > self.best_val_score:
            self.best_val_score = score
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.current_epoch
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        if 'edge' in self.cfg.model.head:
            score = torch.cat([torch.tensor(out[self.cfg.dataset.metric]) for out in self.validation_step_outputs])
            score = torch.mean(score).item()
            self.log("test_score", score)
        elif self.cfg.dataset.name == 'ogbg-code2':
            y_pred = []
            y_true = []
            for out in self.validation_step_outputs:
                y_pred.extend(out['y_pred'])
                y_true.extend(out['y_true'])
            score = self.metric(y_pred, y_true).item()
            self.log('test_score', score)
        else:
            y_pred = torch.cat([out['y_pred'] for out in self.validation_step_outputs])
            y_true = torch.cat([out['y_true'] for out in self.validation_step_outputs])
            score = self.metric(y_pred, y_true).item()
            self.log("test_score", score)
        self.validation_step_outputs.clear()

    def on_fit_end(self):
        self.model.load_state_dict(self.best_weights)
        self.best_weights = None
        if self.metric_mode == 'min':
            self.best_val_score *= -1.

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.get_params(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        lr_scheduler = get_lr_schedule_cls(self.cfg.training.lr_schedule)(
            optimizer,
            self.cfg.training.warmup * self.cfg.training.iterations,
            self.cfg.training.epochs * self.cfg.training.iterations,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
