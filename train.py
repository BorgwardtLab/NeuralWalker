import os
import logging
import hydra
from pyprojroot import here
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from omegaconf import OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate
from timeit import default_timer as timer

from neuralwalker.models import NeuralWalker_pl
from neuralwalker.data import get_dataset
from neuralwalker.data.transforms import RandomWalkSampler, Preprocessor, feature_normalization
from neuralwalker.utils import update_cfg

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


log = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="train"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    preprocessor = Preprocessor(cfg.dataset.name)
    transform = instantiate(cfg.random_walk)
    test_transform = instantiate(cfg.random_walk, length=cfg.random_walk.test_length, sample_rate=1.0)
    transform = Compose([preprocessor, transform])
    test_transform = Compose([preprocessor, test_transform])
    dataset = get_dataset(cfg, transform=transform, test_transform=test_transform)
    feature_normalization(dataset, cfg.dataset.name)

    cfg = update_cfg(cfg, dataset)

    model = NeuralWalker_pl(cfg)

    trainer = pl.Trainer(
        limit_train_batches=5 if cfg.debug else None,
        limit_val_batches=5 if cfg.debug else None,
        limit_test_batches=5 if cfg.debug else None,
        num_sanity_val_steps=2 if cfg.debug else 0,
        max_epochs=cfg.training.epochs,
        precision=cfg.compute.precision,
        accelerator=cfg.compute.accelerator,
        devices="auto",
        strategy="auto",
        enable_checkpointing=False,
        default_root_dir=cfg.logs.path,
        logger=[pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")],
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    tic = timer()
    trainer.fit(model, dataset)
    train_time = timer() - tic

    scores = []
    losses = []
    extra_scores = {}
    if cfg.dataset.name == 'ogbg-code2':
        dataset.kwargs['batch_size'] = 1
    for _ in range(cfg.test_runs):
        out = trainer.test(model, dataset)
        scores.append(out[0]['test_score'])
        losses.append(out[0]['test_loss'])
        for key, value in out[0].items():
            if cfg.dataset.metric in key:
                if key not in extra_scores:
                    extra_scores[key] = [value]
                else:
                    extra_scores[key].append(value)
    log.info(f"Test results ({cfg.test_runs} runs): {np.mean(scores):.4f} +- {np.std(scores):.4f}")
    results = {
        'test_loss': np.mean(losses),
        f'test_{cfg.dataset.metric}': np.mean(scores),
        f'test_{cfg.dataset.metric}_std': np.std(scores),
        f'val_{cfg.dataset.metric}': model.best_val_score,
        'best_epoch': model.best_epoch,
        'train_val_time': train_time
    }
    if len(extra_scores) > 0:
        for key, value in extra_scores.items():
            if key not in results:
                results[key] = np.mean(value)
                results[f'{key}_std'] = np.std(value)
    results = pd.DataFrame.from_dict(results, orient='index')
    os.makedirs(cfg.outdir, exist_ok=True)
    results.to_csv(cfg.outdir + '/results.csv',
                   header=['value'], index_label='name')
    trainer.save_checkpoint(f"{cfg.outdir}/best_model.ckpt")


if __name__ == "__main__":
    main()
