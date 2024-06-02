import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import accuracy, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from functools import partial
from ogb.graphproppred import Evaluator


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                  weight=weight[true])
        return loss

def multilabel_cross_entropy(pred, true):
    bce_loss = nn.BCEWithLogitsLoss()
    is_labeled = true == true  # Filter our nans.
    return bce_loss(pred[is_labeled], true[is_labeled].float())

def code2_loss(pred, true):
    loss = 0
    for i in range(len(pred)):
        loss += F.cross_entropy(pred[i].to(torch.float32), true[:,i])
    return loss / len(pred)


LOSS = {
    'l1': nn.L1Loss(),
    'l2': nn.MSELoss(),
    'ce': nn.CrossEntropyLoss(),
    'bce': lambda x, y: F.binary_cross_entropy_with_logits(x, y.float()),
    'wce': weighted_cross_entropy,
    'mce': multilabel_cross_entropy,
    'code2': code2_loss,
}

def get_loss_function(loss='l1'):
    return LOSS[loss]

def custom_accuracy(preds, targets):
    return accuracy(preds, targets, task='multiclass', num_classes=preds.shape[-1])

def custom_f1(preds, targets):
    return f1_score(preds, targets, average='macro', task='multiclass', num_classes=preds.shape[-1])

def accuracy_sbm(preds, targets):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets.cpu().numpy()
    C = preds.argmax(dim=-1).cpu().numpy()
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return torch.tensor(acc)

def custom_ap(y_pred, y_true):
    '''
        compute Average Precision (AP) averaged across tasks
    '''
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return torch.tensor(sum(ap_list) / len(ap_list))

def custom_ogbg(y_pred, y_true, evaluator, metric):
    if isinstance(y_pred, list):
        score = evaluator.eval(
            {'seq_pred': y_pred, 'seq_ref': y_true}
        )[metric]
        return torch.tensor(score)

    if metric == 'acc':
        y_pred = y_pred.argmax(-1)
        y_pred, y_true = y_pred.view(-1, 1), y_true.view(-1, 1)
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    score = evaluator.eval(
        {'y_pred': y_pred, 'y_true': y_true}
    )[metric]
    return torch.tensor(score)

def setup_metric_function(metric, metric_name, dataset_name):
    if 'ogbg' in metric_name:
        evaluator = Evaluator(dataset_name)
        metric = partial(
            metric,
            evaluator=evaluator,
            metric=metric_name.split('-')[-1]
        )
    return metric


METRIC = {
    'mae': (nn.L1Loss(), 'min'),
    'mse': (nn.MSELoss(), 'min'),
    'accuracy': (custom_accuracy, 'max'),
    'accuracy_sbm': (accuracy_sbm, 'max'),
    'f1': (custom_f1, 'max'),
    'ap': (custom_ap, 'max'),
    'mrr': (None, 'max'),
    'ogbg-ap': (custom_ogbg, 'max'),
    'ogbg-acc': (custom_ogbg, 'max'),
    'ogbg-F1': (custom_ogbg, 'max'),
}

def get_metric_function(metric):
    return METRIC[metric]
