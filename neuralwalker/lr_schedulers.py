import math
import torch


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(optimizer, warmup_epochs, max_epochs=None):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1.0, warmup_epochs)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_inverse_sqrt_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        return math.sqrt(warmup_epochs / epoch)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_lr_schedule_cls(name):
    if name == "cosine":
        return get_cosine_schedule_with_warmup
    elif name == "constant":
        return get_constant_schedule_with_warmup
    elif name == "isqrt":
        return get_inverse_sqrt_schedule_with_warmup
    else:
        raise ValueError(f"Not implemented {name}!")
