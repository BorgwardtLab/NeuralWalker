import torch


def update_cfg(cfg, dataset):
    # update number of iterations
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    iterations = len(dataset.train_dataloader()) // num_devices
    cfg.training.iterations = iterations
    return cfg
