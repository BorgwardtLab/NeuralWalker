import numpy as np
from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import Dataset, Subset
from hydra.utils import get_class


def get_dataset(cfg, transform=None, test_transform=None):
    if test_transform is None:
        test_transform = transform
    cls_name = get_class(cfg.dataset.cls_name)
    kwargs = {
        'batch_size': cfg.training.batch_size,
        'num_workers': cfg.training.num_workers,
        'pin_memory': cfg.training.pin_memory,
    }
    if cfg.dataset.name == 'zinc':
        dataset = LightningDataset(
            cls_name(cfg.dataset.path, subset=True, split='train', transform=transform),
            cls_name(cfg.dataset.path, subset=True, split='val', transform=test_transform),
            cls_name(cfg.dataset.path, subset=True, split='test', transform=test_transform),
            **kwargs
        )
        return dataset
    elif cfg.dataset.name in ['CIFAR10', 'MNIST', 'PATTERN', 'CLUSTER'] +\
        ['PascalVOC-SP', 'COCO-SP', 'Peptides-func', 'Peptides-struct', 'PCQM-Contact']:
        dataset = LightningDataset(
            cls_name(cfg.dataset.path, cfg.dataset.name, split='train', transform=transform),
            cls_name(cfg.dataset.path, cfg.dataset.name, split='val', transform=test_transform),
            cls_name(cfg.dataset.path, cfg.dataset.name, split='test', transform=test_transform),
            **kwargs
        )
        return dataset
    elif 'ogbg' in cfg.dataset.name:
        dataset = cls_name(name=cfg.dataset.name, root=cfg.dataset.path)
        split_idx = dataset.get_idx_split()
        transform.transforms[0].build_vocab(dataset, split_idx)
        train_idx, val_idx = clip_graph_size(dataset, split_idx, getattr(cfg.dataset, 'max_num_nodes', None))
        dataset = LightningDataset(
            MySubset(Subset(dataset, train_idx), transform=transform),
            MySubset(Subset(dataset, val_idx), transform=test_transform),
            MySubset(Subset(dataset, split_idx['test']), transform=test_transform),
            **kwargs
        )
        return dataset
    else:
        raise NotImplementedError


class MySubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x
        
    def __len__(self):
        return len(self.subset)


def clip_graph_size(dataset, split_idx, max_num_nodes=None):
    if max_num_nodes is None:
        return split_idx['train'], split_idx['valid']
    train_mask = np.array([dataset[i].num_nodes for i in split_idx['train']]) <= max_num_nodes
    val_mask = np.array([dataset[i].num_nodes for i in split_idx['valid']]) <= max_num_nodes
    print(f"Ratio of graphs with size <= {max_num_nodes}: {np.sum(train_mask) / len(split_idx['train'])}")
    return split_idx['train'][train_mask], split_idx['valid'][val_mask]
