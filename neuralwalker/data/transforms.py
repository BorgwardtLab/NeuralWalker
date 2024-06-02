import torch
from torch_geometric.data import Data
import numpy as np
from .wrapper import sample_random_walks


class WalkData(Data):
    """A Data wrapper with walk features
    """
    def __init__(self, data=None):
        if data is None:
            data_dict = {}
        else:
            data_dict = {key: item for key, item in data}
        super().__init__(**data_dict)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'walk_node_idx':
            return self.num_nodes
        if key == 'walk_edge_idx':
            return self.edge_index.shape[1]
        else:
            return super(WalkData, self).__inc__(key, value, *args, **kwargs)


class RandomWalkSampler(object):
    def __init__(
        self,
        length=50,
        sample_rate=1.,
        backtracking=False,
        strict=False,
        pad_idx=-1,
        window_size=8,
        **kwargs
    ):
        self.length = length
        self.sample_rate = sample_rate
        self.backtracking = backtracking
        self.strict = strict
        self.pad_idx = pad_idx
        self.window_size = window_size

    def __call__(self, data):
        if not data.is_coalesced():
            data = data.coalesce()
        walk_node_index, walk_edge_index, walk_node_id_encoding, walk_node_adj_encoding = sample_random_walks(
            data.edge_index,
            data.num_nodes,
            self.length,
            self.sample_rate,
            self.backtracking,
            self.strict,
            self.window_size,
            self.pad_idx
        )

        data.walk_node_idx = torch.from_numpy(walk_node_index)
        data.walk_edge_idx = torch.from_numpy(walk_edge_index)
        data.walk_node_mask = data.walk_node_idx == self.pad_idx
        data.walk_edge_mask = data.walk_edge_idx == self.pad_idx

        data.walk_node_id_encoding = torch.from_numpy(walk_node_id_encoding).float()
        data.walk_node_adj_encoding = torch.from_numpy(walk_node_adj_encoding).float()
        return WalkData(data)


class Preprocessor(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def build_vocab(self, code2_dataset, split_idx):
        if self.dataset == 'ogbg-code2':
            from torch_geometric.transforms import Compose
            from .ogbg_code2_utils import idx2vocab, get_vocab_mapping, \
                augment_edge, encode_y_to_arr
            num_vocab = 5000  # The number of vocabulary used for sequence prediction
            max_seq_len = 5  # The maximum sequence length to predict

            seq_len_list = np.array([len(seq) for seq in code2_dataset.data.y])
            print(f"Target sequences less or equal to {max_seq_len} is "
                f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

            vocab2idx, idx2vocab_local = get_vocab_mapping(
                [code2_dataset.data.y[i] for i in split_idx['train']], num_vocab
            )
            idx2vocab.extend(idx2vocab_local)
            transform = Compose(
                [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)]
            )
            self.transform = transform

    def __call__(self, data):
        if self.dataset in ['zinc']:
            data.x = data.x.squeeze(-1)
        if self.dataset in ['MNIST', 'CIFAR10']:
            data.x = torch.cat([data.x, data.pos], dim=-1)
            data.edge_attr = data.edge_attr.view(-1, 1)
        if self.dataset in ['PCQM-Contact']:
            data = structured_neg_sampling_transform(data)
            data.y = data.pop('edge_label')
        if self.dataset == 'ogbg-ppa':
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            data.y = data.y.squeeze(-1)
        if self.dataset == 'ogbg-code2':
            data = self.transform(data)
            data.node_depth = data.node_depth.squeeze(-1)
        return data


def custom_structured_negative_sampling(edge_index, num_nodes: int,
                                        num_neg_per_pos: int,
                                        contains_neg_self_loops: bool = True,
                                        return_ik_only: bool = False):
    r"""Customized `torch_geometric.utils.structured_negative_sampling`.

    Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        num_neg_per_pos (int): Number of negative edges to sample from a head
            (source) of each positive edge
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        return_ik_only: Instead of :obj:`(i,j,k)` return just :obj:`(i,k)`
            leaving out the original tails of the positive edges.

    :rtype: (LongTensor, LongTensor, LongTensor) or (LongTensor, LongTensor)
    """

    def get_redo_indices(neg_idx, pos_idx):
        """
        Compute indices in `neg_idx` that are invalid because they:
        a) overlap with `neg_idx`, i.e. these are in fact positive edges
        b) are duplicates of the same edge in `neg_idx`
        Args:
            neg_idx (LongTensor): Candidate negative edges encodes as indices in
                a serialized adjacency matrix.
            pos_idx (LongTensor): Positive edges encodes as indices in
                a serialized adjacency matrix.

        Returns:
            LongTensor
        """
        _, unique_ind = np.unique(neg_idx, return_index=True)
        duplicate_mask = np.ones(len(neg_idx), dtype=bool)
        duplicate_mask[unique_ind] = False
        mask = torch.from_numpy(np.logical_or(np.isin(neg_idx, pos_idx),
                                              duplicate_mask)).to(torch.bool)
        return mask.nonzero(as_tuple=False).view(-1)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col  # Encodes as the index in a serialized adjacency matrix
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    heads = row.unsqueeze(1).repeat(1, num_neg_per_pos).flatten()
    if not return_ik_only:
        tails = col.unsqueeze(1).repeat(1, num_neg_per_pos).flatten()
    rand = torch.randint(num_nodes, (num_neg_per_pos * row.size(0),),
                         dtype=torch.long)
    neg_idx = heads * num_nodes + rand

    # Resample duplicates or sampled negative edges that are actually positive.
    tries_left = 10
    redo = get_redo_indices(neg_idx, pos_idx)
    while redo.numel() > 0 and tries_left > 0:  # pragma: no cover
        tries_left -= 1
        tmp = torch.randint(num_nodes, (redo.size(0),), dtype=torch.long)
        rand[redo] = tmp
        neg_idx = heads * num_nodes + rand
        redo = get_redo_indices(neg_idx, pos_idx)

    # Remove left-over invalid edges.
    if redo.numel() > 0:
        # print(f"> FORCED TO REMOVE {redo.numel()} edges.")
        del_mask = torch.ones(heads.numel(), dtype=torch.bool)
        del_mask[redo] = False
        heads = heads[del_mask]
        rand = rand[del_mask]
        if not return_ik_only:
            tails = tails[del_mask]

    if not return_ik_only:
        return heads, tails, rand
    else:
        return heads, rand


def structured_neg_sampling_transform(data):
    """ Structured negative sampling for link prediction tasks as a transform.

    Sample `num_neg_per_pos` negative edges for each head node of a positive
    edge.

    Args:
        data (torch_geometric.data.Data): Input data object

    Returns: Transformed data object with negative edges + link pred labels
    """
    from torch_geometric.graphgym.models.transform import create_link_label
    id_pos = data.edge_label_index[:, data.edge_label == 1]  # Positive edge_index
    sampling_out = custom_structured_negative_sampling(
        edge_index=id_pos,
        num_nodes=data.num_nodes,
        num_neg_per_pos=2,
        contains_neg_self_loops=True,
        return_ik_only=True)
    id_neg = torch.stack(sampling_out)

    data.edge_label_index = torch.cat([id_pos, id_neg], dim=-1)
    data.edge_label = create_link_label(id_pos, id_neg).int()
    return data


def get_mean_std_tensors(dataset):
    """Tensors taken from https://github.com/toenshoff/LRGB/blob/main/graphgps/encoder/voc_superpixels_encoder.py
    """
    if dataset == 'PascalVOC-SP':
        node_mean = torch.tensor([[
            4.5824501e-01, 4.3857411e-01, 4.0561178e-01, 6.7938097e-02,
            6.5604292e-02, 6.5742709e-02, 6.5212941e-01, 6.2894762e-01,
            6.0173863e-01, 2.7769071e-01, 2.6425251e-01, 2.3729359e-01,
            1.9344997e+02, 2.3472206e+02
        ]])
        node_std = torch.tensor([[
            2.5952947e-01, 2.5716761e-01, 2.7130592e-01, 5.4822665e-02,
            5.4429270e-02, 5.4474957e-02, 2.6238337e-01, 2.6600540e-01,
            2.7750680e-01, 2.5197381e-01, 2.4986187e-01, 2.6069802e-01,
            1.1768297e+02, 1.4007195e+02
        ]])
        edge_mean = torch.tensor([[0.07640745, 33.73478]])
        edge_std = torch.tensor([[0.0868775, 20.945076]])
    elif dataset == 'COCO-SP':
        node_mean = torch.tensor([[
            4.6977347e-01, 4.4679317e-01, 4.0790915e-01, 7.0808627e-02,
            6.8686441e-02, 6.8498217e-02, 6.7777938e-01, 6.5244222e-01,
            6.2096798e-01, 2.7554795e-01, 2.5910738e-01, 2.2901227e-01,
            2.4261935e+02, 2.8985367e+02
        ]])
        node_std = torch.tensor([[
            2.6218116e-01, 2.5831082e-01, 2.7416739e-01, 5.7440419e-02,
            5.6832556e-02, 5.7100497e-02, 2.5929087e-01, 2.6201612e-01,
            2.7675411e-01, 2.5456995e-01, 2.5140920e-01, 2.6182330e-01,
            1.5152475e+02, 1.7630779e+02
        ]])
        edge_mean = torch.tensor([[0.07848548, 43.68736]])
        edge_std = torch.tensor([[0.08902349, 28.473562]])
    else:
        raise NotImplementedError
    return node_mean, node_std, edge_mean, edge_std


def feature_normalization(dataset, dataset_name):
    if dataset_name in ['PascalVOC-SP', 'COCO-SP']:
        print("Feature normalization...")
        node_mean, node_std, edge_mean, edge_std = get_mean_std_tensors(dataset_name)
        for split in ['train_dataset', 'val_dataset', 'test_dataset']:
            dataset_split = getattr(dataset, split)
            dataset_split.x -= node_mean
            dataset_split.x /= node_std
            dataset_split.edge_attr -= edge_mean
            dataset_split.edge_attr /= edge_std
    return dataset
