import torch
from torch_geometric.utils import coalesce
from neuralwalker.data.wrapper import sample_random_walks


def sample_walks(edge_index, length, sample_rate, window_size, num_nodes, device):
    walk_node_index, walk_edge_index, walk_node_id_encoding, walk_node_adj_encoding = sample_random_walks(
        coalesce(edge_index, num_nodes=num_nodes),
        num_nodes,
        length,
        sample_rate,
        False,
        False,
        window_size,
        -1,
    )
    walk_node_index = torch.from_numpy(walk_node_index)
    walk_edge_index = torch.from_numpy(walk_edge_index)
    walk_node_id_encoding = torch.from_numpy(walk_node_id_encoding)
    walk_node_adj_encoding = torch.from_numpy(walk_node_adj_encoding)
    walk_pe = torch.cat(
        [walk_node_id_encoding, walk_node_adj_encoding], dim=-1
    ).float()

    walk_node_index, walk_edge_index, walk_pe = \
        walk_node_index.to(device), walk_edge_index.to(device), walk_pe.to(device)
    return walk_node_index, walk_edge_index, walk_pe
