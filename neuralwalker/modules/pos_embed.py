import math
import torch
from torch import nn
from torch_scatter import scatter_mean


def get_1d_sine_pos_embed(dim, length, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )
    pos = torch.arange(0, length)
    args = pos[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size, walk_length):
        super().__init__()

        self.walk_pos_embed = nn.Parameter(
            torch.zeros(1, walk_length, hidden_size), requires_grad=False
        )

        self.walk_pos_embed.data.copy_(
            get_1d_sine_pos_embed(hidden_size, walk_length).unsqueeze(0)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        x,
        walk_node_index,
        walk_node_mask,
        edge_attr=None,
        walk_edge_index=None,
        walk_edge_mask=None,
        num_nodes=None,
        num_edges=None,
    ):

        walk_x = torch.where(walk_node_mask[:, :, None], 0., x[walk_node_index])
        walk_x = walk_x + self.walk_pos_embed
        walk_x = self.mlp(walk_x)

        walk_x = scatter_mean(
            walk_x[~walk_node_mask],
            walk_node_index[~walk_node_mask],
            dim=0,
            dim_size=num_nodes,
        )

        return walk_x
