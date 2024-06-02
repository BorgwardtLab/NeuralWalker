import torch
from torch import nn
from torch_scatter import scatter_sum, scatter_mean


class VirtualNodeLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.0, norm_first=True, norm_type='batchnorm', pooling='sum'):
        super().__init__()

        self.pooling = pooling
        norm_fn = nn.LayerNorm if norm_type == 'layernorm' else nn.BatchNorm1d

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            norm_fn(hidden_size) if norm_first else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            norm_fn(hidden_size) if not norm_first else nn.Identity()
        )

    def forward(self, x, batch):
        virtual_node = getattr(batch, 'virtual_node', None)
        if self.pooling == 'mean':
            h = scatter_mean(x, batch.batch, dim=0)
        else:
            h = scatter_sum(x, batch.batch, dim=0)
        if virtual_node is not None:
            h = h + virtual_node
        virtual_node = self.mlp(h)
        x = x + virtual_node[batch.batch]

        batch.virtual_node = virtual_node
        return x
