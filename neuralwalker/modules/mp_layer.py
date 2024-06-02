import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from .gnn_layers import get_gnn_layer
from .virtual_node_layer import VirtualNodeLayer


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        local_gnn_type='gin',
        global_model_type=None,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        vn_norm_first=True,
        vn_norm_type='batchnorm',
        vn_pooling='sum',
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.local_mp = get_gnn_layer(
            local_gnn_type,
            hidden_size,
            dropout
        )

        self.global_model_type = global_model_type
        self.global_mp = get_global_layer(
            global_model_type,
            hidden_size,
            num_heads,
            attn_dropout,
            dropout,
            vn_norm_first=vn_norm_first,
            vn_norm_type=vn_norm_type,
            vn_pooling=vn_pooling,
        )

        self.use_attn = global_model_type is not None and global_model_type != 'vn'
        self.use_ff = self.use_attn

        if self.use_attn:
            self.norm1_local = nn.BatchNorm1d(hidden_size)
            self.norm1_attn = nn.BatchNorm1d(hidden_size)


        # Feed Forward block.
        if self.use_ff:
            self.ff_linear1 = nn.Linear(hidden_size, hidden_size * 2)
            self.ff_linear2 = nn.Linear(hidden_size * 2, hidden_size)
            self.act_fn_ff = nn.ReLU()
            self.norm2 = nn.BatchNorm1d(hidden_size)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in = h
        edge_attr = batch.edge_attr

        if self.local_mp is not None:
            h, edge_attr = self.local_mp(h, batch.edge_index, edge_attr)

        if self.global_model_type is not None:
            if self.global_model_type == 'vn':
                h = self.global_mp(h, batch)
            else:
                h = self.norm1_local(h)
                h_dense, mask = to_dense_batch(h, batch.batch)
                if self.global_model_type == 'transformer':
                    h_attn = self._sa_block(h_dense, None, ~mask)[mask]
                elif self.global_model_type == 'performer':
                    h_attn = self.global_mp(h_dense, mask=mask)[mask]
                else:
                    raise RuntimeError(f"Unexpected {self.global_model_type}")

                h_attn = h_in + h_attn  # Residual connection.
                h = self.norm1_attn(h_attn)
                del h_attn, h_in

        if self.use_ff:
            h = self.norm2(h + self._ff_block(h))

        batch.x = h
        batch.edge_attr = edge_attr
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.global_mp(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


def get_global_layer(
    global_model_type,
    hidden_size,
    num_heads,
    attn_dropout,
    dropout,
    vn_norm_first=True,
    vn_norm_type='batchnorm',
    vn_pooling='sum',
):
    if global_model_type == 'transformer':
        return nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )
    elif global_model_type == 'performer':
        from performer_pytorch import SelfAttention
        return SelfAttention(
            dim=hidden_size, heads=num_heads,
            dropout=attn_dropout, causal=False
        )
    elif global_model_type == 'vn':
        return VirtualNodeLayer(
            hidden_size, dropout,
            norm_first=vn_norm_first, norm_type=vn_norm_type,
            pooling=vn_pooling
        )
    else:
        return None
