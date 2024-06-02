import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch_scatter import scatter_mean, scatter_sum
from .mp_layer import MessagePassingLayer
from .transformer import TransformerLayer
from .pos_embed import get_1d_sine_pos_embed


class WalkEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        sequence_layer_type='conv',
        d_state=16,
        d_conv=9,
        expand=2,
        num_heads=4,
        mlp_ratio=1,
        use_encoder_norm=True,
        proj_mlp_ratio=1,
        dropout=0.,
        walk_length=50,
        use_positional_encoding=True,
        pos_embed=False,
        window_size=8,
        bidirection=False,
        layer_idx=None,
    ):
        super().__init__()

        self.use_positional_encoding= use_positional_encoding
        walk_pe_dim = window_size * 2 - 1 if use_positional_encoding else 0
        self.bidirection = bidirection
        self.edge_proj = nn.Linear(hidden_size, hidden_size)
        self.walk_pe_proj = nn.Linear(walk_pe_dim, hidden_size)

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, walk_length, hidden_size), requires_grad=False
            )
            self.pos_embed.data.copy_(
                get_1d_sine_pos_embed(hidden_size, walk_length).unsqueeze(0)
            )

        # build sequence layer
        self.norm = None
        if use_encoder_norm:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-05)
        self.seq_layer_backward = None
        self.sequence_layer_type = sequence_layer_type
        if sequence_layer_type == "conv":
            self.seq_layer = nn.Sequential(
                Rearrange("a b c -> a c b"),
                nn.Conv1d(hidden_size, hidden_size, d_conv, groups=hidden_size, padding=d_conv // 2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1, padding=0),
                nn.ReLU(),
                Rearrange("a b c -> a c b")
            )
        elif sequence_layer_type == "s4":
            from .s4 import S4Block
            self.seq_layer = S4Block(
                d_model=hidden_size,
                lr=0.001
            )
            if bidirection:
                self.seq_layer_backward = S4Block(
                    d_model=hidden_size,
                    lr=0.001
                )
        elif sequence_layer_type == 'mamba':
            from mamba_ssm import Mamba
            self.seq_layer = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
            )
            if bidirection:
                self.seq_layer_backward = Mamba(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=layer_idx,
                )
        elif sequence_layer_type == 'transformer':
            self.seq_layer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
        else:
            raise NotImplementedError(
                f"not supported sequence layer type: {sequence_layer_type}"
            )

        self.out_node_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * proj_mlp_ratio),
            nn.BatchNorm1d(hidden_size * proj_mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * proj_mlp_ratio, hidden_size)
        )

        self.out_edge_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * proj_mlp_ratio),
            nn.BatchNorm1d(hidden_size * proj_mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * proj_mlp_ratio, hidden_size)
        )

    def forward(self, batch):
        x = batch.x
        edge_attr = batch.edge_attr
        walk_node_index, walk_edge_index = batch.walk_node_idx, batch.walk_edge_idx
        walk_node_mask, walk_edge_mask = batch.walk_node_mask, batch.walk_edge_mask
        walk_pe = batch.walk_pe

        walk_x = x[walk_node_index]
        walk_x = torch.where(walk_node_mask[:, :, None], 0., walk_x)

        if edge_attr is not None:
            walk_e = edge_attr[walk_edge_index]
            walk_e = torch.where(walk_edge_mask[:, :, None], 0., walk_e)
            walk_x = walk_x + self.edge_proj(walk_e)
            del walk_e

        if self.use_positional_encoding and walk_pe is not None:
            walk_x = walk_x + self.walk_pe_proj(walk_pe)

        if self.pos_embed is not None:
            walk_x = walk_x + self.pos_embed

        if self.norm is not None:
            walk_x = self.norm(walk_x)

        if self.sequence_layer_type == 'transformer':
            walk_x_forward = self.seq_layer(walk_x, ~walk_node_mask)
        else:
            walk_x_forward = self.seq_layer(walk_x)

        if self.seq_layer_backward is not None:
            walk_x_backward = self.seq_layer_backward(walk_x.flip([1])).flip([1])
            walk_x_forward = (walk_x_forward + walk_x_backward) * 0.5
            del walk_x_backward

        walk_x = walk_x_forward

        node_agg = scatter_mean(
            walk_x[~walk_node_mask],
            walk_node_index[~walk_node_mask],
            dim=0,
            dim_size=batch.num_nodes,
        )

        x = x + self.out_node_proj(node_agg)

        del node_agg

        edge_agg = scatter_mean(
            walk_x[~walk_edge_mask],
            walk_edge_index[~walk_edge_mask],
            dim=0,
            dim_size=batch.edge_index.shape[-1],
        )

        if edge_attr is not None:
            edge_attr = edge_attr + self.out_edge_proj(edge_agg)
        else:
            edge_attr = self.out_edge_proj(edge_agg)

        batch.x = x
        batch.edge_attr = edge_attr
        return batch


class NeuralWalkerLayer(nn.Module):
    """LongWalker layer.
    """

    def __init__(
        self,
        hidden_size,
        sequence_layer_type,
        d_state=16,
        d_conv=9,
        expand=2,
        mlp_ratio=1,
        use_encoder_norm=True,
        proj_mlp_ratio=1,
        walk_length=50,
        use_positional_encoding=True,
        pos_embed=False,
        window_size=8,
        bidirection=False,
        layer_idx=None,
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

        self.walk_encoder = WalkEncoder(
            hidden_size=hidden_size,
            sequence_layer_type=sequence_layer_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_encoder_norm=use_encoder_norm,
            proj_mlp_ratio=proj_mlp_ratio,
            dropout=dropout,
            walk_length=walk_length,
            use_positional_encoding=use_positional_encoding,
            pos_embed=pos_embed,
            window_size=window_size,
            bidirection=bidirection,
            layer_idx=layer_idx,
        )

        self.mp_layer = MessagePassingLayer(
            hidden_size=hidden_size,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            vn_norm_first=vn_norm_first,
            vn_norm_type=vn_norm_type,
            vn_pooling=vn_pooling
        )

    def forward(self, batch):
        batch = self.walk_encoder(batch)
        batch = self.mp_layer(batch)
        return batch
