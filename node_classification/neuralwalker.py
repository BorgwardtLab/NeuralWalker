import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch_scatter import scatter_mean, scatter_sum


class WalkEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        sequence_layer_type='conv',
        d_state=16,
        d_conv=9,
        expand=1,
        num_heads=4,
        mlp_ratio=1,
        use_edge_proj=True,
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

        self.hidden_size = hidden_size
        self.use_positional_encoding= use_positional_encoding
        walk_pe_dim = window_size * 2 - 1 if use_positional_encoding else 0
        self.bidirection = bidirection
        self.use_edge_proj = use_edge_proj
        if use_edge_proj:
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
                # nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1, padding=0),
                nn.ReLU(),
                Rearrange("a b c -> a c b")
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
        else:
            raise NotImplementedError(
                f"not supported sequence layer type: {sequence_layer_type}"
            )

        self.out_node_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * proj_mlp_ratio),
            nn.ReLU(),
            nn.Linear(hidden_size * proj_mlp_ratio, hidden_size),
            nn.Dropout(dropout),
        )
        # self.node_ln = nn.LayerNorm(hidden_size, eps=1e-05)

        if use_edge_proj:
            self.out_edge_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * proj_mlp_ratio),
                nn.ReLU(),
                nn.Linear(hidden_size * proj_mlp_ratio, hidden_size),
                nn.Dropout(dropout),
            )
            self.edge_ln = nn.LayerNorm(hidden_size, eps=1e-05)

    def reset_parameters(self):
        def reset_module_parameters(module):
            for layer in module.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
                reset_module_parameters(layer)
        reset_module_parameters(self)

    def forward(self, x, edge_index, walk_node_index, walk_edge_index, walk_pe, num_nodes, edge_attr=None):
        walk_x = x[walk_node_index]

        if self.use_edge_proj and edge_attr is not None:
            walk_e = edge_attr[walk_edge_index]
            walk_x = walk_x + self.edge_proj(walk_e)
            del walk_e

        if self.use_positional_encoding and walk_pe is not None:
            walk_x = walk_x + self.walk_pe_proj(walk_pe)

        if self.pos_embed is not None:
            walk_x = walk_x + self.pos_embed

        if self.norm is not None:
            walk_x = self.norm(walk_x)

        walk_x_forward = self.seq_layer(walk_x)

        if self.seq_layer_backward is not None:
            walk_x_backward = self.seq_layer_backward(walk_x.flip([1])).flip([1])
            walk_x_forward = (walk_x_forward + walk_x_backward) * 0.5
            del walk_x_backward

        walk_x = walk_x_forward

        node_agg = scatter_mean(
            walk_x.reshape(-1, self.hidden_size),
            walk_node_index.flatten(),
            dim=0,
            dim_size=num_nodes,
        )

        # x = self.node_ln(x + self.out_node_proj(node_agg))
        x = x + self.out_node_proj(node_agg)

        if self.use_edge_proj:
            del node_agg

            edge_agg = scatter_mean(
                walk_x[:, :-1].reshape(-1, self.hidden_size),
                walk_edge_index[:, :-1].flatten(),
                dim=0,
                dim_size=edge_index.shape[-1],
            )

            if edge_attr is not None:
                edge_attr = edge_attr + self.out_edge_proj(edge_agg)
            else:
                edge_attr = self.out_edge_proj(edge_agg)
            edge_attr = self.edge_ln(edge_attr)
        
        return x, edge_attr
