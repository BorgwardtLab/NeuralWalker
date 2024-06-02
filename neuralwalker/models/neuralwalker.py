import torch
from torch import nn
import torch_geometric.nn as gnn
from ..modules.head import PredictionHead
from ..modules.feature_encoder import FeatureEncoder
from ..modules.neuralwalker_layer import NeuralWalkerLayer


class NeuralWalker(nn.Module):
    def __init__(
        self,
        in_node_dim,
        num_class,
        hidden_size=64,
        num_layers=2,
        walk_encoder='conv',
        in_edge_dim=None,
        node_embed=True,
        edge_embed=True,
        walk_pos_embed=False,
        walk_length=50,
        use_positional_encoding=False,
        window_size=8,
        virtual_node=False,
        dropout=0.0,
        global_pool=None,
        head='mlp',
        pad_idx=-1,
        **kwargs
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.feature_encoder = FeatureEncoder(
            hidden_size=hidden_size,
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            node_embed=node_embed,
            edge_embed=edge_embed,
        )

        # Walk encoder
        self.walk_encoder = walk_encoder
        global_mp_type = kwargs.get('global_mp_type', 'vn')
        self.blocks = nn.ModuleList([
            NeuralWalkerLayer(
                hidden_size=hidden_size,
                sequence_layer_type=walk_encoder,
                d_state=kwargs.get('d_state', 16),
                d_conv=kwargs.get('d_conv', 4),
                expand=kwargs.get('expand', 2),
                mlp_ratio=kwargs.get('mlp_ratio', 2),
                use_encoder_norm=kwargs.get('use_encoder_norm', True),
                proj_mlp_ratio=kwargs.get('proj_mlp_ratio', 1),
                walk_length=walk_length,
                use_positional_encoding=kwargs.get('use_positional_encoding', True),
                pos_embed=walk_pos_embed,
                window_size=window_size,
                bidirection=kwargs.get('bidirection', True),
                layer_idx=i,
                local_gnn_type=kwargs.get('local_mp_type', 'gin'),
                global_model_type=None if global_mp_type == 'vn' and i == num_layers - 1 else global_mp_type,
                num_heads=kwargs.get('num_heads', 4),
                dropout=dropout,
                attn_dropout=kwargs.get('attn_dropout', 0.0),
                vn_norm_first=kwargs.get('vn_norm_first', True),
                vn_norm_type=kwargs.get('vn_norm_type', 'batchnorm'),
                vn_pooling=kwargs.get('vn_pooling', 'sum')
            ) for i in range(num_layers)
        ])

        self.node_out = None
        if kwargs.get('node_out', True):
            if global_mp_type is None or global_mp_type == 'vn':
                self.node_out = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size)
                )

        if global_pool == 'mean':
            self.global_pool = gnn.global_mean_pool
        elif global_pool == 'sum':
            self.global_pool = gnn.global_add_pool
        else:
            self.global_pool = None

        self.out_head = PredictionHead(
            hidden_size=hidden_size,
            num_class=num_class,
            head=head,
            dropout=dropout,
        )

    def forward(self, batch):
        batch.walk_pe = torch.cat(
            [batch.walk_node_id_encoding, batch.walk_node_adj_encoding], dim=-1
        )

        batch = self.feature_encoder(batch)

        for i, block in enumerate(self.blocks):
            batch = block(batch)

        h = batch.x
        if self.node_out is not None:
            h = self.node_out(h)

        # Readout
        if self.global_pool is not None:
            h = self.global_pool(h, batch.batch)

        return self.out_head(h, batch)

    def get_params(self):
        if self.walk_encoder == "s4":
            # All parameters in the model
            all_parameters = list(self.parameters())

            # General parameters don't contain the special _optim key
            param_groups = [{"params": [p for p in all_parameters if not hasattr(p, "_optim")]}]

            # Add parameters with special hyperparameters
            hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
            hps = [
                dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            ]  # Unique dicts
            for hp in hps:
                params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
                param_groups.append(
                    {"params": params, **hp}
                )

            # Print optimizer info
            keys = sorted(set([k for hp in hps for k in hp.keys()]))
            for i, g in enumerate(param_groups):
                group_hps = {k: g.get(k, None) for k in keys}
                print(' | '.join([
                    f"Optimizer group {i}",
                    f"{len(g['params'])} tensors",
                ] + [f"{k} {v}" for k, v in group_hps.items()]))

            return param_groups
        return self.parameters()
