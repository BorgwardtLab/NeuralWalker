import torch
from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        in_node_dim,
        in_edge_dim=None,
        node_embed=True,
        edge_embed=True,
    ):
        super().__init__()
        self.use_edge_attr = in_edge_dim is not None

        # Node embedding
        if node_embed:
            if isinstance(in_node_dim, int):
                self.node_embed = nn.Embedding(in_node_dim, hidden_size)
            elif isinstance(in_node_dim, str):
                if in_node_dim == 'atom':
                    self.node_embed = AtomEncoder(hidden_size)
                elif in_node_dim == 'ast':
                    self.node_embed = ASTNodeEncoder(hidden_size)
                else:
                    raise NotImplementedError
            elif isinstance(in_node_dim, nn.Module):
                self.node_embed = in_node_dim
            else:
                raise NotImplementedError
        else:
            self.node_embed = nn.Linear(in_node_dim, hidden_size, bias=False)

        # Edge embedding
        if in_edge_dim is not None:
            if edge_embed:
                if isinstance(in_edge_dim, int):
                    self.edge_embed = nn.Embedding(in_edge_dim, hidden_size)
                elif isinstance(in_edge_dim, str):
                    if in_edge_dim == 'bond':
                        self.edge_embed = BondEncoder(hidden_size)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                self.edge_embed = nn.Linear(in_edge_dim, hidden_size, bias=False)

    def forward(self, batch):
        x = batch.x
        edge_attr = getattr(batch, 'edge_attr', None)
        # Node embedding
        node_depth = getattr(batch, 'node_depth', None)
        if node_depth is None:
            h = self.node_embed(x)
        else:
            h = self.node_embed(x, node_depth)

        # Edge embedding
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_embed(edge_attr)

        batch.x = h
        batch.edge_attr = edge_attr
        return batch


class ASTNodeEncoder(nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.
        Output:
            emb_dim-dimensional vector
    '''
    def __init__(self, emb_dim, num_nodetypes=98, num_nodeattributes=10030, max_depth=20):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)


    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)
