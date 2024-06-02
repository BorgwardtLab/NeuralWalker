import torch
from torch import nn
import numpy as np


class InductiveEdgeHead(nn.Module):
    """ GNN prediction head for inductive edge/link prediction tasks.

    Implementation adapted from the transductive GraphGym's GNNEdgeHead.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.layer_post_mp = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-05),
            nn.Linear(hidden_size, hidden_size),
        )

        self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)

    def _apply_index(self, x, batch):
        return x[batch.edge_label_index]

    def forward(self, x, batch):
        x = self.layer_post_mp(x)
        pred = self._apply_index(x, batch)
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        if not self.training:  # Compute extra stats when in evaluation mode.
            batch.x = x
            stats = self.compute_mrr(batch)
            return pred, stats
        else:
            return pred

    def compute_mrr(self, batch):
        stats = {}
        for data in batch.to_data_list():
            pred = data.x @ data.x.transpose(0, 1)

            pos_edge_index = data.edge_label_index[:, data.y == 1]
            num_pos_edges = pos_edge_index.shape[1]

            pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]

            if num_pos_edges > 0:
                # raw MRR
                neg_mask = torch.ones([num_pos_edges, data.num_nodes], dtype=torch.bool)
                neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
                pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
                mrr_list = self._eval_mrr(pred_pos, pred_neg, 'torch', suffix='')

                # filtered MRR
                pred_masked = pred.clone()
                pred_masked[pos_edge_index[0], pos_edge_index[1]] -= float("inf")
                pred_neg = pred_masked[pos_edge_index[0]]
                mrr_list.update(self._eval_mrr(pred_pos, pred_neg, 'torch', suffix='_filt'))

                pred_masked[torch.arange(data.num_nodes), torch.arange(data.num_nodes)] -= float("inf")
                pred_neg = pred_masked[pos_edge_index[0]]
                mrr_list.update(self._eval_mrr(pred_pos, pred_neg, 'torch', suffix='_filt_self'))
            else:
                # Return empty stats.
                mrr_list = self._eval_mrr(pred_pos, pred_pos, 'torch')

            for key, val in mrr_list.items():
                if key.endswith('_list'):
                    key = key[:-len('_list')]
                    val = float(val.mean().item())
                if np.isnan(val):
                    val = 0.
                if key not in stats:
                    stats[key] = [val]
                else:
                    stats[key].append(val)

        return stats

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info, suffix=''):
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).

        Implementation from OGB:
        https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

        Args:
            y_pred_neg: array with shape (batch size, num_entities_neg).
            y_pred_pos: array with shape (batch size, )
        """

        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1. / ranking_list.to(torch.float)

            return {f'hits@1{suffix}_list': hits1_list,
                    f'hits@3{suffix}_list': hits3_list,
                    f'hits@10{suffix}_list': hits10_list,
                    f'mrr{suffix}_list': mrr_list}
        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg],
                                    axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1. / ranking_list.astype(np.float32)

            return {'hits@1_list': hits1_list,
                    'hits@3_list': hits3_list,
                    'hits@10_list': hits10_list,
                    'mrr_list': mrr_list}


class OGBCodeGraphHead(nn.Module):
    """
    Sequence prediction head for ogbg-code2 graph-level prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): IGNORED, kept for GraphGym framework compatibility
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out=5002, max_seq_len=5):
        super().__init__()
        self.max_seq_len = max_seq_len
        num_class = 5002

        self.classifier = nn.ModuleList()
        for i in range(max_seq_len):
            self.classifier.append(nn.Linear(dim_in, num_class))

    def forward(self, h):
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.classifier[i](h))

        return pred_list


class PredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_class,
        head='mlp',
        dropout=0.0,
    ):
        super().__init__()
        self.head = head
        if head == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_class)
            )
        elif head == 'linear':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_class)
            )
        elif head == 'inductive_edge':
            self.classifier = InductiveEdgeHead(
                hidden_size, num_class
            )
        elif head == 'code2':
            self.classifier = OGBCodeGraphHead(
                hidden_size, num_class
            )
        else:
            raise NotImplementedError

    def forward(self, x, batch):
        if self.head == 'inductive_edge':
            return self.classifier(x, batch)
        return self.classifier(x)
