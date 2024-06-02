import torch
import torch.nn.functional as F
from utils import sample_walks

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, edge_index, device, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        num_nodes = dataset.graph['num_nodes']
        walk_node_index, walk_edge_index, walk_pe = sample_walks(
            edge_index,
            length=args.length,
            sample_rate=min(args.test_sample_rate, 1.0),
            window_size=args.window_size,
            num_nodes=num_nodes,
            device=device,
        )
        out = model(
            dataset.graph['node_feat'],
            dataset.graph['edge_index'],
            walk_node_index,
            walk_edge_index,
            walk_pe,
            num_nodes,
        )

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
    valid_loss = valid_loss.item()

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    num_nodes = dataset.graph['num_nodes']
    walk_node_index, walk_edge_index, walk_pe = sample_walks(
        edge_index,
        length=args.length,
        sample_rate=min(args.test_sample_rate, 1.0),
        window_size=args.window_size,
        num_nodes=num_nodes,
        device=torch.device('cpu'),
    )
    out = model(x, edge_index, walk_node_index, walk_edge_index, walk_pe, num_nodes)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
    valid_loss = valid_loss.item()

    return train_acc, valid_acc, test_acc, valid_loss, out
