import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph

from lg_parse import parse_method, parser_add_main_args
import sys
sys.path.append("../")
from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from utils import sample_walks
import os
import pandas as pd
from timeit import default_timer as timer


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
if not args.global_dropout:
    args.global_dropout = args.dropout
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

# get the splits for all runs
if args.dataset in ('ogbn-arxiv', 'ogbn-products'):
    split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)


### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

### Load method ###
model = parse_method(args, n, c, d, device)

criterion = nn.NLLLoss()
eval_func = eval_acc
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
true_label = dataset.label

### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True

    model.reset_parameters()
    model._global = False
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)
    num_batch = n // args.batch_size + 1

    train_time = 0
    for epoch in range(args.local_epochs+args.global_epochs):
        if epoch == args.local_epochs:
            print("start global attention!!!!!!")
            if args.save_model:
                model, optimizer = load_model(args, model, optimizer, run)
            model._global = True
        model.to(device)
        model.train()

        loss_train = 0
        idx = torch.randperm(n)
        tic = timer()
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            train_mask_i = train_mask[idx_i]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            walk_node_index, walk_edge_index, walk_pe = sample_walks(
                edge_index_i,
                length=args.length,
                sample_rate=args.sample_rate,
                window_size=args.window_size,
                num_nodes=len(x_i),
                device=device,
            )
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            optimizer.zero_grad()
            out_i = model(x_i, edge_index_i, walk_node_index, walk_edge_index, walk_pe, len(x_i))
            out_i = F.log_softmax(out_i, dim=1)
            loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= num_batch
        train_time += timer() - tic

        if epoch % args.eval_step == 0 and epoch > args.eval_epoch:
            result = evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]
                if args.save_model:
                    save_model(args, model, optimizer, run)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss_train:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%, '
                      f'Best Valid: {100 * best_val:.2f}%, '
                      f'Best Test: {100 * best_test:.2f}%')
    if args.save_model:
        model, _ = load_model(args, model, optimizer, run)
        all_results = []
        for _ in range(args.test_runs):
            result = evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device)
            all_results.append(result[:-1])
        all_results = np.asarray(all_results)
        result = all_results.mean(0)
        result_std = all_results.std(0)
        result[1] = best_val + 1e-06 # add a small value to make it the best
        logger.add_result(run, result)
        results = {
            'train': result[0],
            'train_std': result_std[0],
            'val': result[1],
            'val_std': result_std[1],
            'test': result[2],
            'test_std': result_std[2],
            'train_time': train_time,
            'train_time_per_epoch': train_time / (args.local_epochs+args.global_epochs),
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        outdir = f'results/{args.dataset}'
        os.makedirs(outdir, exist_ok=True)
        results.to_csv(f'{outdir}/{args.length}_{args.window_size}_{args.sample_rate}_{args.seq_layer_type}_{args.walk_encoder_dropout}_{run}.csv',
                       header=['value'], index_label='name')
    logger.print_statistics(run)

results = logger.print_statistics()
### Save results ###
save_result(args, results)
