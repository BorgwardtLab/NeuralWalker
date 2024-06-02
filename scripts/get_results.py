import os
import hydra
from omegaconf import OmegaConf
from pyprojroot import here
from pathlib import Path
import pandas as pd


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="results"
)
def main(cfg):
    PATH = Path(cfg.outdir)

    param_names = ['rw.len', 'rw.ws', 'rw.rate', 'tr.lr', 'tr.wd', 'm.name', 'm.bid', 'm.drop', 'm.gpool']

    seeds = range(4)
    if 'ogb' in cfg.dataset.name:
        seeds = range(10)

    all_results = []
    for length in cfg.random_walk.length:
        for window_size in cfg.random_walk.window_size:
            for sample_rate in cfg.random_walk.sample_rate:
                for lr in cfg.training.lr:
                    for weight_decay in cfg.training.weight_decay:
                        for model_name in cfg.model.name:
                            for bidirection in cfg.model.bidirection:
                                for dropout in cfg.model.dropout:
                                    for global_pool in cfg.model.global_pool:
                                        avg_metric = []
                                        for seed in seeds:
                                            path = PATH / f"{cfg.dataset.name}" \
                                                / f"{length}_{window_size}_{sample_rate}"\
                                                / f"{lr}_{weight_decay}" \
                                                / f"{model_name}_{bidirection}_{dropout}_{global_pool}"\
                                                / f"{seed}" / "results.csv"
                                            if os.path.isfile(path):
                                                metric_df = pd.read_csv(path, index_col=0).T
                                            else:
                                                continue
                                            avg_metric.append(metric_df)
                                        if len(avg_metric) == 0:
                                            continue

                                        avg_metric = pd.concat(avg_metric).reset_index(drop=True)
                                        metric = pd.DataFrame(avg_metric.mean(axis=0)).T
                                        metric_name = cfg.dataset.metric
                                        metric = metric[[f'test_{metric_name}']]
                                        metric[f'test_{metric_name}_std_gl'] = avg_metric[f'test_{metric_name}'].std()
                                        metric[f'test_{metric_name}_std'] = avg_metric[f'test_{metric_name}_std'].mean()
                                        metric[f'val_{metric_name}'] = avg_metric[f'val_{metric_name}'].mean()
                                        metric[f'val_{metric_name}_std'] = avg_metric[f'val_{metric_name}'].std()
                                        metric['seeds'] = len(avg_metric)
                                        params = [
                                            length,
                                            window_size,
                                            sample_rate,
                                            lr,
                                            weight_decay,
                                            model_name,
                                            bidirection,
                                            dropout,
                                            global_pool
                                        ]
                                        for param, param_name in zip(params, param_names):
                                            metric[param_name] = [param]
                                        # append to big list
                                        all_results.append(pd.DataFrame.from_dict(metric))

    table = pd.concat(all_results).reset_index(drop=True)
    table = table.iloc[table[f'test_{cfg.dataset.metric}'].argsort()]
    print(table.round(6))


if __name__ == "__main__":
    main()
