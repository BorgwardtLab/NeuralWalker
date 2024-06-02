# Run NeuralWalker on node classification tasks

We showcase how to integrate NeuralWalker with existing message-passing based models.
We integrate here NeuralWalker with [Polynormer](https://github.com/cornell-zhang/Polynormer), SOTA model for node classification.

## Reproducing results on heterophilous datasets

To run our model on `pokec`, you need to install `pip install gdown`.

```bash
# running all experiments with full batch training
bash run.sh

# running all experiments with mini-batch training (only required for ogbn-products and pokec)
cd large_graph_exp
bash run_large.sh
```