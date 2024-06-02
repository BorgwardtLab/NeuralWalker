# NeuralWalker

The repository implements the NeuralWalker in [Pytorch Geometric][1] described in the following paper

>Dexiong Chen, Till Schulz, and Karsten Borgwardt.
[Learning Long Range Dependencies on Graphs via Random Walks][2], Preprint 2024.

**TL;DR**: A novel random-walk based neural architecture for graph representation learning.

![NeuralWalker](images/overview.png)

NeuralWalker samples random walks with a predefined sampling rate and length, then uses advanced sequence models to process them. 
Additionally, local and global message passing can be employed to capture complementary information.
The main components of NeuralWalker are a random walk sampler, and a stack of neural walker blocks (a walk encoder block + a message passing block).
Each walk encoder block has a walk embedder, a sequence layer, and a walk aggregator.

## Installation

NeuralWalker relies on a sequence model to process random walk sequences. If you want to use NeuralWalker with the state-space model [Mamba](https://github.com/state-spaces/mamba),
please consult its installation guideline.

We manage dependencies using [miniconda](https://docs.conda.io/projects/miniconda/en/latest) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```bash
# Replace micromamba with conda if you use conda or miniconda
micromamba env create -f environment.yaml 
micromamba activate neuralwalker
pip install -e .
```

## Reproducing results in the paper

### Train NeuralWalker on Benchmarking GNNs, LRGB, and OGB

All configurations for the experiments are managed by [hydra](https://hydra.cc/), stored in `./config`.

Below you can find the list of experiments conducted in the paper:

- Benchmarking GNNs: zinc, mnist, cifar10, pattern, cluster
- LRGB: pascalvoc, coco, peptides_func, peptides_struct, pcqm_contact
- OGB: ogbg_molpcba, ogbg_ppa, ogbg_code2

```bash
python train.py experiment=zinc

# Running NeuralWalker with a different model architecture
python train.py experiment=zinc experiment/model=conv+vn_3L
```

You can replace `conv+vn_3L` with any model provided in `config/experiment/model`, or a customized model by creating a new one in that folder.

### Train NeuralWalker on node classification tasks

We integrate NeuralWalker with Polynormer, SOTA model for node classifcation. See [node_classifcation](./node_classification) for more details.


### Debug mode

```bash
python train.py mode=debug
```


[1]: https://pytorch-geometric.readthedocs.io/
[2]: 