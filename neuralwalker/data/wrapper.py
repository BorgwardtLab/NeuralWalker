import numpy as np
from torch_geometric import utils
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from .walks import get_random_walks


def sample_random_walks(
    edge_index,
    num_nodes=None,
    length=50,
    sample_rate=1.,
    backtracking=False,
    strict=False,
    window_size=0,
    pad_value=-1,
    rng=None,
):
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    csr_matrix = utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).astype(np.int32).tocsr()
    return get_random_walks(csr_matrix, length, sample_rate, pad_value, backtracking, strict, window_size, rng)
