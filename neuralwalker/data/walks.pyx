# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3


cimport numpy as np
import numpy as np
np.import_array()

ctypedef np.int64_t int64_t
ctypedef np.uint32_t uint32_t


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)

    return seed[0] % (<uint32_t>RAND_R_MAX + 1)

cdef inline uint32_t rand_int(uint32_t end, uint32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef inline int fmin(int x, int y) nogil:
    if x < y:
        return x
    return y


def get_random_walks(
    csr_matrix,
    int walk_length,
    double sample_rate=1.0,
    int pad_value=-1,
    bint backtracking=False,
    bint strict=False,
    int window_size=0,
    object rng=None
):
    if rng is None:
        rng = np.random.RandomState(0)

    cdef:
        int[:] indices = csr_matrix.indices
        int[:] indptr = csr_matrix.indptr
        int num_nodes = csr_matrix.shape[0]
        int num_active_nodes = num_nodes

        int[:] neighbors = np.empty(num_nodes, dtype=np.int32)

        int i, j, k, node_index, choice_index, next_index, prev_index
        int edge_index, repeated_index
        int index, neighbors_i
        int walk_index = 0

        int[:] degrees = np.asarray(csr_matrix.sum(axis=-1), dtype=np.int32).flatten()
        int degree

        bint use_node_mask = sample_rate < 1.0
        np.uint8_t[:] active_nodes

        uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
        uint32_t* rand_r_state = &rand_r_state_seed

    if use_node_mask:
        active_nodes = (rng.rand(num_nodes) < sample_rate).astype(np.uint8)
        num_active_nodes = np.asarray(active_nodes).sum()

    cdef int64_t[:, ::1] walk_node_index = np.full([num_active_nodes, walk_length], pad_value, dtype=np.int64)
    cdef int64_t[:, ::1] walk_edge_index = np.full([num_active_nodes, walk_length], pad_value, dtype=np.int64)
    cdef np.uint8_t[:, :, ::1] walk_node_id_encoding
    cdef np.uint8_t[:, :, ::1] walk_node_adj_encoding
    cdef int o, s, adj_prev_node_index
    if window_size > 0:
        walk_node_id_encoding = np.zeros([num_active_nodes, walk_length, window_size], dtype=np.uint8)
        walk_node_adj_encoding = np.zeros([num_active_nodes, walk_length, window_size - 1], dtype=np.uint8)

    with nogil:
        for i in range(num_nodes):
            if use_node_mask and not active_nodes[i]:
                continue
            node_index = i
            prev_index = -1
            walk_node_index[walk_index, 0] = node_index
            for j in range(walk_length - 1):
                degree = degrees[node_index]
                if degree == 0: # stop walk
                    break
                edge_index = indptr[node_index]
                neighbors_i = 0
                repeated_index = -1
                for k in range(indptr[node_index], indptr[node_index + 1]):
                    index = indices[k]
                    if backtracking or (index != prev_index):
                        neighbors[neighbors_i] = index
                        neighbors_i += 1
                    else:
                        degree -= 1
                        repeated_index = neighbors_i
                if degree == 0 and strict: # stop walk
                    break

                if degree == 0:
                    next_index = prev_index
                else:
                    choice_index = rand_int(<uint32_t> degree, rand_r_state)

                    next_index = neighbors[choice_index]
                    edge_index += choice_index
                    if (repeated_index != -1) and (choice_index >= repeated_index):
                        edge_index += 1

                walk_node_index[walk_index][j + 1] = next_index
                walk_edge_index[walk_index][j] = edge_index

                prev_index = node_index
                node_index = next_index

                if window_size > 0:
                    o = fmin(window_size, j + 1)
                    for s in range(o):
                        adj_prev_node_index = walk_node_index[walk_index][j - s]
                        walk_node_id_encoding[walk_index][j + 1][window_size - 1 - s] = next_index == adj_prev_node_index
                        if s > 0:
                            for k in range(indptr[adj_prev_node_index], indptr[adj_prev_node_index + 1]):
                                if indices[k] > next_index:
                                    break
                                if indices[k] == next_index:
                                    walk_node_adj_encoding[walk_index][j + 1][window_size - 1 - s] = 1
                                    break
            walk_index += 1

    if window_size > 0:
        return np.asarray(walk_node_index), np.asarray(walk_edge_index), np.asarray(walk_node_id_encoding), np.asarray(walk_node_adj_encoding)

    return np.asarray(walk_node_index), np.asarray(walk_edge_index)
