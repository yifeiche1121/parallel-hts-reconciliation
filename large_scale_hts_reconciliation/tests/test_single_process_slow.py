import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict

DATASETS = ["m5_hobbies"] #, "m5_full"]

num_leaves = {"m5_hobbies": 5650, "m5_full": 30490}
num_nodes = {"m5_hobbies": 6218, "m5_full": 33549}
num_levels = {"m5_hobbies": 4, "m5_full": 4}


ROOT = "/home/peiyuan20013/large-scale-hts-reconciliation/large_scale_hts_reconciliation"  # "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "/notebooks/"

S_compacts = {}
top_down_ps = {}
level_2_ps = {}
gts = {}
yhats = {}

for DATA_ROOT in DATASETS:
    S_compact = np.load(open(data_dir + DATA_ROOT + "/m5_hierarchy_parent.npy", "rb"))
    top_down_p = np.load(open(data_dir + DATA_ROOT + "/top_down_tensor.npy", "rb"))[
        :, 0
    ].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + DATA_ROOT + "/level_2_tensor.npy", "rb"))[
        :, 0
    ].reshape(-1, 1)

    yhat = np.load(open(data_dir + DATA_ROOT + "/pred_tensor.npy", "rb"))
    gt = np.load(open(data_dir + DATA_ROOT + "/gt_tensor.npy", "rb"))

    S_compacts[DATA_ROOT] = S_compact
    top_down_ps[DATA_ROOT] = top_down_p
    level_2_ps[DATA_ROOT] = level_2_p
    gts[DATA_ROOT] = gt
    yhats[DATA_ROOT] = yhat

methods = ["OLS", "WLS"]
modes = ["dense_algo", "sparse_algo", "dense_matrix", "sparse_matrix"]


def run(mode, method, dataset):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix(
            method, S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 1.5
        )
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo(
            method, S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 1.5
        )
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix(
            method, S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 1.5
        )
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo(
            method, S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 1.5
        )


d = defaultdict(dict)


@pytest.mark.parametrize("mode,method,dataset", itertools.product(modes, methods, DATASETS))
@pytest.mark.benchmark(
    min_rounds=1,
    max_time=10,
)
@pytest.mark.mpi_skip()
def test_single_process_slow(benchmark, mode, method, dataset):
    benchmark.group = method
    result = benchmark.pedantic(run(mode, method, dataset), iterations=1, rounds=1)

    d[method][mode] = result
    for (i, j) in itertools.combinations(d[method].values(), 2):
        assert np.allclose(i, j, rtol=1e-3, atol=1e-5)
