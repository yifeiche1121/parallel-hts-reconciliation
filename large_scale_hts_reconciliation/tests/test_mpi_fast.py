import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict
import mpi4py

mpi4py.rc.threaded = True
# mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import Distributed
import sys
import numpy as np
from timeit import default_timer as timer

DATASETS = ["TourismSmall"]

hierarchy_prefix = {"TourismSmall": "tourism"}
num_leaves = {"TourismSmall": 56}
num_nodes = {"TourismSmall": 89}
num_levels = {"TourismSmall": 4}


ROOT = "/home/yifeiche/parallel-hts-reconciliation"  # "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "/notebooks/"

S_compacts = {}
top_down_ps = {}
level_2_ps = {}
gts = {}
yhats = {}

for DATA_ROOT in DATASETS:
    S_compact = np.load(open(data_dir + DATA_ROOT + "/parent.npy", "rb"))
    top_down_p = np.load(open(data_dir + DATA_ROOT + "/top_down_tensor.npy", "rb"))[
        :, 0
    ].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + DATA_ROOT + "/level_2_tensor.npy", "rb"))[
        :, 0
    ].reshape(-1, 1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gt = np.load(
        open(data_dir + DATA_ROOT + "/mpi/gt_tensor_" + str(rank) + ".npy", "rb")
    )
    yhat = np.load(
        open(data_dir + DATA_ROOT + "/mpi/yhat_tensor_" + str(rank) + ".npy", "rb")
    )

    S_compacts[DATA_ROOT] = S_compact
    top_down_ps[DATA_ROOT] = top_down_p
    level_2_ps[DATA_ROOT] = level_2_p
    gts[DATA_ROOT] = gt
    yhats[DATA_ROOT] = yhat

methods = ["bottom_up", "top_down", "middle_out"]
modes = ["dp_matrix", "dp_optimized", "gather"]


def run_bottom_up(mode, dataset):
    comm = MPI.COMM_WORLD
    distrib = Distributed()
    if mode == "gather":
        result = distrib.reconcile_gather(
            "bottom_up",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "dp_matrix":
        result = distrib.reconcile_dp_matrix(
            "bottom_up",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "dp_optimized":
        result = distrib.reconcile_dp_optimized(
            "bottom_up",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    comm.Barrier()
    return result


def run_top_down(mode, dataset):
    comm = MPI.COMM_WORLD
    distrib = Distributed()
    if mode == "gather":
        result = distrib.reconcile_gather(
            "top_down",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "dp_matrix":
        result = distrib.reconcile_dp_matrix(
            "top_down",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "dp_optimized":
        result = distrib.reconcile_dp_optimized(
            "top_down",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    comm.Barrier()
    return result


def run_middle_out(mode, dataset):
    comm = MPI.COMM_WORLD
    distrib = Distributed()
    if mode == "gather":
        result = distrib.reconcile_gather(
            "middle_out",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            level_2_ps[dataset],
            2,
            1.5,
        )
    elif mode == "dp_matrix":
        result = distrib.reconcile_dp_matrix(
            "middle_out",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            level_2_ps[dataset],
            2,
            1.5,
        )
    elif mode == "dp_optimized":
        result = distrib.reconcile_dp_optimized(
            "middle_out",
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            level_2_ps[dataset],
            2,
            1.5,
        )
    comm.Barrier()
    return result


d = defaultdict(lambda: defaultdict(dict))


@pytest.mark.mpi()
def test_ping():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print("Hello world from rank 0")
    else:
        print("Hello world from rank " + str(rank))
    comm.Barrier()


@pytest.mark.mpi()
@pytest.mark.parametrize(
    "mode,method,dataset", itertools.product(modes, methods, DATASETS)
)
@pytest.mark.benchmark(
    min_rounds=1,
    max_time=1,
)
def test_mpi(benchmark, mode, method, dataset):

    benchmark.group = method + "/" + dataset

    def f():
        if method == "bottom_up":
            result = run_bottom_up(mode, dataset)
        elif method == "middle_out":
            result = run_middle_out(mode, dataset)
        elif method == "top_down":
            result = run_top_down(mode, dataset)
        return result

    result = benchmark.pedantic(f, iterations=1, rounds=20)

    d[dataset][method][mode] = result
    for (i, j) in itertools.combinations(d[dataset][method].values(), 2):
        assert np.allclose(i, j, rtol=1e-3, atol=1e-5)

    return
