from mpi4py import MPI

def pytest_configure(config):
    if MPI.COMM_WORLD.rank == 0:
        config.pluginmanager.register(
            'memory_profiler',
            'pytest_monitor.plugin.MemoryProfiler',
            '--memory-profile'
        )
