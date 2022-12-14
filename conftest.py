import sqlite3

def pytest_configure(config):
    sqlite3.connect(':memory:')
    config.pluginmanager.register(
        'memory_profiler',
        'pytest_monitor.plugin.MemoryProfiler',
        '--memory-profile'
    )
