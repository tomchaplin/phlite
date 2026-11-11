"""pytest-benchmark configuration for the benchmarks in this folder (the helpers they share are in `common.py`)."""


def pytest_benchmark_update_machine_info(config, machine_info):
    """Leave out what shouldn't be committed with the results, or changes from run to run on the same machine.

    The hostname isn't needed, and the CPU's current clock speed varies between runs, which would make pytest-benchmark
    warn that the machine differs from the baseline's every time.
    """
    machine_info.pop("node", None)
    for key in ["hz_actual", "hz_actual_friendly"]:
        machine_info.get("cpu", {}).pop(key, None)
