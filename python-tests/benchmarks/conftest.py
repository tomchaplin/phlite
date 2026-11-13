"""pytest-benchmark configuration for the benchmarks in this folder (the helpers they share are in `common.py`)."""


def pytest_benchmark_update_machine_info(config, machine_info):
    """Leave out what shouldn't be committed with the results, or changes from run to run on the same machine.

    The hostname isn't needed, and the CPU's current clock speed varies between runs, which would make pytest-benchmark
    warn that the machine differs from the baseline's every time.
    """
    machine_info.pop("node", None)
    for key in ["hz_actual", "hz_actual_friendly"]:
        machine_info.get("cpu", {}).pop(key, None)


def pytest_collection_modifyitems(config, items):
    """Run only some implementations (the references still compute the expected barcodes):
    * with `PHLITE_BENCH_ONLY` set to implementation names separated by `|` (e.g. `phlite (serial)|ripser`), those;
    * with `PHLITE_BENCH_LIGHT` set (the light benchmark for CI), `phlite (serial)`, `phlite (parallel)` and the
      references in each suite's `SUITE["light_references"]` (see suites.py)."""
    import os

    only = os.environ.get("PHLITE_BENCH_ONLY")
    light = os.environ.get("PHLITE_BENCH_LIGHT")
    if not only and not light:
        return

    def wanted(item):
        if only:
            return set(only.split("|"))
        suite = getattr(item.module, "SUITE", {})
        return {"phlite (serial)", "phlite (parallel)", *suite.get("light_references", [])}

    keep, drop = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        implementation = callspec.params.get("implementation") if callspec else None
        (keep if implementation in wanted(item) else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
