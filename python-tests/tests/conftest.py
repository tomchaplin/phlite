import os

import pytest
from hypothesis import HealthCheck, settings

# `just test`: a quick run. `just fuzz N`: N examples per test with no time limit per example.
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "fuzz",
    max_examples=int(os.environ.get("FUZZ_EXAMPLES", "5000")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def time_limit():
    """Seconds each test may take before it counts as hung (`PHLITE_TEST_TIMEOUT` overrides): generous, and scaled
    with the number of Hypothesis examples."""
    if "PHLITE_TEST_TIMEOUT" in os.environ:
        return float(os.environ["PHLITE_TEST_TIMEOUT"])
    return 300 + 0.5 * settings.default.max_examples


def pytest_collection_modifyitems(config, items):
    """Give every test (not the benchmarks, which time-limit themselves) a time limit, so that a bug that makes a
    reduction loop forever fails the run instead of hanging it. The `thread` method, since a signal can't interrupt
    Rust code running without the GIL: on timeout it prints every thread's stack and exits."""
    timeout = time_limit()
    for item in items:
        if "benchmarks" not in item.path.parts and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(timeout, method="thread"))
