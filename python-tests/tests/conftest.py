import os

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
