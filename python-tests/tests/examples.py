"""How many Hypothesis examples a test runs: `examples(n)` is `n` in a quick run (`just test`), and in a fuzz run
(`just fuzz N`) `n / 100` times `N`, at least `n`. For tests that pin their number of examples (because each is slow,
or because a quick run wants more): pinning `max_examples` directly would override the fuzz profile, so that `just
fuzz` would never run them any harder."""

import os


def examples(n: int) -> int:
    if os.environ.get("HYPOTHESIS_PROFILE") != "fuzz":
        return n
    fuzz = int(os.environ.get("FUZZ_EXAMPLES", "5000"))
    return max(n, fuzz * n // 100)
