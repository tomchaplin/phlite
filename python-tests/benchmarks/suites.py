"""The benchmark suites, for report.py and light_report.py.

Each suite is a file `benchmarks/bench_<suite>.py`, named for what it benchmarks: `bench_clouds_rips.py` is the suite
`clouds_rips`. A suite describes itself in a module-level `SUITE` dict of plain literals, read here without importing
the module (whose imports need the references installed):

    SUITE = {
        "title": "...",               # the section's heading in BENCHMARKS.md
        "description": "...",         # the paragraph under it
        "order": 10,                  # sections are sorted by this, then by name
        "summary": ["ripser", ...],   # the implementations in the section's summary table, in order (default: all)
        "default": "phlite (serial)", # phlite's default, compared with the fastest other implementation
        "compare_only": "reduce",     # optional: only compare implementations whose name contains this
        "light_references": [...],    # the references the light CI benchmark runs, in order of preference
    }

A new suite needs nothing else: the justfile picks up `bench_<area>_*.py` files, and the reports read their `SUITE`.
"""

import ast
import functools
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent

DEFAULTS = {
    "description": "",
    "order": 50,
    "summary": [],
    "default": "phlite (serial)",
    "light_references": [],
}


def suite_name(benchmark):
    """The suite of a benchmark from pytest-benchmark's JSON: its file's name without `bench_`."""
    return Path(benchmark["fullname"].split("::")[0]).stem.removeprefix("bench_")


@functools.cache
def load_suite(name):
    """The `SUITE` dict of `bench_<name>.py`, with defaults filled in (a file without one, or a suite whose file is
    gone, gets a plain section)."""
    suite = dict(DEFAULTS, title=name)
    path = BENCHMARKS / f"bench_{name}.py"
    if path.exists():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SUITE" for t in node.targets):
                suite.update(ast.literal_eval(node.value))
    return suite
