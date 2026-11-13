"""Summarise a light benchmark run (`just bench-light`) for a pull request comment.

    python benchmarks/light_report.py RESULTS_DIR [--baseline BASELINE_DIR] --out report.md

RESULTS_DIR (and BASELINE_DIR, from the latest run on main) hold pytest-benchmark JSON files. Shared CI machines vary
from run to run, so absolute times are only a rough guide; what is reported is each phlite time *divided by a
reference implementation's time in the same run* (ripser, grpphati, PHAT), which is far more stable. Against the
baseline, a ratio that grows by more than `--threshold` (default 25%) is flagged. Nothing fails.
"""

import argparse
import glob
import json
import os

# The reference for each case: the first of these that was run
REFERENCES = ["ripser", "giotto-ph (1 thread)", "grpphati (GrPPH)", "phat (twist)"]
PHLITE = ["phlite (serial)", "phlite (parallel)"]


def load(directory):
    """{case: {implementation: fastest time in seconds}} from every JSON file in `directory`."""
    times = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        for bench in json.load(open(path))["benchmarks"]:
            implementation = bench["params"]["implementation"]
            times.setdefault(bench["group"], {})[implementation] = bench["stats"]["min"]
    return times


def ratios(times):
    """{(case, phlite implementation): (reference, time / reference's time)}."""
    out = {}
    for case, by_impl in times.items():
        reference = next((r for r in REFERENCES if r in by_impl), None)
        if reference is None:
            continue
        for implementation in PHLITE:
            if implementation in by_impl:
                out[(case, implementation)] = (reference, by_impl[implementation] / by_impl[reference])
    return out


def duration(seconds):
    return f"{seconds * 1e3:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("--baseline")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    times = load(args.results)
    now = ratios(times)
    base = ratios(load(args.baseline)) if args.baseline and os.path.isdir(args.baseline) else {}

    lines = [
        "## Light benchmark",
        "",
        "phlite's default algorithms on the quick cases, as a multiple of a reference implementation's time **in the "
        "same run** (below 1× means phlite is faster). Shared CI machines are noisy, so treat small changes as noise.",
        "",
    ]
    if base:
        lines += ["| case | algorithm | time | × reference | on main | change |", "|---|---|---:|---:|---:|---:|"]
    else:
        lines += ["| case | algorithm | time | × reference |", "|---|---|---:|---:|"]
    flagged = 0
    for (case, implementation), (reference, ratio) in sorted(now.items()):
        row = [case, implementation.removeprefix("phlite "), duration(times[case][implementation]),
               f"{ratio:.2f}× {reference}"]
        if base:
            if (case, implementation) in base:
                before = base[(case, implementation)][1]
                change = ratio / before - 1
                warn = change > args.threshold
                flagged += warn
                row += [f"{before:.2f}×", f"{'⚠️ ' if warn else ''}{change:+.0%}"]
            else:
                row += ["", "new"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if not base:
        lines.append("_No baseline from main yet (it is saved by each run on main)._")
    elif flagged:
        lines.append(f"⚠️ {flagged} ratio(s) grew by more than {args.threshold:.0%} relative to main.")
    else:
        lines.append(f"No ratio grew by more than {args.threshold:.0%} relative to main.")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
