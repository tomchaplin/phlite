"""Write BENCHMARKS.md from pytest-benchmark's JSON results and criterion's results in `target/criterion`.

Usage: python benchmarks/report.py RESULTS.json... [--criterion DIR] [--out FILE]

This only formats results; run them with `just bench-docs` at the root of the repository, which also calls this.
"""

import argparse
import json
import re
from pathlib import Path

from suites import load_suite, suite_name

REPO = Path(__file__).resolve().parent.parent.parent

# Each pytest-benchmark suite is a file `benchmarks/bench_<suite>.py` (e.g. `bench_cloud_rips.py`), presented as one
# section, as its `SUITE` dict says (see `suites.py`)
CRITERION = {
    "title": "Rust: the clearing reduction",
    "description": "The core reduction alone, from criterion (`cargo bench`), on the Rips coboundary matrix of random "
    "points in the unit cube (`phlite_cloud/benches/reduction.rs`) and the GrPPH coboundary matrix of random digraphs "
    "(`phlite_network/benches/reduction.rs`). Median time per reduction; `parallel …/N` uses N threads.",
}


def format_seconds(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"


def python_sections(paths):
    """The header lines (machine, commit) and a section per suite, from pytest-benchmark JSON files."""
    by_suite, machines = {}, []
    for path in paths:
        data = json.loads(Path(path).read_text())
        machine, commit = data["machine_info"], data["commit_info"]
        machines.append((machine, commit, data["datetime"][:10]))
        for b in data["benchmarks"]:
            by_suite.setdefault(suite_name(b), {"python": machine["python_version"], "rows": []})["rows"].append(b)

    out = []
    for suite, info in sorted(by_suite.items(), key=lambda item: (load_suite(item[0])["order"], item[0])):
        meta = load_suite(suite)
        rows = info["rows"]
        summary = meta["summary"] or list(dict.fromkeys(b["params"]["implementation"] for b in rows))
        columns = [c for c in summary if any(b["params"]["implementation"] == c for b in rows)]
        groups = list(dict.fromkeys(b["group"] for b in rows))
        cell = {(b["group"], b["params"]["implementation"]): b for b in rows}
        out += ["", f"## {meta['title']}", "", meta["description"], "", f"Python {info['python']}.", ""]
        out.append("| case | " + " | ".join(columns) + " |")
        out.append("|---|" + "---:|" * len(columns))
        for group in groups:
            cells = []
            for c in columns:
                b = cell.get((group, c))
                cells.append("" if b is None else cell_text(b))
            out.append(f"| {group} | " + " | ".join(cells) + " |")
        out += comparison_table(rows, groups, meta["default"], meta.get("compare_only"))
        out += configurations_table(rows, groups)
    return machines, out


def is_phlite(implementation):
    return implementation.startswith("phlite")


def cell_text(b):
    peak = b["extra_info"].get("peak_mb")
    return format_seconds(b["stats"]["min"]) + ("" if peak is None else f" · {peak:.0f} MB")


def comparison_table(rows, groups, default, only=None):
    """Per case, on one thread: the fastest implementation that isn't phlite, phlite's default, and the fastest phlite
    configuration. With `only`, just the implementations whose name contains it (e.g. timings of the same thing)."""
    scope = "" if only is None else f" Only `({only})` timings are compared, which leave out loading the input."
    out = ["", "How phlite compares on one thread, per case: the fastest other implementation, phlite's default "
           f"(`{default}`), and the fastest phlite configuration (all of them are in the next table)." + scope +
           " Ratios are to the fastest other implementation (below 1 means phlite is faster).", ""]
    out.append("| case | fastest other (1 thread) | phlite default | fastest phlite (1 thread) |")
    out.append("|---|---|---:|---|")

    def comparable(b):
        name = b["params"]["implementation"]
        return "parallel" not in name and (only is None or only in name)

    for group in groups:
        here = [b for b in rows if b["group"] == group and comparable(b)]
        others = [b for b in here if not is_phlite(b["params"]["implementation"])]
        phlites = [b for b in here if is_phlite(b["params"]["implementation"])]
        best_other = min(others, key=lambda b: b["stats"]["min"], default=None)
        best_phlite = min(phlites, key=lambda b: b["stats"]["min"], default=None)
        default_b = next((b for b in here if b["params"]["implementation"] == default), None)

        def ratio(b):
            return "" if best_other is None or b is None else f" ({b['stats']['min'] / best_other['stats']['min']:.2f}×)"

        other_text = "" if best_other is None else f"{best_other['params']['implementation']}: {cell_text(best_other)}"
        default_text = "" if default_b is None else cell_text(default_b) + ratio(default_b)
        best_text = "" if best_phlite is None else (
            f"{best_phlite['params']['implementation']}: {cell_text(best_phlite)}{ratio(best_phlite)}"
        )
        out.append(f"| {group} | {other_text} | {default_text} | {best_text} |")
    return out


def configurations_table(rows, groups):
    """Every phlite configuration (rows) on every case (columns), with the fastest in each case in bold."""
    names = list(dict.fromkeys(b["params"]["implementation"] for b in rows if is_phlite(b["params"]["implementation"])))
    cell = {(b["group"], b["params"]["implementation"]): b for b in rows}
    fastest = {}
    for group in groups:
        here = [cell[(group, n)] for n in names if (group, n) in cell]
        if here:
            fastest[group] = min(here, key=lambda b: b["stats"]["min"])["params"]["implementation"]
    out = ["", "<details><summary>Every phlite configuration (fastest per case in bold)</summary>", ""]
    out.append("| configuration | " + " | ".join(groups) + " |")
    out.append("|---|" + "---:|" * len(groups))
    for name in names:
        cells = []
        for group in groups:
            b = cell.get((group, name))
            text = "" if b is None else cell_text(b)
            cells.append(f"**{text}**" if b is not None and fastest.get(group) == name else text)
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "</details>"]
    return out


def criterion_section(directory):
    """A section from criterion's results: a row per group, a column per benchmark (function and parameter)."""
    results = {}
    for estimates in sorted(Path(directory).glob("*/**/new/estimates.json")):
        bench = json.loads((estimates.parent / "benchmark.json").read_text())
        name = bench["function_id"] + (f"/{bench['value_str']}" if bench["value_str"] else "")
        median = json.loads(estimates.read_text())["median"]["point_estimate"] / 1e9
        results.setdefault(bench["group_id"], {})[name] = median
    if not results:
        return []
    columns = sorted({name for row in results.values() for name in row}, key=criterion_order)
    out = ["", f"## {CRITERION['title']}", "", CRITERION["description"], ""]
    out.append("| case | " + " | ".join(columns) + " |")
    out.append("|---|" + "---:|" * len(columns))
    for group, row in sorted(results.items(), key=lambda item: natural_key(item[0])):
        out.append(f"| {group} | " + " | ".join(format_seconds(row[c]) if c in row else "" for c in columns) + " |")
    return out


def natural_key(text):
    """Sort with numbers in order, so that n=50 comes before n=100."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def criterion_order(name):
    """Serial first, then parallel locking and lock-free storage, each by number of threads."""
    function, _, threads = name.partition("/")
    kinds = [
        "serial merged cached",
        "serial merged uncached",
        "serial heap cached",
        "serial heap uncached",
        "serial sorted cached",
        "serial sorted uncached",
        "serial merged store-r",
        "parallel locking",
        "parallel lock-free",
    ]
    return (kinds.index(function) if function in kinds else len(kinds), function, int(threads) if threads.isdigit() else 0)


def report(paths, criterion):
    machines, python = python_sections(paths)
    machine, commit, date = machines[0]
    commit_str = commit.get("id", "?")[:7] + (" (with uncommitted changes)" if commit.get("dirty") else "")
    threads = {b for p in paths for b in [json.loads(Path(p).read_text())["benchmarks"][0]["extra_info"].get("threads")]}
    out = [
        "# Benchmarks",
        "",
        "<!-- Generated by `just bench-docs` at the root of the repository; don't edit by hand. -->",
        "",
        "phlite's Python bindings against reference implementations (with "
        "[pytest-benchmark](https://pytest-benchmark.readthedocs.io/)), and the Rust reduction on its own (with "
        "[criterion](https://bheisler.github.io/criterion.rs/book/)), on one machine.",
        "In the Python tables each time is the fastest of several runs (one run for the largest cases), with the peak "
        "heap memory allocated during a separate run in MB (measured with [memray](https://bloomberg.github.io/memray/), "
        "which includes native allocations, such as Rust's). "
        f"Parallel implementations use {', '.join(map(str, sorted(t for t in threads if t)))} "
        "threads. Every barcode agrees with the reference implementation's (each benchmark checks this). An empty cell "
        "is a case not run (as each section says), or one whose first run went over the time limit (10 minutes, by "
        "default) and was stopped.",
        "",
        f"* Machine: {machine['cpu'].get('brand_raw', machine['processor'])} ({machine['cpu'].get('count', '?')} "
        f"cores), {machine['system']} {machine['machine']}",
        f"* Measured {date} at commit `{commit_str}`",
        "",
        "Full results are in `python-tests/benchmarks/results/` (pytest-benchmark JSON; compare runs with "
        "`pytest-benchmark compare`). See [python-tests/README.md](README.md#benchmarks) for how to run them and check "
        "for regressions.",
    ]
    return "\n".join(out + python + criterion_section(criterion)) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="pytest-benchmark JSON files")
    parser.add_argument("--criterion", default=str(REPO / "target" / "criterion"), help="criterion's output directory")
    parser.add_argument("--out", help="write here instead of stdout")
    args = parser.parse_args()
    text = report(args.results, args.criterion)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
