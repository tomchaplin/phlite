"""A tiny benchmark harness: each case runs in a fresh subprocess so that peak memory can be measured per case.

A benchmark script defines `IMPLEMENTATIONS` (name -> function(input, case) returning a barcode fingerprint),
`make_input(case)` and `case_label(case)`, then calls `main(...)`.
When re-invoked with `--worker <json>` the script runs a single case and prints a JSON result.
"""

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time


def _proc_status_mb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024  # reported in kB
    raise KeyError(field)


def reset_peak_rss():
    """Start measuring peak memory from now, returning the current memory use in MB.

    On Linux we reset the kernel's high-water mark (VmHWM), so the peak is exactly that of the computation.
    Elsewhere we can only use the lifetime peak, so runs that stay below the peak reached during setup report ~0.
    """
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return _proc_status_mb("VmRSS")
    except OSError:
        return peak_rss_mb()


def peak_rss_mb():
    try:
        return _proc_status_mb("VmHWM")
    except OSError:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def fingerprint(dims):
    """Summarise a barcode (a list, per dimension, of (birth, death) bars) so implementations can be compared."""
    out = []
    for bars in dims:
        bars = [(b, d) for b, d in bars if b < d]
        finite = [d - b for b, d in bars if d != math.inf]
        out.append([len(finite), len(bars) - len(finite), round(sum(finite), 6)])
    return out


def run_worker(implementations, make_input, case_json):
    case = json.loads(case_json)
    data = make_input(case)
    function = implementations[case["implementation"]]
    baseline = reset_peak_rss()
    start = time.perf_counter()
    result = function(data, case)
    seconds = time.perf_counter() - start
    print(json.dumps({"seconds": seconds, "peak_mb": max(peak_rss_mb() - baseline, 0.0), "fingerprint": result}))


def run_case(case, timeout):
    """Run one case in a subprocess, returning the parsed result or an error string."""
    try:
        proc = subprocess.run(
            [sys.executable, sys.argv[0], "--worker", json.dumps(case)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout ({timeout}s)"}
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return {"error": lines[-1] if lines else f"exit code {proc.returncode}"}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(description, implementations, make_input, case_label, add_arguments, cases_from_args, reference):
    """Parse arguments, run every case `--repeats` times and print a markdown table.

    `reference` names the implementation whose fingerprint the others are checked against.
    """
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_worker(implementations, make_input, sys.argv[2])
        return

    parser = argparse.ArgumentParser(description=description)
    add_arguments(parser)
    parser.add_argument("--repeats", type=int, default=3, help="runs per case; we report the fastest (default 3)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="threads for parallel implementations")
    parser.add_argument("--only", nargs="*", default=None, help="only run implementations containing one of these strings")
    parser.add_argument("--timeout", type=float, default=600, help="seconds before a run is abandoned (default 600)")
    parser.add_argument("--json", help="also write all results to this file")
    parser.add_argument("--skip-reference", action="store_true", help=f"don't run the reference implementation ({reference}), so don't check agreement")
    args = parser.parse_args()

    names = [n for n in implementations if args.only is None or any(s in n for s in args.only)]
    names = [n for n in names if n in default_implementations(args, implementations)]
    if args.skip_reference:
        names = [n for n in names if n != reference]
    elif reference in implementations and reference not in names:
        names.insert(0, reference)

    rows = []
    print(f"| {'case':<24} | {'implementation':<30} | {'time (s)':>9} | {'peak MB':>8} | agrees |")
    print(f"|{'-' * 26}|{'-' * 32}|{'-' * 11}|{'-' * 10}|--------|")
    for base_case in cases_from_args(args):
        expected = None
        for name in names:
            case = dict(base_case, implementation=name, threads=args.threads)
            runs = [run_case(case, args.timeout) for _ in range(args.repeats)]
            ok = [r for r in runs if "error" not in r]
            if not ok:
                row = dict(case, error=runs[0]["error"])
                time_str, mem_str, agrees = runs[0]["error"][:30], "", ""
            else:
                best = min(ok, key=lambda r: r["seconds"])
                row = dict(
                    case,
                    seconds=best["seconds"],
                    median_seconds=statistics.median(r["seconds"] for r in ok),
                    peak_mb=max(r["peak_mb"] for r in ok),
                    fingerprint=best["fingerprint"],
                )
                if name == reference:
                    expected = best["fingerprint"]
                agrees = "" if expected is None else ("yes" if best["fingerprint"] == expected else "**NO**")
                time_str, mem_str = f"{best['seconds']:.3f}", f"{row['peak_mb']:.1f}"
            rows.append(row)
            print(f"| {case_label(base_case):<24} | {name:<30} | {time_str:>9} | {mem_str:>8} | {agrees:>6} |", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


def default_implementations(args, implementations):
    """Implementations whose name starts with '*' only run when --all is given."""
    return [n for n in implementations if getattr(args, "all", False) or not n.startswith("*")]
