# Tests and benchmarks for the whole workspace. Requires `just`, `uv` and a C++ compiler (for PHAT).
# The Python side lives in python-tests/, see its justfile and README.

set shell := ["bash", "-euo", "pipefail", "-c"]
# Pass recipe arguments on as "$@", keeping their quoting
set positional-arguments

# List available recipes
default:
    @just --list

# Run every test: the Rust workspace, then the Python fuzz tests and phlite_py's tests
test:
    cargo test --workspace --all-features
    cd python-tests && just test

# Fuzz for longer, e.g. `just fuzz 20000`: that many cases per Rust property test (proptest), then examples per
# Python test (Hypothesis)
fuzz examples="5000":
    PROPTEST_CASES="$1" cargo test --release --workspace --all-features
    cd python-tests && just fuzz "$1"

# Build the documentation site into docs/site: the book (docs/book, mdBook) at its root, the Python API reference
# (docs/python, Sphinx) in docs/site/api, and the example notebooks (docs/notebooks, marimo) in docs/site/notebooks.
# Needs `mdbook` on your PATH; the notebooks run in python-tests' two environments (`just venv` and `just build` first).
docs:
    cd python-tests && uv sync --inexact --group docs && uv sync --inexact --project latest --group docs
    cd python-tests && uv run --no-sync --project latest python ../docs/figures/make_figures.py
    mdbook build docs/book
    cd python-tests && uv run --no-sync --project latest sphinx-build -q -b html ../docs/python ../docs/site/api
    mkdir -p docs/site/notebooks
    cd python-tests && for notebook in phlite_image phlite_delaunay; do \
        uv run --no-sync --project latest marimo export html ../docs/notebooks/$notebook.py \
            -o ../docs/site/notebooks/$notebook.html; done
    cd python-tests && for notebook in rips_points grpph_digraph; do \
        uv run --no-sync marimo export html ../docs/notebooks/$notebook.py \
            -o ../docs/site/notebooks/$notebook.html; done
    @echo "Built docs/site (open docs/site/index.html, or run just docs-serve)"

# Serve the built site (run `just docs` first) at http://localhost:8000
docs-serve:
    python3 -m http.server --directory docs/site 8000

# The criterion benchmarks of the Rust reduction (in phlite_rips and phlite_grpph)
cargo_bench := "cargo bench -p phlite_rips -p phlite_grpph --features phlite_rips/lock-free --bench reduction"

# Run every benchmark: Rust (criterion), then Python (pytest-benchmark)
bench: bench-rust bench-py

# Run the Rust benchmarks, with criterion options, e.g. `just bench-rust 'rips n=100'`
bench-rust *args:
    {{cargo_bench}} -- "$@"

# Run the Python benchmarks, with pytest options, e.g. `just bench-py -k grid`
bench-py *args:
    cd python-tests && just bench "$@"

# Check for performance regressions against the saved baselines (only the Python side fails on one; criterion reports)
bench-check:
    cd python-tests && just bench-check
    {{cargo_bench}} -- --baseline-lenient quick

# Save new baselines for `bench-check` (after an intended change in performance, or on a new machine)
bench-baseline:
    cd python-tests && just bench-baseline
    {{cargo_bench}} -- --save-baseline quick

# Rerun every benchmark and regenerate python-tests/BENCHMARKS.md
bench-docs:
    {{cargo_bench}}
    cd python-tests && just bench-docs

# Everything a change should pass: all tests, then the regression check
check: test bench-check
