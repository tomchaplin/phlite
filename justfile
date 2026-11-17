# Tests and benchmarks for the whole workspace. Requires `just`, `uv` and a C++ compiler (for PHAT).
# The Python side lives in python-tests/, see its justfile and README.

set shell := ["bash", "-euo", "pipefail", "-c"]
# Pass recipe arguments on as "$@", keeping their quoting
set positional-arguments

# List available recipes
default:
    @just --list

# Run every test: the Rust workspace, then the Python fuzz tests and phlite_complexes' tests
test:
    cargo test --workspace --all-features
    cd python-tests && just test

# Debug builds pile up: every rebuild of a test or example keeps its own copy (a few hundred MB each), and
# incremental-compilation state grows too, so after a busy day target/debug can hold well over 100 GB of files (much less
# on disk if the filesystem compresses them: about 12 GB under btrfs's zstd, where `du` shows the uncompressed size,
# `compsize` the real one). This removes it
# (the next `just test` rebuilds what it needs), and the PGO builds; release builds (the Python wheels') are kept.
# `cargo clean` removes everything.
#
# Free the disk space taken by debug and PGO builds
clean:
    cargo clean --profile dev
    rm -rf target/pgo-generate target/pgo-use pgo

# Check that the impls forwarding a trait to an inner matrix (`&M`, `WithTranspose`, `MatrixWithBasis`, ...) forward
# every method, those with defaults included: a missed one would silently drop the inner matrix's override (a
# performance hook, say). They are marked `#[deny(clippy::missing_trait_methods)]`; every other lint is allowed here.
lint-forwarding:
    cargo clippy -p phlite --all-features --all-targets -- -A clippy::all

# Fuzz for longer, e.g. `just fuzz 20000`: that many cases per Rust property test (proptest), then examples per
# Python test (Hypothesis)
fuzz examples="5000":
    PROPTEST_CASES="$1" cargo test --release --workspace --all-features
    cd python-tests && just fuzz "$1"

# Every phlite Python package's crate, phlite_bindings (phlite-core) first: the others depend on it
packages := "phlite_bindings phlite_complexes phlite_clouds phlite_networks phlite_images"

# Build the phlite packages the docs' notebooks use, into python-tests' latest environment (with its docs
# dependencies; unlike `just build` there, this doesn't need PHAT)
docs-modules:
    cd python-tests && uv sync --inexact --project latest --group docs
    cd python-tests && for crate in {{packages}}; do \
        uv run --no-sync --project latest maturin develop --uv --release -m ../$crate/Cargo.toml; done

# Regenerate every package's type stubs (<crate>/python/phlite/<module>/__init__.pyi)
stubs:
    cargo run -p phlite_bindings --features stub-gen --bin phlite_bindings_stub_gen
    for crate in phlite_complexes phlite_clouds phlite_networks phlite_images; do \
        cargo run -p $crate --features stub-gen --bin ${crate}_stub_gen; done

# Build the documentation site into docs/site:
# * the landing page (docs/landing) at its root;
# * the book (docs/book, mdBook) in docs/site/book;
# * the Python API reference (docs/python, Sphinx) in docs/site/api;
# * the example notebooks (docs/notebooks, marimo) in docs/site/notebooks;
# * the playground (phlite_web: phlite in WebAssembly, and a Svelte app) in docs/site/playground.
# Needs `mdbook` on your PATH, the notebooks run in python-tests' latest environment (`just docs-modules` first), and the
# playground needs what phlite_web/justfile says (a nightly Rust, wasm-bindgen-cli and Node).
# CI builds and deploys it to Netlify (.github/workflows/docs.yml).
#
# Build the whole documentation site into docs/site
docs: docs-pages docs-playground
    @echo "Built docs/site (run just docs-serve to look at it)"

# Everything but the playground (see `docs`)
docs-pages:
    mkdir -p docs/site
    find docs/site -mindepth 1 -maxdepth 1 ! -name playground -exec rm -rf {} +
    cp docs/landing/index.html docs/landing/_headers docs/landing/favicon.svg docs/site/
    cd python-tests && uv sync --inexact --project latest --group docs
    cd python-tests && uv run --no-sync --project latest python ../docs/figures/make_figures.py
    mdbook build docs/book
    cd python-tests && uv run --no-sync --project latest sphinx-build -q -b html ../docs/python ../docs/site/api
    mkdir -p docs/site/notebooks
    cd python-tests && for notebook in ../docs/notebooks/*.py; do \
        uv run --no-sync --project latest marimo export html "$notebook" \
            -o "../docs/site/notebooks/$(basename "$notebook" .py).html"; done

# The playground (see `docs`)
docs-playground:
    just phlite_web/build
    rm -rf docs/site/playground
    mkdir -p docs/site
    cp -r phlite_web/app/dist docs/site/playground

# The site deploys from the `release` branch (.github/workflows/docs.yml), so this fast-forwards it to main's latest
# commit on GitHub (`ref` to publish another, e.g. a tag). On GitHub, the Docs workflow's Run workflow button does the
# same.
#
# Publish the docs to phlite.tomchaplin.xyz
publish-docs ref="origin/main":
    git fetch origin
    git push origin "$1:refs/heads/release"

# Serve the built site (run `just docs` first) at http://localhost:8000, with the headers the playground needs
docs-serve:
    python3 docs/serve.py 8000

# The criterion benchmarks of the Rust reduction (in phlite_clouds and phlite_networks)
cargo_bench := "cargo bench -p phlite_clouds -p phlite_networks --features phlite_clouds/lock-free --bench reduction"

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

# Releases: every crate and every Python package share one version, `[workspace.package] version` in Cargo.toml. Bump
# it there, in the exact pins of `[workspace.dependencies]` just below it, and in phlite_meta/pyproject.toml (its version
# and pins) and the `phlite-core==` pin in each package's pyproject.toml (`cargo test -p phlite_bindings` checks them
# all). Then `just release-check`, and push a tag `vX.Y.Z`: .github/workflows/release.yml builds every wheel and the
# metapackage and publishes them to PyPI, and publishes every crate to crates.io (running that workflow by hand
# builds and checks everything without publishing). Publish the docs too (`just publish-docs`).
#
# As the release workflow does first: each crate is packaged and built against the others as they will be published, in
# dependency order, uploading nothing.
#
# Check that every crate can be published to crates.io
release-check:
    cargo publish --workspace --dry-run --allow-dirty

# Build a release wheel of every package, and the metapackage `phlite`, into dist/ (for this machine's Python)
wheels:
    for crate in {{packages}}; do (cd $crate && uvx maturin build --release --out ../dist); done
    uv build --wheel --out-dir dist phlite_meta

# Profile-guided optimisation (PGO, about 4% faster overall and 15-20% on GrPPH and PPH in our benchmarks): build every
# package instrumented (`-Cprofile-generate`) into a scratch environment, run python-tests/pgo_train.py there, and
# merge the profiles into pgo/merged.profdata. Needs uv and the `llvm-tools` rustup component of the toolchain that
# will use the profile (`rustup component add llvm-tools`). The released Linux x86_64 wheels are built this way (see
# .github/workflows/release.yml); on any other platform, `just pgo-wheels` builds PGO wheels for it. On Windows, run
# it from Git Bash.
pgo-profile:
    #!/usr/bin/env bash
    set -euo pipefail
    # A native path (on Windows, C:/... rather than Git Bash's /c/...), for rustc and LLVM
    here=$(pwd -W 2>/dev/null || pwd)
    rm -rf pgo && mkdir -p pgo/profiles
    uv venv -q pgo/venv
    export VIRTUAL_ENV="$here/pgo/venv"
    uv pip install -q numpy
    for crate in {{packages}}; do
        RUSTFLAGS="-Cprofile-generate=$here/pgo/profiles" CARGO_TARGET_DIR=target/pgo-generate \
            uvx maturin develop -q --uv --release -m $crate/Cargo.toml
    done
    python=pgo/venv/bin/python
    [ -x "$python" ] || python=pgo/venv/Scripts/python.exe
    "$python" python-tests/pgo_train.py
    host=$(rustc -vV | sed -n 's/^host: //p')
    profdata="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata"
    [ -x "$profdata" ] || profdata="$profdata.exe"
    "$profdata" merge -o pgo/merged.profdata pgo/profiles
    echo "Wrote pgo/merged.profdata"

# Build a PGO-optimised release wheel of every package into dist/ for this machine (and its Python), profiling it
# first: `pip install dist/*.whl` then installs them (see the book's appendix, "Profile-guided optimisation")
pgo-wheels: pgo-profile
    #!/usr/bin/env bash
    set -euo pipefail
    here=$(pwd -W 2>/dev/null || pwd)
    for crate in {{packages}}; do
        (cd $crate && RUSTFLAGS="-Cprofile-use=$here/pgo/merged.profdata" CARGO_TARGET_DIR=../target/pgo-use \
            uvx maturin build --release --out ../dist)
    done
    uv build --wheel --out-dir dist phlite_meta
