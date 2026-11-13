# phlite_web: the playground

phlite compiled to WebAssembly, with a [Svelte](https://svelte.dev) app around it. Everything runs in the browser:
the page sends the data to a Web Worker, which runs phlite's reductions on a pool of further workers (threads, via
[wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon)). There is no server-side code.

* `src/lib.rs`: the WebAssembly API, one function per pipeline:
  * `rips` (Vietoris–Rips, by `phlite_rips` with 64-bit simplex indices);
  * `alpha` (the Delaunay alpha filtration from `alphalpha`, reduced as an explicit complex);
  * `grpph` (`phlite_grpph`);
  * `image` (cubical complexes);
  * `complex` (explicit complexes over Z/2, 3, 5 or 7, by `phlite_py`'s explicit complexes).
  Each takes an options object (missing options take the pipeline's Python defaults) and returns the bars, with
  representatives if asked for.
* `app/`: the Svelte + TypeScript app (Vite). `src/lib/worker.ts` owns the WebAssembly module, and
  `src/lib/engine.svelte.ts` talks to it, cancelling a computation by restarting the worker. Each pipeline lives in
  `src/pipelines/<name>/` (its data sources, view and page), and shares the barcode, diagram and options components in
  `src/components/`.

## Building

You need [rustup](https://rustup.rs) (the nightly in `rust-toolchain.toml`, with `rust-src`, is installed on first
use), `wasm-bindgen-cli` at the version pinned in `Cargo.toml` and `justfile`
(`cargo install wasm-bindgen-cli --version 0.2.130 --locked`), and Node with npm.

```sh
just wasm    # build the WebAssembly package into pkg/ (release, optimised with wasm-opt)
just dev     # the app with hot reloading, at http://localhost:5173
just build   # both, into app/dist
just check   # type-check the app
```

The root `just docs` builds the playground into the documentation site, at `docs/site/playground`.

## Why it needs a nightly and special headers

Threads in WebAssembly share one memory, which needs the standard library rebuilt with atomics (`build-std`, nightly
only; see `.cargo/config.toml`). Browsers only allow shared memory on *cross-origin isolated* pages, served with

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The Vite dev server sends these, as do `docs/landing/_headers` (on Netlify) and `docs/serve.py` (locally). Without them
the playground says so instead of starting.

One more thing: `usize` is 32 bits in WebAssembly, so the playground indexes Rips simplices with `u64`
(`phlite_rips::rips_barcode::<_, u64>`). With `usize` it could only handle 254 points for H₂.
