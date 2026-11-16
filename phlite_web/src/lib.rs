#![feature(alloc_error_hook)]
//! phlite compiled to WebAssembly for the browser playground.
//!
//! The pipelines, each taking plain arrays and an [`Options`] object and returning a [`Barcode`]:
//!
//! * [`rips`]: Vietoris–Rips persistence of a point cloud (any dimension) or a distance matrix, optionally DTM-weighted,
//!   by `phlite_clouds`;
//! * [`delaunay`]: alpha or Delaunay–Čech persistence of points in the plane or in space, by `phlite_clouds`;
//! * [`grpph`]: (grounded) persistent path homology of a weighted digraph, by `phlite_networks`;
//! * [`network`]: directed flag, ordered-tuple and Dowker complexes of a weighted digraph, by `phlite_networks`;
//! * [`image`]: persistence of a greyscale image (its cubical complex, T- or V-construction, sublevel or superlevel
//!   sets), by `phlite_images`;
//! * [`complex`]: persistence of an explicit filtered chain complex over Z/2, Z/3, Z/5 or Z/7, by `phlite_complexes`.
//!
//! With the `threads` feature (the default), the parallel algorithm runs on the global rayon pool, which
//! [`init_thread_pool`] starts on Web Workers (`wasm-bindgen-rayon`): the page must be cross-origin isolated (COOP/COEP
//! headers) for shared memory. Without it (the fallback build, for pages that can't share memory), rayon runs
//! everything on the calling thread.
//!
//! A panic or a failed allocation aborts, which leaves the module unusable and, on a thread of the pool, would leave
//! the calling thread waiting forever. So [`set_crash_hooks`] first reports the crash on a `BroadcastChannel`, which the
//! page listens to and answers by restarting the worker.

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Bar, Method},
    columns::{ColumnRepresentation, HeapColumns, MergedColumns, PivotColumns, SortedColumns},
    fields::{Z2, Z3, Z5, Z7},
    reduction::{Clearing, Standard, Storage},
};
use phlite_complexes::{
    complex::{Chain, Complex, RawChain},
    field::IntField,
};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Whether this build has threads (shared memory and a pool started by `initThreadPool`).
#[wasm_bindgen]
pub fn threaded() -> bool {
    cfg!(feature = "threads")
}

// Posting to a BroadcastChannel works from any thread, including one of the pool whose caller is blocked waiting
// for it, and the message is sent even though the thread aborts straight after.
#[wasm_bindgen(inline_js = "
export function report_crash(channel, message) {
  try { const c = new BroadcastChannel(channel); c.postMessage(message); c.close(); } catch (e) {}
}")]
extern "C" {
    fn report_crash(channel: &str, message: JsValue);
}

/// The channel crashes are reported on (in shared memory, so every thread of the pool sees it).
static CRASH_CHANNEL: OnceLock<String> = OnceLock::new();

/// Log panics to the console and report panics and failed allocations on the `BroadcastChannel` named `channel`, as
/// `{kind: "panic", message}` or `{kind: "oom", bytes}` (call once, at start-up).
#[wasm_bindgen]
pub fn set_crash_hooks(channel: String) {
    if CRASH_CHANNEL.set(channel).is_err() {
        return;
    }
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
        let message = js_sys_object(&[("kind", "panic".into()), ("message", info.to_string().into())]);
        report_crash(CRASH_CHANNEL.get().unwrap(), message);
    }));
    // Allocate as little as possible here: the heap is probably full
    std::alloc::set_alloc_error_hook(|layout| {
        let message = js_sys_object(&[("kind", "oom".into()), ("bytes", (layout.size() as f64).into())]);
        report_crash(CRASH_CHANNEL.get().unwrap(), message);
    });
}

/// A plain JavaScript object with the given properties.
fn js_sys_object(properties: &[(&str, JsValue)]) -> JsValue {
    let object = js_sys::Object::new();
    for (key, value) in properties {
        let _ = js_sys::Reflect::set(&object, &JsValue::from_str(key), value);
    }
    object.into()
}

/// The number of threads in the global pool.
#[wasm_bindgen]
pub fn n_threads() -> usize {
    rayon::current_num_threads()
}

/// Crash on purpose, to test how the page recovers: `"panic"` (on this thread), `"pool"` (a panic on a thread of the
/// pool, while this thread waits for it) or `"oom"` (run out of memory).
#[wasm_bindgen]
pub fn crash_for_testing(how: &str) {
    match how {
        "panic" => panic!("a test panic"),
        "pool" => {
            // The pool runs the first closure on one of its threads, while this one waits
            rayon::join(|| panic!("a test panic on a pool thread"), || ());
        }
        "oom" => {
            let mut blocks: Vec<Vec<u8>> = vec![];
            loop {
                blocks.push(vec![1u8; 1 << 28]);
            }
        }
        _ => {}
    }
}

// ---- Options ----

/// How to compute: the reduction algorithm and its options, the method and whether to compute representatives.
/// Every field is optional; a missing one takes the pipeline's default (the same as its Python pipeline's
/// `Algorithm.serial()`, or `Algorithm.parallel()` for the parallel algorithm).
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Options {
    /// `"standard"`, `"clearing"` (the default) or `"parallel"` (parallel clearing).
    pub algorithm: Option<String>,
    /// `"merged"`, `"heap"`, `"sorted"` or `"pivot"`.
    pub columns: Option<String>,
    pub cache: Option<bool>,
    pub store_r: Option<bool>,
    pub skip_zero_apparent_pairs: Option<bool>,
    /// `"locking"` or `"lock-free"`, for the parallel algorithm.
    pub storage: Option<String>,
    /// `"cohomology"` (the default) or `"homology"`.
    pub method: Option<String>,
    pub representatives: bool,
    /// Find dimension 0 by union-find (Rips and images only; the default there).
    pub union_find: Option<bool>,
    /// Build the matrices in parallel (by default, exactly when the algorithm is parallel clearing).
    pub parallel_setup: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Standard,
    Clearing,
    Parallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Columns {
    Merged,
    Heap,
    Sorted,
    Pivot,
}

/// A pipeline's defaults, as in Python (`phlite.core.defaults()`).
#[derive(Debug, Clone, Copy)]
struct Defaults {
    columns: Columns,
    cache: bool,
    store_r: bool,
    /// Whether the matrices know the rank of every row, which pivot columns need.
    row_ranks: bool,
    /// Whether dimension 0 is found by union-find by default, or `None` if the pipeline can't.
    union_find: Option<bool>,
    /// Whether the pipeline can build its matrices in parallel.
    parallel_setup: bool,
}

const RIPS_DEFAULTS: Defaults = Defaults {
    columns: Columns::Heap,
    cache: false,
    store_r: false,
    row_ranks: false,
    union_find: Some(true),
    parallel_setup: true,
};
const GRPPH_DEFAULTS: Defaults = Defaults {
    columns: Columns::Merged,
    cache: true,
    store_r: false,
    row_ranks: false,
    union_find: None,
    parallel_setup: false,
};
/// The lazy network complexes' (directed flag, ordered tuples, Dowker), as `phlite.networks`'.
const NETWORK_DEFAULTS: Defaults = Defaults {
    columns: Columns::Merged,
    cache: true,
    store_r: false,
    row_ranks: false,
    union_find: None,
    parallel_setup: true,
};
const IMAGE_DEFAULTS: Defaults = Defaults {
    columns: Columns::Merged,
    cache: false,
    store_r: true,
    row_ranks: false,
    union_find: Some(true),
    parallel_setup: true,
};
const COMPLEX_DEFAULTS: Defaults = Defaults {
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    row_ranks: true,
    union_find: None,
    parallel_setup: false,
};

/// The resolved configuration.
#[derive(Debug, Clone)]
struct Config {
    kind: Kind,
    columns: Columns,
    cache: bool,
    store_r: bool,
    skip_zero_apparent_pairs: bool,
    storage: Storage,
    method: Method,
    representatives: bool,
    union_find: bool,
    parallel_setup: bool,
    /// Whether the pipeline has a choice of setup (to describe).
    has_setup: bool,
}

/// An error for JavaScript, from a message or from any of phlite's errors.
fn err(message: impl std::fmt::Display) -> JsError {
    JsError::new(&message.to_string())
}

impl Config {
    fn resolve(options: JsValue, defaults: Defaults) -> Result<Self, JsError> {
        let options: Options = if options.is_undefined() || options.is_null() {
            Options::default()
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|e| err(format!("Bad options: {e}")))?
        };
        let kind = match options.algorithm.as_deref().unwrap_or("clearing") {
            "standard" => Kind::Standard,
            "clearing" => Kind::Clearing,
            "parallel" => Kind::Parallel,
            other => return Err(err(format!("Unknown algorithm {other:?}"))),
        };
        let columns = match options.columns.as_deref() {
            None => {
                // As in the Python presets: only serial clearing uses pivot columns
                if defaults.columns == Columns::Pivot && kind != Kind::Clearing {
                    Columns::Merged
                } else {
                    defaults.columns
                }
            }
            Some("merged") => Columns::Merged,
            Some("heap") => Columns::Heap,
            Some("sorted") => Columns::Sorted,
            Some("pivot") => {
                if kind != Kind::Clearing {
                    return Err(err("Pivot columns are only used by serial clearing"));
                }
                if !defaults.row_ranks {
                    return Err(err(
                        "Pivot columns need the rank of every row, which this pipeline's matrices don't know",
                    ));
                }
                Columns::Pivot
            }
            Some(other) => return Err(err(format!("Unknown columns {other:?}"))),
        };
        let store_r = options
            .store_r
            .unwrap_or(defaults.store_r && kind == Kind::Clearing);
        if store_r && kind != Kind::Clearing {
            return Err(err("Storing R is only supported by serial clearing"));
        }
        let skip = options
            .skip_zero_apparent_pairs
            .unwrap_or(kind != Kind::Standard);
        if skip && kind == Kind::Standard {
            return Err(err(
                "The standard algorithm can't skip zero-length apparent pairs",
            ));
        }
        let storage = match options.storage.as_deref().unwrap_or("locking") {
            "locking" => Storage::Locking,
            "lock-free" => Storage::LockFree,
            other => return Err(err(format!("Unknown storage {other:?}"))),
        };
        let method = match options.method.as_deref().unwrap_or("cohomology") {
            "cohomology" => Method::Cohomology,
            "homology" => Method::Homology,
            other => return Err(err(format!("Unknown method {other:?}"))),
        };
        let union_find = match (options.union_find, defaults.union_find) {
            (Some(true), None) => return Err(err("This pipeline has no union-find for dimension 0")),
            (Some(u), Some(_)) => u,
            (_, d) => d.unwrap_or(false),
        };
        let parallel_setup = options.parallel_setup.unwrap_or(kind == Kind::Parallel);
        if parallel_setup && !defaults.parallel_setup {
            return Err(err("This pipeline has no parallel setup"));
        }
        if parallel_setup && !threaded() {
            return Err(err("A parallel setup needs threads, which this page has not got"));
        }
        Ok(Self {
            kind,
            columns,
            cache: options.cache.unwrap_or(defaults.cache),
            store_r,
            skip_zero_apparent_pairs: skip,
            storage,
            method,
            representatives: options.representatives,
            union_find,
            parallel_setup,
            has_setup: defaults.parallel_setup,
        })
    }

    fn run<V: Visitor>(&self, visitor: V) -> V::Output {
        match self.columns {
            Columns::Merged => self.run_with(MergedColumns, visitor),
            Columns::Heap => self.run_with(HeapColumns, visitor),
            Columns::Sorted => self.run_with(SortedColumns, visitor),
            Columns::Pivot => self.run_with(PivotColumns, visitor),
        }
    }

    fn run_with<R: ColumnRepresentation, V: Visitor>(&self, columns: R, visitor: V) -> V::Output {
        let clearing = |columns: R| {
            Clearing::new()
                .with_columns(columns)
                .cache_columns(self.cache)
                .store_r(self.store_r)
                .skip_zero_apparent_pairs(self.skip_zero_apparent_pairs)
        };
        match self.kind {
            Kind::Standard => visitor.visit(Algorithm::Standard(
                Standard::new()
                    .with_columns(columns)
                    .cache_columns(self.cache),
            )),
            Kind::Clearing => visitor.visit(Algorithm::Clearing(clearing(columns))),
            Kind::Parallel => visitor.visit(Algorithm::ParallelClearing(
                clearing(columns).parallel().storage(self.storage),
            )),
        }
    }
}

/// Something to do with an [`Algorithm`] (a trait rather than a closure, since the columns are a type parameter).
trait Visitor {
    type Output;
    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output;
}

// ---- Results ----

/// A bar, with cells given as small arrays whose meaning depends on the pipeline:
/// the vertices of a simplex (Rips), the nodes of a path (GrPPH), the doubled grid coordinates `[x, y]` of a cubical
/// cell (images; even coordinates are pixels) or the index of a cell (explicit complexes).
#[derive(Debug, Serialize)]
pub struct OutBar {
    pub dim: usize,
    pub birth: f64,
    /// `null` for an essential bar.
    pub death: Option<f64>,
    pub birth_cell: Vec<u32>,
    pub death_cell: Option<Vec<u32>>,
    /// A representative cycle as `[coefficient, cell]` pairs, if representatives were asked for.
    pub rep: Option<Vec<(u32, Vec<u32>)>>,
    /// For extended persistence: `"ordinary"`, `"relative"`, `"extended+"` or `"extended-"` (absent otherwise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Barcode {
    pub bars: Vec<OutBar>,
    /// What was computed, in words (sizes and the resolved algorithm).
    pub summary: String,
    /// The filtration's edges, to draw the complex at any value (as typed arrays, see [`FiltrationEdges`]).
    #[serde(skip)]
    pub edges: Option<FiltrationEdges>,
}

/// The edges of the filtration a barcode was computed from, sorted by when they enter, so that the page can draw the
/// complex at any value from the very values phlite used: the edge `pairs[2k] → pairs[2k + 1]` at `values[k]` (both
/// ways round for a flag filtration of points; one way for a digraph's), with each point's birth if not all are born
/// at 0 (weighted Rips). Edges that never enter are left out.
#[derive(Debug, Default)]
pub struct FiltrationEdges {
    pairs: Vec<u32>,
    values: Vec<f64>,
    births: Option<Vec<f64>>,
}

impl FiltrationEdges {
    fn from_unsorted(mut edges: Vec<(u32, u32, f64)>, births: Option<Vec<f64>>) -> Self {
        edges.retain(|e| e.2.is_finite());
        edges.sort_by(|a, b| a.2.total_cmp(&b.2));
        Self {
            pairs: edges.iter().flat_map(|&(a, b, _)| [a, b]).collect(),
            values: edges.iter().map(|e| e.2).collect(),
            births,
        }
    }

    /// Every pair of points of a (weighted) Rips filtration.
    fn flag<D: phlite_clouds::rips::Distances>(distances: &D) -> Self {
        let n = distances.n_points();
        let edges = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i as u32, j as u32, distances.distance(i, j).into_inner())))
            .collect();
        let births = D::HAS_BIRTHS.then(|| (0..n).map(|v| distances.birth(v).into_inner()).collect());
        Self::from_unsorted(edges, births)
    }

    fn to_js(&self) -> JsValue {
        js_sys_object(&[
            ("pairs", js_sys::Uint32Array::from(&self.pairs[..]).into()),
            ("values", js_sys::Float64Array::from(&self.values[..]).into()),
            (
                "births",
                self.births
                    .as_ref()
                    .map_or(JsValue::NULL, |b| js_sys::Float64Array::from(&b[..]).into()),
            ),
        ])
    }
}

/// Filtrations with more points than this aren't drawn (and their edges not sent).
const DRAWN_POINTS: usize = 1200;

fn to_js(barcode: Barcode) -> Result<JsValue, JsError> {
    let value = barcode
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| err(e.to_string()))?;
    if let Some(edges) = &barcode.edges {
        let _ = js_sys::Reflect::set(&value, &JsValue::from_str("edges"), &edges.to_js());
    }
    Ok(value)
}

/// Translate bars, dropping those of length zero, with `cell` describing a cell, `dim` its dimension and `value`
/// turning filtration values into output values.
fn translate<C: Copy, CF: IntField>(
    bars: Vec<Bar<C, NotNan<f64>, CF>>,
    dim: impl Fn(C) -> usize,
    cell: impl Fn(C) -> Vec<u32>,
    value: impl Fn(f64) -> f64,
) -> Vec<OutBar> {
    bars.into_iter()
        .filter(|bar| bar.death.is_none_or(|(_, d)| d != bar.birth_value))
        .map(|bar| OutBar {
            dim: dim(bar.birth),
            birth: value(bar.birth_value.into_inner()),
            death: bar.death.map(|(_, d)| value(d.into_inner())),
            birth_cell: cell(bar.birth),
            death_cell: bar.death.map(|(c, _)| cell(c)),
            rep: bar.representative.map(|rep| {
                rep.into_iter()
                    .map(|(coeff, c)| (coeff.to_int() as u32, cell(c)))
                    .collect()
            }),
            kind: None,
        })
        .collect()
}

fn describe(config: &Config) -> String {
    let kind = match config.kind {
        Kind::Standard => "standard",
        Kind::Clearing => "clearing",
        Kind::Parallel => "parallel clearing",
    };
    let columns = match config.columns {
        Columns::Merged => "merged",
        Columns::Heap => "heap",
        Columns::Sorted => "sorted",
        Columns::Pivot => "pivot",
    };
    let mut s = format!(
        "{} by {kind}, {columns} columns",
        match config.method {
            Method::Cohomology => "cohomology",
            Method::Homology => "homology",
        }
    );
    if config.cache {
        s += ", cached";
    }
    if config.store_r {
        s += ", storing R";
    }
    if config.union_find {
        s += ", union-find in dimension 0";
    }
    if config.has_setup && config.parallel_setup != (config.kind == Kind::Parallel) {
        s += if config.parallel_setup { ", parallel setup" } else { ", serial setup" };
    }
    if config.kind == Kind::Parallel {
        s += &format!(
            ", {} storage on {}",
            match config.storage {
                Storage::LockFree => "lock-free",
                _ => "locking",
            },
            match rayon::current_num_threads() {
                1 => "1 thread".to_string(),
                n => format!("{n} threads"),
            }
        );
    }
    s
}

// ---- Vietoris–Rips ----

/// Vietoris–Rips persistence.
///
/// * `coords`: `n × d` point coordinates, row by row, with `d = point_dim`; or, with `point_dim = 0`, an `n × n`
///   distance matrix (row by row, symmetric).
/// * `max_dim`: the top dimension of homology.
/// * `threshold`: only build the complex up to this diameter (`undefined` or infinite for no limit).
/// * `metric`: for coordinates, `"euclidean"`, `"manhattan"` or `"chebyshev"`.
/// * `dtm_k`: if given, DTM-Rips: weighted Rips with each point weighted by its distance to measure over its `dtm_k`
///   nearest neighbours (itself included, `q = 2`), as `phlite.clouds.rips(weights=dtm(k=dtm_k))`.
#[wasm_bindgen]
pub fn rips(
    coords: &[f64],
    point_dim: usize,
    max_dim: usize,
    threshold: Option<f64>,
    metric: &str,
    dtm_k: Option<usize>,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let config = Config::resolve(options, RIPS_DEFAULTS)?;
    use phlite_clouds::rips::Metric;
    let metric_name = metric;
    let metric = match metric {
        "euclidean" => Metric::Euclidean,
        "manhattan" => Metric::Manhattan,
        "chebyshev" => Metric::Chebyshev,
        other => return Err(err(format!("Unknown metric {other:?}"))),
    };
    let distances = if point_dim == 0 {
        let n = (coords.len() as f64).sqrt() as usize;
        if n * n != coords.len() {
            return Err(err("A distance matrix must be square"));
        }
        (0..n)
            .map(|i| (0..n).map(|j| not_nan(coords[i * n + j])).collect())
            .collect::<Result<Vec<Vec<_>>, _>>()?
    } else {
        if !coords.len().is_multiple_of(point_dim) {
            return Err(err("The coordinates don't divide into points"));
        }
        if !coords.iter().all(|x| x.is_finite()) {
            return Err(err("Coordinates must be finite"));
        }
        let points: Vec<&[f64]> = coords.chunks(point_dim).collect();
        points
            .iter()
            .map(|p| points.iter().map(|q| metric.distance(p, q)).collect())
            .collect::<Vec<Vec<_>>>()
    };
    let n = distances.len();
    if n == 0 {
        return to_js(Barcode {
            bars: vec![],
            summary: "No points".into(),
            edges: None,
        });
    }
    // usize is 32 bits here, so index simplices by u64: up to 65 534 points for H₂ rather than 254
    if !phlite_clouds::rips::RipsIndex::<u64>::fits(n, max_dim + 1) {
        return Err(err(format!(
            "Too many points ({n}) for simplices of dimension {}",
            max_dim + 1
        )));
    }
    let threshold = not_nan(threshold.unwrap_or(f64::INFINITY))?;
    let weights = match dtm_k {
        None => None,
        Some(k) => Some(
            phlite_clouds::rips::distance_to_measure(
                n,
                |i| distances[i].iter().map(|d| d.into_inner()).collect(),
                k,
                2.0,
            )
            .map_err(err)?,
        ),
    };

    struct V<D> {
        distances: D,
        max_dim: usize,
        threshold: NotNan<f64>,
        method: Method,
        representatives: bool,
        union_find: bool,
        parallel_setup: bool,
    }
    impl<D: phlite_clouds::rips::Distances> Visitor for V<D> {
        type Output = Vec<phlite_clouds::rips::RipsBar<u64>>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            phlite_clouds::rips::rips_barcode_with::<_, u64, _>(
                self.distances,
                self.max_dim,
                self.threshold,
                self.method,
                algorithm,
                self.representatives,
                self.union_find,
                self.parallel_setup,
            )
        }
    }
    let (method, representatives) = (config.method, config.representatives);
    let (union_find, parallel_setup) = (config.union_find, config.parallel_setup);
    // The filtration's edges, from the distances the barcode is computed from
    let drawn = n <= DRAWN_POINTS;
    let (bars, edges) = match &weights {
        None => {
            let edges = drawn.then(|| FiltrationEdges::flag(&distances));
            (config.run(V { distances, max_dim, threshold, method, representatives, union_find, parallel_setup }), edges)
        }
        Some(weights) => {
            let distances = phlite_clouds::rips::Weighted::new(distances, weights).map_err(err)?;
            let edges = drawn.then(|| FiltrationEdges::flag(&distances));
            let v = V { distances, max_dim, threshold, method, representatives, union_find, parallel_setup };
            (config.run(v), edges)
        }
    };
    let bars = translate::<_, Z2>(
        bars,
        |s| s.dimension(n),
        |s| s.vertices(n).iter().map(|&v| v as u32).collect(),
        |v| v,
    );
    to_js(Barcode {
        bars,
        summary: format!(
            "{n} points, dimensions 0–{max_dim}{}{}; {}",
            if point_dim > 0 && metric_name != "euclidean" { format!(", {metric_name} metric") } else { String::new() },
            dtm_k.map_or(String::new(), |k| format!(", DTM-weighted (k = {k})")),
            describe(&config)
        ),
        edges,
    })
}

fn not_nan(x: f64) -> Result<NotNan<f64>, JsError> {
    NotNan::new(x).map_err(|_| err("NaN in the input"))
}

// ---- GrPPH ----

/// The edges `sources[i] → targets[i]` of weight `weights[i]`, checked.
fn read_edges(
    n_vertices: u32,
    sources: &[u32],
    targets: &[u32],
    weights: &[f64],
) -> Result<Vec<(u32, u32, f64)>, JsError> {
    if sources.len() != targets.len() || sources.len() != weights.len() {
        return Err(err(
            "sources, targets and weights must have the same length",
        ));
    }
    if n_vertices > u16::MAX as u32 {
        return Err(err("Too many nodes"));
    }
    let mut edges = Vec::with_capacity(sources.len());
    for ((&s, &t), &w) in sources.iter().zip(targets).zip(weights) {
        if s >= n_vertices || t >= n_vertices {
            return Err(err(format!("Edge {s} → {t} has a node out of range")));
        }
        if s == t {
            return Err(err(format!("Self-loop at node {s}")));
        }
        if w.is_nan() || w < 0.0 {
            return Err(err(format!(
                "Edge {s} → {t} has weight {w}; weights must be non-negative"
            )));
        }
        edges.push((s, t, w));
    }
    Ok(edges)
}

/// Grounded persistent path homology (or, not `grounded`, persistent path homology) of a weighted digraph with
/// `n_vertices` nodes and edges `sources[i] → targets[i]` of weight `weights[i]`.
#[wasm_bindgen]
pub fn grpph(
    n_vertices: u32,
    sources: &[u32],
    targets: &[u32],
    weights: &[f64],
    grounded: bool,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let config = Config::resolve(options, GRPPH_DEFAULTS)?;
    let edges = read_edges(n_vertices, sources, targets, weights)?;
    let filtration = phlite_networks::grpph::build_filtration(n_vertices, &edges).map_err(err)?;
    let drawn = FiltrationEdges::from_unsorted(
        filtration
            .iter()
            .enumerate()
            .flat_map(|(a, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |&(b, _)| a != b)
                    .filter_map(move |(b, d)| d.map(|d| (a as u32, b as u32, d.into_inner())))
            })
            .collect(),
        None,
    );
    let edge_set = if grounded {
        phlite_networks::grpph::build_edge_set(&edges)
    } else {
        phlite_networks::grpph::coboundary::DigraphEdgeSet::default()
    };

    struct V<'a> {
        filtration: &'a phlite_networks::grpph::coboundary::VecDigraphFiltration,
        edge_set: &'a phlite_networks::grpph::coboundary::DigraphEdgeSet,
        n: u16,
        method: Method,
        representatives: bool,
    }
    impl Visitor for V<'_> {
        type Output = Vec<phlite_networks::grpph::GrPPHBar>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            phlite_networks::grpph::grpph_barcode(
                self.filtration,
                self.edge_set,
                self.n,
                self.method,
                algorithm,
                self.representatives,
            )
        }
    }
    let bars = config.run(V {
        filtration: &filtration,
        edge_set: &edge_set,
        n: n_vertices as u16,
        method: config.method,
        representatives: config.representatives,
    });
    use phlite_networks::grpph::coboundary::{PathHom2Cell, PathHomCell};
    let bars = translate::<_, Z2>(
        bars,
        |c: PathHomCell| c.dimension(),
        |c| match c {
            PathHomCell::Node(a) => vec![a as u32],
            PathHomCell::Edge(a, b) => vec![a as u32, b as u32],
            PathHomCell::TwoCell(PathHom2Cell::DoubleEdge(a, b)) => {
                vec![a as u32, b as u32, a as u32]
            }
            PathHomCell::TwoCell(PathHom2Cell::DirectedTriangle(a, b, c)) => {
                vec![a as u32, b as u32, c as u32]
            }
            PathHomCell::TwoCell(PathHom2Cell::LongSquare(a, b, c, d)) => {
                vec![a as u32, b as u32, c as u32, d as u32]
            }
        },
        |v| v,
    );
    to_js(Barcode {
        bars,
        summary: format!(
            "{n_vertices} nodes, {} edges, dimensions 0–1, {}; {}",
            edges.len(),
            if grounded { "grounded" } else { "not grounded" },
            describe(&config)
        ),
        edges: Some(drawn),
    })
}

// ---- Directed flag, ordered-tuple and Dowker complexes ----

/// Persistence of a complex of a weighted digraph (edges as for [`grpph`]), in dimensions `0..=max_dim`, cells given
/// by their nodes. `kind` is one of
/// * `"flag"`: the directed flag complex, each edge at its weight (flagser's);
/// * `"flag-sp"`: the directed flag complex, each pair at the length of the shortest path;
/// * `"grounded-flag"`: the directed flag complex grounded at the digraph's edges (dimensions 0 and 1);
/// * `"tuples"`: the ordered-tuple complex (directed Rips), each edge at its weight;
/// * `"dowker-sink"`, `"dowker-source"`: the Dowker sink or source complex;
/// * `"pph"`: persistent path homology (Chowdhury and Mémoli), by `phlite_networks::pph` (cells are regular paths).
///
/// With `shortest_path`, each pair `a → b` enters at the length of the shortest path from `a` to `b` rather than with
/// the edge's weight (as `filtration='shortest_path'` in Python; the grounded flag complex always does).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn network(
    kind: &str,
    shortest_path: bool,
    n_vertices: u32,
    sources: &[u32],
    targets: &[u32],
    weights: &[f64],
    max_dim: usize,
    options: JsValue,
) -> Result<JsValue, JsError> {
    use phlite_networks::digraph::{Digraph, Distances};
    let config = Config::resolve(options, NETWORK_DEFAULTS)?;
    let edges = read_edges(n_vertices, sources, targets, weights)?;
    let distances = if shortest_path { Distances::ShortestPath } else { Distances::Weight };
    let (bars, sizes, drawn) = match kind {
        "pph" => pph_bars(n_vertices, &edges, shortest_path, max_dim, &config)?,
        _ => {
            // The digraph the complex is built on, and its edges to draw
            let graph = if kind == "grounded-flag" {
                Digraph::shortest_paths(n_vertices, &edges)
            } else {
                Digraph::with_distances(n_vertices, &edges, distances)
            }
            .map_err(err)?;
            let drawn = FiltrationEdges::from_unsorted(
                graph.edges().map(|(a, b, w)| (a, b, w.into_inner())).collect(),
                None,
            );
            let (bars, sizes) = network_bars(kind, &edges, graph, max_dim, &config)?;
            (bars, sizes, drawn)
        }
    };
    to_js(Barcode {
        bars,
        summary: format!(
            "{n_vertices} nodes, {} edges, {sizes}{}; {}",
            edges.len(),
            if shortest_path || kind == "grounded-flag" { ", shortest-path filtration" } else { "" },
            describe(&config)
        ),
        edges: Some(drawn),
    })
}

/// The barcode of a lazy complex of a digraph (every kind of [`network`] but PPH).
fn network_bars(
    kind: &str,
    edges: &[(u32, u32, f64)],
    graph: phlite_networks::digraph::Digraph,
    max_dim: usize,
    config: &Config,
) -> Result<(Vec<OutBar>, String), JsError> {
    use phlite_networks::{
        dowker::{DowkerComplex, Side},
        flag::{FlagComplex, Kind as FlagKind},
    };
    Ok(match kind {
        "flag" => lazy_bars(&FlagComplex::new(FlagKind::DirectedFlag, graph), max_dim, config)?,
        "grounded-flag" => {
            let base = edges.iter().map(|&(a, b, _)| (a, b));
            let complex = FlagComplex::grounded(FlagKind::DirectedFlag, graph, base).map_err(err)?;
            lazy_bars(&complex, max_dim.min(1), config)?
        }
        "tuples" => lazy_bars(&FlagComplex::new(FlagKind::OrderedTuples, graph), max_dim, config)?,
        "dowker-sink" => lazy_bars(&DowkerComplex::new(&graph, Side::Sink), max_dim, config)?,
        "dowker-source" => lazy_bars(&DowkerComplex::new(&graph, Side::Source), max_dim, config)?,
        other => return Err(err(format!("Unknown network complex {other:?}"))),
    })
}

/// Persistent path homology in dimensions `0..=max_dim`, cells as their vertices.
fn pph_bars(
    n_vertices: u32,
    edges: &[(u32, u32, f64)],
    shortest_path: bool,
    max_dim: usize,
    config: &Config,
) -> Result<(Vec<OutBar>, String, FiltrationEdges), JsError> {
    use phlite_networks::pph::{pph_barcode_with, Network, PPHBar};
    let edges: Vec<(u16, u16, f64)> = if shortest_path {
        // Every pair joined by a path, at the length of the shortest one
        phlite_networks::grpph::build_filtration(n_vertices, edges)
            .map_err(err)?
            .iter()
            .enumerate()
            .flat_map(|(a, row)| {
                row.iter()
                    .enumerate()
                    .filter_map(move |(b, d)| d.map(|d| (a as u16, b as u16, d.into_inner())))
            })
            .collect()
    } else {
        edges.iter().map(|&(a, b, w)| (a as u16, b as u16, w)).collect()
    };
    let network = Network::from_edges(n_vertices as u16, &edges).map_err(err)?;
    let n = n_vertices as u16;
    let drawn = FiltrationEdges::from_unsorted(
        (0..n)
            .flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b)))
            .filter_map(|(a, b)| network.edge_time(a, b).map(|t| (a as u32, b as u32, t.into_inner())))
            .collect(),
        None,
    );
    if max_dim >= network.max_dimension() {
        return Err(err(format!(
            "Too many nodes ({n_vertices}) for paths of dimension {}",
            max_dim + 1
        )));
    }
    struct V<'a> {
        network: &'a Network,
        max_dim: usize,
        method: Method,
        representatives: bool,
        parallel_setup: bool,
    }
    impl Visitor for V<'_> {
        type Output = Vec<PPHBar<Z2>>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            pph_barcode_with(
                self.network,
                self.max_dim,
                self.method,
                algorithm,
                self.representatives,
                self.parallel_setup,
            )
        }
    }
    let bars = config.run(V {
        network: &network,
        max_dim,
        method: config.method,
        representatives: config.representatives,
        parallel_setup: config.parallel_setup,
    });
    let bars = translate::<_, Z2>(
        bars,
        |p| p.dimension(),
        |p| network.vertices(p).into_iter().map(u32::from).collect(),
        |v| v,
    );
    Ok((bars, format!("dimensions 0–{max_dim}, regular paths"), drawn))
}

/// The barcode of a lazy complex in dimensions `0..=max_dim`, cells as their vertices.
fn lazy_bars<K: phlite_networks::lazy::LazyComplex>(
    complex: &K,
    max_dim: usize,
    config: &Config,
) -> Result<(Vec<OutBar>, String), JsError> {
    use phlite_networks::lazy::{barcode_with, fits};
    if !fits(complex, max_dim) {
        return Err(err(format!(
            "Too many nodes ({}) for simplices of dimension {}",
            complex.n_vertices(),
            max_dim + 1
        )));
    }
    struct V<'a, K> {
        complex: &'a K,
        max_dim: usize,
        method: Method,
        representatives: bool,
        parallel_setup: bool,
    }
    impl<K: phlite_networks::lazy::LazyComplex> Visitor for V<'_, K> {
        type Output = Vec<Bar<phlite_networks::simplex::Simplex, NotNan<f64>, Z2>>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            barcode_with(
                self.complex,
                self.max_dim,
                self.method,
                algorithm,
                self.representatives,
                self.parallel_setup,
            )
        }
    }
    let bars = config.run(V {
        complex,
        max_dim,
        method: config.method,
        representatives: config.representatives,
        parallel_setup: config.parallel_setup,
    });
    let codec = complex.codec();
    let bars = translate::<_, Z2>(
        bars,
        |s| s.dimension(),
        |s| codec.vertices(s).to_vec(),
        |v| v,
    );
    Ok((bars, format!("dimensions 0–{max_dim}")))
}

// ---- Undirected graphs ----

/// Persistence of the flag complex of the undirected graph with the edges `sources[i] - targets[i]` of weight
/// `weights[i]` (either way round: an edge given both ways keeps its smaller weight), in dimensions `0..=max_dim`.
/// `kind` is `"flag"` (each clique at its heaviest edge) or `"flag-sp"` (shortest paths: Rips of the graph's metric).
///
/// As `phlite.networks.flag`: by weight, the complex is listed and reduced explicitly (with `phlite.complexes`'
/// defaults); shortest paths use Rips' machinery (with its defaults).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn graph(
    kind: &str,
    n_vertices: u32,
    sources: &[u32],
    targets: &[u32],
    weights: &[f64],
    max_dim: usize,
    options: JsValue,
) -> Result<JsValue, JsError> {
    use phlite_networks::undirected::{cliques_within, explicit_flag_barcode, flag_barcode, Graph};
    use phlite_clouds::rips::Distances;
    let edges = read_edges(n_vertices, sources, targets, weights)?;
    let n = n_vertices as usize;
    let graph = match kind {
        "flag" => Graph::new(n_vertices, &edges, None),
        "flag-sp" => Graph::shortest_paths(n_vertices, &edges, None),
        other => return Err(err(format!("Unknown graph complex {other:?}"))),
    }
    .map_err(err)?;
    // The edges to draw: every edge of the graph (with its value), both ways round
    let drawn = FiltrationEdges::from_unsorted(
        (0..n)
            .flat_map(|u| graph.neighbours(u, 0).map(move |v| (u, v)).collect::<Vec<_>>())
            .map(|(u, v)| (u as u32, v as u32, graph.distance(u, v).into_inner()))
            .collect(),
        None,
    );
    let top = max_dim + 1;
    let simplices = if kind == "flag-sp" {
        None
    } else {
        cliques_within(&graph, top, graph.enclosing_radius(), 1_000_000)
    };
    let (bars, how) = match simplices {
        Some(simplices) => {
            let config = Config::resolve(options, COMPLEX_DEFAULTS)?;
            struct V<'a> {
                simplices: &'a [(NotNan<f64>, Vec<u32>)],
                max_dim: usize,
                method: Method,
                representatives: bool,
            }
            impl Visitor for V<'_> {
                type Output = Vec<Bar<Vec<u32>, NotNan<f64>, Z2>>;
                fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
                    explicit_flag_barcode(self.simplices, self.max_dim, self.method, algorithm, self.representatives)
                }
            }
            let bars = config.run(V {
                simplices: &simplices,
                max_dim,
                method: config.method,
                representatives: config.representatives,
            });
            // (Not `translate`, which needs cells that are Copy: these are lists of nodes)
            let out = bars
                .into_iter()
                .filter(|bar| bar.death.as_ref().is_none_or(|(_, d)| *d != bar.birth_value))
                .map(|bar| OutBar {
                    dim: bar.birth.len() - 1,
                    birth: bar.birth_value.into_inner(),
                    death: bar.death.as_ref().map(|(_, d)| d.into_inner()),
                    birth_cell: bar.birth.clone(),
                    death_cell: bar.death.as_ref().map(|(c, _)| c.clone()),
                    rep: bar.representative.map(|rep| rep.into_iter().map(|(_, c)| (1, c)).collect()),
                    kind: None,
                })
                .collect();
            (out, format!("{} simplices listed; {}", simplices.len(), describe(&config)))
        }
        None => {
            let config = Config::resolve(options, RIPS_DEFAULTS)?;
            if !phlite_clouds::rips::RipsIndex::<u64>::fits(n, max_dim + 1) {
                return Err(err(format!("Too many nodes ({n}) for simplices of dimension {}", max_dim + 1)));
            }
            struct V<'a> {
                graph: &'a Graph,
                max_dim: usize,
                method: Method,
                representatives: bool,
                union_find: bool,
                parallel_setup: bool,
            }
            impl Visitor for V<'_> {
                type Output = Vec<phlite_clouds::rips::RipsBar<u64>>;
                fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
                    flag_barcode(
                        self.graph,
                        self.max_dim,
                        phlite::values::INFINITY,
                        self.method,
                        algorithm,
                        self.representatives,
                        self.union_find,
                        self.parallel_setup,
                    )
                }
            }
            let bars = config.run(V {
                graph: &graph,
                max_dim,
                method: config.method,
                representatives: config.representatives,
                union_find: config.union_find,
                parallel_setup: config.parallel_setup,
            });
            let bars = translate::<_, Z2>(
                bars,
                |s| s.dimension(n),
                |s| s.vertices(n).iter().map(|&v| v as u32).collect(),
                |v| v,
            );
            (bars, format!("by Rips' machinery; {}", describe(&config)))
        }
    };
    to_js(Barcode {
        bars,
        summary: format!(
            "{n_vertices} nodes, {} edges as undirected, dimensions 0–{max_dim}, {}, {how}",
            graph.n_edges(),
            if kind == "flag" { "flag complex by weight" } else { "shortest paths" }
        ),
        edges: Some(drawn),
    })
}

// ---- Explicit complexes ----

/// The barcode of `chain` filtered by `values` (with `NaN` for a value to fill in from the cell's faces).
/// The barcode of a filtered complex with whichever algorithm `config` chose.
fn complex_barcode<CF: IntField + Send + Sync + 'static>(
    complex: &Complex<CF>,
    max_dim: Option<usize>,
    config: &Config,
) -> Vec<Bar<phlite_complexes::complex::Cell, NotNan<f64>, CF>> {
    struct V<'a, CF> {
        complex: &'a Complex<CF>,
        max_dim: Option<usize>,
        method: Method,
        representatives: bool,
    }
    impl<CF: IntField + Send + Sync + 'static> Visitor for V<'_, CF> {
        type Output = Vec<Bar<phlite_complexes::complex::Cell, NotNan<f64>, CF>>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            self.complex
                .barcode(self.max_dim, self.method, algorithm, self.representatives)
        }
    }
    config.run(V {
        complex,
        max_dim,
        method: config.method,
        representatives: config.representatives,
    })
}

fn complex_bars<CF: IntField + Send + Sync + 'static>(
    raw: RawChain,
    values: Option<Vec<f64>>,
    max_dim: Option<usize>,
    extended: bool,
    config: &Config,
) -> Result<(Vec<OutBar>, String), JsError> {
    let chain = Chain::<CF>::from_raw(raw).map_err(|e| err(e.to_string()))?;
    let values = match values {
        None => None,
        Some(values) => Some(
            chain
                .extend(
                    values
                        .into_iter()
                        .map(|v| (!v.is_nan()).then_some(v))
                        .collect(),
                )
                .map_err(|e| err(e.to_string()))?,
        ),
    };
    let complex = chain.filter(values).map_err(|e| err(e.to_string()))?;
    if extended {
        // Extended persistence (going back down by the least value of each cell and its faces): bars with a kind
        struct V<'a, CF> {
            complex: &'a Complex<CF>,
            method: Method,
        }
        impl<CF: IntField + Send + Sync + 'static> Visitor for V<'_, CF> {
            type Output = Vec<phlite::extended::ExtendedBar<phlite_complexes::complex::Cell, f64>>;
            fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
                phlite_complexes::extended::extended_persistence(self.complex, self.method, algorithm)
            }
        }
        let bars = config
            .run(V { complex: &complex, method: config.method })
            .into_iter()
            .map(|bar| OutBar {
                dim: bar.dimension,
                birth: bar.birth,
                death: Some(bar.death),
                birth_cell: vec![bar.birth_cell.0],
                death_cell: Some(vec![bar.death_cell.0]),
                rep: None,
                kind: Some(bar.kind.name()),
            })
            .collect();
        let summary = format!("{} cells, extended persistence", complex.n_cells());
        return Ok((bars, summary));
    }
    let bars = complex_barcode(&complex, max_dim, config);
    let bars = translate(bars, |c| complex.dimension(c), |c| vec![c.0], |v| v);
    let summary = format!(
        "{} cells, {} boundary entries",
        complex.n_cells(),
        complex.n_entries()
    );
    Ok((bars, summary))
}

/// Persistence of an explicit chain complex.
///
/// * Cell `j` has dimension `dimensions[j]` and boundary `Σ coefficients[k] · indices[k]` over
///   `k in indptr[j]..indptr[j + 1]` (`coefficients` may be empty for all ones).
/// * `values`: a filtration value for every cell (`NaN` to take the largest of its faces' values), or empty to use
///   the order of the cells.
/// * `modulus`: 2, 3, 5 or 7.
/// * `max_dim`: the top dimension of homology, or `undefined` for every dimension.
/// * `extended`: extended persistence instead (going up through the filtration, then back down by the least value
///   of each cell and its faces; bars come with a `kind`).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn complex(
    dimensions: &[u32],
    indptr: &[u32],
    indices: &[u32],
    coefficients: &[i32],
    values: &[f64],
    modulus: u32,
    max_dim: Option<usize>,
    extended: bool,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let config = Config::resolve(options, COMPLEX_DEFAULTS)?;
    let raw = RawChain {
        dimensions: dimensions.iter().map(|&d| d as i64).collect(),
        indptr: indptr.iter().map(|&i| i as i64).collect(),
        indices: indices.iter().map(|&i| i as i64).collect(),
        coefficients: (!coefficients.is_empty())
            .then(|| coefficients.iter().map(|&c| c as i64).collect()),
    };
    let values = (!values.is_empty()).then(|| values.to_vec());
    let (bars, sizes) = match modulus {
        2 => complex_bars::<Z2>(raw, values, max_dim, extended, &config)?,
        3 => complex_bars::<Z3>(raw, values, max_dim, extended, &config)?,
        5 => complex_bars::<Z5>(raw, values, max_dim, extended, &config)?,
        7 => complex_bars::<Z7>(raw, values, max_dim, extended, &config)?,
        other => {
            return Err(err(format!(
                "Unsupported modulus {other}; use 2, 3, 5 or 7"
            )))
        }
    };
    to_js(Barcode {
        bars,
        summary: format!("{sizes}, over Z/{modulus}; {}", describe(&config)),
        edges: None,
    })
}

// ---- Alpha and Delaunay–Čech ----

/// Alpha (or, with `cech`, Delaunay–Čech) persistence of points in the plane or in space (`coords` row by row, with
/// `point_dim` 2 or 3 coordinates each), by `phlite_clouds::alpha`: the exact Delaunay triangulation filtered by
/// radii (not squared radii). Cells are given by their vertices (input point indices); repeated points are merged into
/// the first.
#[wasm_bindgen]
pub fn delaunay(coords: &[f64], point_dim: usize, cech: bool, options: JsValue) -> Result<JsValue, JsError> {
    use phlite_clouds::alpha::{filtered_delaunay, vertices, DelaunayFiltration};
    let config = Config::resolve(options, COMPLEX_DEFAULTS)?;
    if !(point_dim == 2 || point_dim == 3) || !coords.len().is_multiple_of(point_dim) {
        return Err(err("Alpha and Delaunay–Čech persistence need points in the plane or in space"));
    }
    let n = coords.len() / point_dim;
    let points: Vec<Vec<f64>> = coords.chunks(point_dim).map(|p| p.to_vec()).collect();
    let filtration = if cech { DelaunayFiltration::DelaunayCech } else { DelaunayFiltration::Alpha };
    let complex = filtered_delaunay(&points, filtration).map_err(err)?;
    let max_dim = point_dim - 1;
    let filtered = complex.filtered(false);
    let bars = complex_barcode(&filtered, Some(max_dim), &config);
    let bars = translate(
        bars,
        |c| complex.cell(c).0,
        |c| vertices(complex.cell(c).1).to_vec(),
        |v| v,
    );
    to_js(Barcode {
        bars,
        summary: format!(
            "{n} points, Delaunay triangulation of {} cells, dimensions 0–{max_dim}, {} radii; {}",
            complex.n_simplices(),
            if cech { "Delaunay–Čech" } else { "alpha" },
            describe(&config)
        ),
        edges: None,
    })
}

// ---- Images ----

/// Persistence of a greyscale image (`pixels` row by row), by its cubical complex (`phlite_images::cubical`): with
/// `construction` `"T"`, pixels are squares (8-connected); with `"V"`, pixels are vertices (4-connected). With
/// `superlevel`, bright features come first (the filtration is by decreasing intensity, and the values returned are
/// intensities, so bars run downwards). Representatives and cells are doubled grid coordinates `[x, y]`: in the
/// T-construction pixel `(x, y)` is the square `[2x + 1, 2y + 1]`, in the V-construction the vertex `[2x, 2y]`.
#[wasm_bindgen]
pub fn image(
    pixels: &[f64],
    width: usize,
    height: usize,
    superlevel: bool,
    construction: &str,
    options: JsValue,
) -> Result<JsValue, JsError> {
    use phlite_images::cubical::{filtration_barcode, Construction, CubicalComplex, Filtration};
    let config = Config::resolve(options, IMAGE_DEFAULTS)?;
    if width == 0 || height == 0 || pixels.len() != width * height {
        return Err(err("The image size doesn't match its pixels"));
    }
    let construction = match construction {
        "T" => Construction::T,
        "V" => Construction::V,
        other => return Err(err(format!("Unknown construction {other:?}"))),
    };
    let filtration = if superlevel { Filtration::Superlevel } else { Filtration::Sublevel };
    let complex = CubicalComplex::new(pixels.to_vec(), [height, width], construction, filtration)
        .map_err(|e| err(e.to_string()))?;
    struct V<'a> {
        complex: &'a CubicalComplex<2>,
        method: Method,
        representatives: bool,
        union_find: bool,
        parallel_setup: bool,
    }
    impl Visitor for V<'_> {
        type Output = Vec<phlite_images::cubical::CubicalBar>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            // In the sublevel filtration's values, so that classes born at +inf (never) are told from those born at
            // -inf (first, for superlevel sets)
            filtration_barcode(
                self.complex,
                1,
                self.method,
                algorithm,
                self.representatives,
                false,
                self.union_find,
                self.parallel_setup,
            )
        }
    }
    let bars = config.run(V {
        complex: &complex,
        method: config.method,
        representatives: config.representatives,
        union_find: config.union_find,
        parallel_setup: config.parallel_setup,
    });
    let sign = if superlevel { -1.0 } else { 1.0 };
    let bars = bars
        .into_iter()
        .filter(|bar| bar.birth_value.into_inner() != f64::INFINITY)
        .map(|mut bar| {
            if bar.death.is_some_and(|(_, d)| d.into_inner() == f64::INFINITY) {
                bar.death = None;
            }
            bar
        })
        .collect();
    let bars = translate::<_, Z2>(
        bars,
        |c| complex.dimension(c),
        |c| {
            let [row, column] = complex.coordinates(c);
            vec![column, row]
        },
        |v| sign * v,
    );
    to_js(Barcode {
        bars,
        summary: format!(
            "{width}×{height} image, {} cells ({}-construction), {} sets; {}",
            complex.n_cells(),
            if construction == Construction::T { "T" } else { "V" },
            if superlevel { "superlevel" } else { "sublevel" },
            describe(&config)
        ),
        edges: None,
    })
}
