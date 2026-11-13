//! phlite compiled to WebAssembly for the browser playground.
//!
//! Four pipelines, each taking plain arrays and an [`Options`] object and returning a [`Barcode`]:
//!
//! * [`rips`]: Vietoris–Rips persistence of a point cloud (any dimension) or a distance matrix, by `phlite_rips`;
//! * [`grpph`]: grounded persistent path homology of a weighted digraph, by `phlite_grpph`;
//! * [`image`]: persistence of a greyscale image (its cubical complex, pixels as vertices, sublevel or superlevel
//!   sets), by `phlite_py`'s explicit complexes;
//! * [`alpha`]: alpha persistence of points in the plane (the Delaunay triangulation filtered by `alphalpha`), by
//!   `phlite_py`'s explicit complexes;
//! * [`complex`]: persistence of an explicit filtered chain complex over Z/2, Z/3, Z/5 or Z/7, by `phlite_py`.
//!
//! The parallel algorithm runs on the global rayon pool, which [`init_thread_pool`] starts on Web Workers
//! (`wasm-bindgen-rayon`): the page must be cross-origin isolated (COOP/COEP headers) for shared memory.

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Bar, Method},
    columns::{ColumnRepresentation, HeapColumns, MergedColumns, PivotColumns, SortedColumns},
    fields::{Z2, Z3, Z5, Z7},
    reduction::{Clearing, Standard, Storage},
};
use phlite_py::{
    complex::{Chain, Complex, RawChain},
    field::IntField,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub use wasm_bindgen_rayon::init_thread_pool;

/// Log panics to the console (call once, at start-up).
#[wasm_bindgen]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

/// The number of threads in the global pool.
#[wasm_bindgen]
pub fn n_threads() -> usize {
    rayon::current_num_threads()
}

// ---- Options ----

/// How to compute: the reduction algorithm and its options, the method and whether to compute representatives.
/// Every field is optional; a missing one takes the pipeline's default (the same as its Python package's
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

/// A pipeline's defaults, as in its Python package.
#[derive(Debug, Clone, Copy)]
struct Defaults {
    columns: Columns,
    cache: bool,
    store_r: bool,
    /// Whether the matrices know the rank of every row, which pivot columns need.
    row_ranks: bool,
}

const RIPS_DEFAULTS: Defaults = Defaults {
    columns: Columns::Heap,
    cache: false,
    store_r: false,
    row_ranks: false,
};
const GRPPH_DEFAULTS: Defaults = Defaults {
    columns: Columns::Merged,
    cache: true,
    store_r: false,
    row_ranks: false,
};
const COMPLEX_DEFAULTS: Defaults = Defaults {
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    row_ranks: true,
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
}

fn err(message: impl Into<String>) -> JsError {
    JsError::new(&message.into())
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
        Ok(Self {
            kind,
            columns,
            cache: options.cache.unwrap_or(defaults.cache),
            store_r,
            skip_zero_apparent_pairs: skip,
            storage,
            method,
            representatives: options.representatives,
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
}

#[derive(Debug, Serialize)]
pub struct Barcode {
    pub bars: Vec<OutBar>,
    /// What was computed, in words (sizes and the resolved algorithm).
    pub summary: String,
}

fn to_js(barcode: Barcode) -> Result<JsValue, JsError> {
    barcode
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| err(e.to_string()))
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
    if config.kind == Kind::Parallel {
        s += &format!(
            ", {} storage on {} threads",
            match config.storage {
                Storage::LockFree => "lock-free",
                _ => "locking",
            },
            rayon::current_num_threads()
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
#[wasm_bindgen]
pub fn rips(
    coords: &[f64],
    point_dim: usize,
    max_dim: usize,
    threshold: Option<f64>,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let config = Config::resolve(options, RIPS_DEFAULTS)?;
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
        let points: Vec<&[f64]> = coords.chunks(point_dim).collect();
        points
            .iter()
            .map(|p| {
                points
                    .iter()
                    .map(|q| {
                        not_nan(
                            p.iter()
                                .zip(*q)
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum::<f64>()
                                .sqrt(),
                        )
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<_>>, _>>()?
    };
    let n = distances.len();
    if n == 0 {
        return to_js(Barcode {
            bars: vec![],
            summary: "No points".into(),
        });
    }
    // usize is 32 bits here, so index simplices by u64: up to 65 534 points for H₂ rather than 254
    if !phlite_rips::RipsIndex::<u64>::fits(n, max_dim + 1) {
        return Err(err(format!(
            "Too many points ({n}) for simplices of dimension {}",
            max_dim + 1
        )));
    }
    let threshold = not_nan(threshold.unwrap_or(f64::INFINITY))?;

    struct V {
        distances: Vec<Vec<NotNan<f64>>>,
        max_dim: usize,
        threshold: NotNan<f64>,
        method: Method,
        representatives: bool,
    }
    impl Visitor for V {
        type Output = Vec<phlite_rips::RipsBar<u64>>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            phlite_rips::rips_barcode::<_, u64>(
                self.distances,
                self.max_dim,
                self.threshold,
                self.method,
                algorithm,
                self.representatives,
            )
        }
    }
    let bars = config.run(V {
        distances,
        max_dim,
        threshold,
        method: config.method,
        representatives: config.representatives,
    });
    let bars = translate::<_, Z2>(
        bars,
        |s| s.dimension(n),
        |s| s.vertices(n).iter().map(|&v| v as u32).collect(),
        |v| v,
    );
    to_js(Barcode {
        bars,
        summary: format!("{n} points, dimensions 0–{max_dim}; {}", describe(&config)),
    })
}

fn not_nan(x: f64) -> Result<NotNan<f64>, JsError> {
    NotNan::new(x).map_err(|_| err("NaN in the input"))
}

// ---- GrPPH ----

/// Grounded persistent path homology of a weighted digraph with `n_vertices` nodes and edges
/// `sources[i] → targets[i]` of weight `weights[i]`.
#[wasm_bindgen]
pub fn grpph(
    n_vertices: u32,
    sources: &[u32],
    targets: &[u32],
    weights: &[f64],
    options: JsValue,
) -> Result<JsValue, JsError> {
    let config = Config::resolve(options, GRPPH_DEFAULTS)?;
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
    let filtration = phlite_grpph::build_filtration(n_vertices, &edges);
    let edge_set = phlite_grpph::build_edge_set(&edges);

    struct V<'a> {
        filtration: &'a phlite_grpph::coboundary::VecDigraphFiltration,
        edge_set: &'a phlite_grpph::coboundary::DigraphEdgeSet,
        n: u16,
        method: Method,
        representatives: bool,
    }
    impl Visitor for V<'_> {
        type Output = Vec<phlite_grpph::GrPPHBar>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            phlite_grpph::grpph_barcode(
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
    use phlite_grpph::coboundary::{PathHom2Cell, PathHomCell};
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
            "{n_vertices} nodes, {} edges, dimensions 0–1; {}",
            edges.len(),
            describe(&config)
        ),
    })
}

// ---- Explicit complexes ----

/// The barcode of `chain` filtered by `values` (with `NaN` for a value to fill in from the cell's faces).
fn complex_bars<CF: IntField + Send + Sync + 'static>(
    raw: RawChain,
    values: Option<Vec<f64>>,
    max_dim: Option<usize>,
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

    struct V<'a, CF> {
        complex: &'a Complex<CF>,
        max_dim: Option<usize>,
        method: Method,
        representatives: bool,
    }
    impl<CF: IntField + Send + Sync + 'static> Visitor for V<'_, CF> {
        type Output = Vec<Bar<phlite_py::complex::Cell, NotNan<f64>, CF>>;
        fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
            self.complex
                .barcode(self.max_dim, self.method, algorithm, self.representatives)
        }
    }
    let bars = config.run(V {
        complex: &complex,
        max_dim,
        method: config.method,
        representatives: config.representatives,
    });
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
        2 => complex_bars::<Z2>(raw, values, max_dim, &config)?,
        3 => complex_bars::<Z3>(raw, values, max_dim, &config)?,
        5 => complex_bars::<Z5>(raw, values, max_dim, &config)?,
        7 => complex_bars::<Z7>(raw, values, max_dim, &config)?,
        other => {
            return Err(err(format!(
                "Unsupported modulus {other}; use 2, 3, 5 or 7"
            )))
        }
    };
    to_js(Barcode {
        bars,
        summary: format!("{sizes}, over Z/{modulus}; {}", describe(&config)),
    })
}

// ---- Alpha ----

/// Alpha persistence of points in the plane (`coords` as `x0, y0, x1, y1, ...`): the Delaunay triangulation filtered
/// by the alpha filtration, with values the radii (not squared radii) at which cells appear. Cells are given by their
/// vertices (input point indices); repeated points are merged into the first.
#[wasm_bindgen]
pub fn alpha(coords: &[f64], options: JsValue) -> Result<JsValue, JsError> {
    use spade::{Point2, Triangulation};
    use std::collections::HashMap;
    let config = Config::resolve(options, COMPLEX_DEFAULTS)?;
    if coords.len() % 2 != 0 {
        return Err(err("Alpha persistence needs points in the plane"));
    }
    if coords.iter().any(|c| !c.is_finite()) {
        return Err(err("Coordinates must be finite"));
    }
    let n = coords.len() / 2;
    if n < 3 {
        return Err(err("Alpha persistence needs at least three points"));
    }
    let points: Vec<Point2<f64>> = coords.chunks(2).map(|p| Point2::new(p[0], p[1])).collect();
    // spade numbers vertices its own way (and merges repeats), so find each one's input point by position
    let mut by_position: HashMap<(u64, u64), u32> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        by_position
            .entry((p.x.to_bits(), p.y.to_bits()))
            .or_insert(i as u32);
    }
    let filtration = alphalpha::alpha_filtration(points);
    let input_index = |v: spade::handles::FixedVertexHandle| {
        let p = filtration.vertex(v).position();
        by_position[&(p.x.to_bits(), p.y.to_bits())]
    };

    // The complex: vertices, then edges, then triangles, each cell's vertices kept for the output
    let mut cells: Vec<Vec<u32>> = vec![];
    let mut raw = RawChain {
        indptr: vec![0],
        ..RawChain::default()
    };
    let mut values = vec![];
    let mut vertex_cell = HashMap::new();
    for v in filtration.fixed_vertices() {
        vertex_cell.insert(v, cells.len() as i64);
        cells.push(vec![input_index(v)]);
        raw.dimensions.push(0);
        raw.indptr.push(raw.indices.len() as i64);
        values.push(0.0);
    }
    let mut edge_cell = HashMap::new();
    for e in filtration.fixed_undirected_edges() {
        let edge = filtration.undirected_edge(e);
        let [a, b] = edge.vertices().map(|v| v.fix());
        edge_cell.insert(e, cells.len() as i64);
        let mut vs = vec![input_index(a), input_index(b)];
        vs.sort_unstable();
        cells.push(vs);
        raw.dimensions.push(1);
        raw.indices.extend([vertex_cell[&a], vertex_cell[&b]]);
        raw.indptr.push(raw.indices.len() as i64);
        values.push(edge.data().filtration.max(0.0).sqrt());
    }
    for f in filtration.fixed_inner_faces() {
        let face = filtration.face(f);
        let mut vs: Vec<u32> = face
            .vertices()
            .iter()
            .map(|v| input_index(v.fix()))
            .collect();
        vs.sort_unstable();
        cells.push(vs);
        raw.dimensions.push(2);
        raw.indices.extend(
            face.adjacent_edges()
                .iter()
                .map(|e| edge_cell[&e.as_undirected().fix()]),
        );
        raw.indptr.push(raw.indices.len() as i64);
        values.push(face.data().filtration.max(0.0).sqrt());
    }
    let n_cells = cells.len();
    let (mut bars, _) = complex_bars::<Z2>(raw, Some(values), Some(1), &config)?;
    let vertices = |c: &mut Vec<u32>| *c = cells[c[0] as usize].clone();
    for bar in &mut bars {
        vertices(&mut bar.birth_cell);
        if let Some(c) = &mut bar.death_cell {
            vertices(c);
        }
        if let Some(rep) = &mut bar.rep {
            for (_, c) in rep {
                vertices(c);
            }
        }
    }
    to_js(Barcode {
        bars,
        summary: format!(
            "{n} points, Delaunay triangulation of {n_cells} cells, dimensions 0–1, alpha radii; {}",
            describe(&config)
        ),
    })
}

// ---- Images ----

/// The cubical complex of a `width × height` image, with pixels as vertices: cells at the points of the doubled grid
/// of size `(2·width − 1) × (2·height − 1)`, a cell's dimension being how many of its coordinates are odd.
///
/// Cells are numbered by dimension (vertices, then edges, then squares), so that every face comes before its cofaces:
/// phlite breaks ties between equal values by number, and a face must enter first. Returns the chain and, for every
/// cell, its position `y · (2·width − 1) + x` on the doubled grid.
fn cubical_chain(width: usize, height: usize) -> (RawChain, Vec<u32>) {
    let (w2, h2) = (2 * width - 1, 2 * height - 1);
    let n = w2 * h2;
    let dim = |pos: usize| ((pos % w2) % 2 + (pos / w2) % 2) as i64;
    let mut position: Vec<u32> = (0..n as u32).collect();
    position.sort_by_key(|&pos| dim(pos as usize));
    let mut number = vec![0i64; n];
    for (cell, &pos) in position.iter().enumerate() {
        number[pos as usize] = cell as i64;
    }
    let mut raw = RawChain {
        dimensions: Vec::with_capacity(n),
        indptr: Vec::with_capacity(n + 1),
        indices: Vec::with_capacity(2 * n),
        coefficients: None,
    };
    raw.indptr.push(0);
    for &pos in &position {
        let pos = pos as usize;
        let (x, y) = (pos % w2, pos / w2);
        let id = |x: usize, y: usize| number[y * w2 + x];
        raw.dimensions.push(dim(pos));
        if x % 2 == 1 {
            raw.indices.extend([id(x - 1, y), id(x + 1, y)]);
        }
        if y % 2 == 1 {
            raw.indices.extend([id(x, y - 1), id(x, y + 1)]);
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    (raw, position)
}

/// Persistence of a greyscale image (`pixels` row by row), by the cubical complex with pixels as vertices.
/// With `superlevel`, bright features come first (the filtration is by decreasing intensity, and the values returned
/// are intensities, so bars run downwards). Representatives and cells are doubled grid coordinates `[x, y]`.
#[wasm_bindgen]
pub fn image(
    pixels: &[f64],
    width: usize,
    height: usize,
    superlevel: bool,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let config = Config::resolve(options, COMPLEX_DEFAULTS)?;
    if width == 0 || height == 0 || pixels.len() != width * height {
        return Err(err("The image size doesn't match its pixels"));
    }
    let w2 = 2 * width - 1;
    let sign = if superlevel { -1.0 } else { 1.0 };
    let (chain, position) = cubical_chain(width, height);
    // Values for the vertices (pixels); the other cells take the largest of their faces'
    let mut values = vec![f64::NAN; position.len()];
    for (cell, &pos) in position.iter().enumerate() {
        let (x, y) = (pos as usize % w2, pos as usize / w2);
        if x % 2 == 0 && y % 2 == 0 {
            let v = pixels[(y / 2) * width + x / 2];
            if v.is_nan() {
                return Err(err("NaN pixel"));
            }
            values[cell] = sign * v;
        }
    }
    let (mut bars, _) = complex_bars::<Z2>(chain, Some(values), Some(1), &config)?;
    for bar in &mut bars {
        bar.birth *= sign;
        bar.death = bar.death.map(|d| d * sign);
        let coords = |c: &mut Vec<u32>| {
            let pos = position[c[0] as usize] as usize;
            *c = vec![(pos % w2) as u32, (pos / w2) as u32];
        };
        coords(&mut bar.birth_cell);
        if let Some(c) = &mut bar.death_cell {
            coords(c);
        }
        if let Some(rep) = &mut bar.rep {
            for (_, c) in rep {
                coords(c);
            }
        }
    }
    to_js(Barcode {
        bars,
        summary: format!(
            "{width}×{height} image, {} cells, {} sets; {}",
            w2 * (2 * height - 1),
            if superlevel { "superlevel" } else { "sublevel" },
            describe(&config)
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubical_two_by_two_is_a_square() {
        let (raw, position) = cubical_chain(2, 2);
        assert_eq!(raw.dimensions, vec![0, 0, 0, 0, 1, 1, 1, 1, 2]);
        assert_eq!(position, vec![0, 2, 6, 8, 1, 3, 5, 7, 4]);
        // Every face comes before its cofaces
        for cell in 0..raw.dimensions.len() {
            let faces = &raw.indices[raw.indptr[cell] as usize..raw.indptr[cell + 1] as usize];
            assert!(faces.iter().all(|&f| (f as usize) < cell));
        }
        let chain = Chain::<Z2>::from_raw(raw).unwrap();
        assert_eq!(chain.n_entries(), 4 * 2 + 4);
        // A constant image: all values tie, so this only works if faces come first
        let values = (0..9)
            .map(|c| if c < 4 { Some(0.0) } else { None })
            .collect();
        let values = chain.extend(values).unwrap();
        chain.filter(Some(values)).unwrap();
    }
}
