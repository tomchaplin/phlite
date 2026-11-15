//! `phlite.networks.directed_flag`, `ordered_tuples` and `dowker`: the complexes of [`crate::flag`] and
//! [`crate::dowker`], described lazily (see [`crate::lazy`]).

use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{
    algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars,
};

use super::graph::{parse_distances, read_graph};
use crate::{
    digraph::{Digraph, Distances},
    dowker::{DowkerComplex, Side},
    flag::{FlagComplex, Kind},
    lazy::{barcode, fits, LazyComplex},
};

/// Columns of these complexes are listed on the fly, as for Vietoris–Rips, but cost more to recompute (intersecting
/// adjacency lists, or scanning every node for Dowker), and many need several additions: merged, cached columns are
/// about 2-2.5x faster than heap columns on the directed flag complex (as on GrPPH), and never slower.
const LAZY: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "",
    columns: Columns::Merged,
    cache: true,
    store_r: false,
    row_ranks: false,
    union_find: None,
};

/// The pipeline `'networks.directed_flag'`.
pub(crate) const FLAG_DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.directed_flag",
    ..LAZY
};

/// The pipeline `'networks.ordered_tuples'`.
pub(crate) const TUPLES_DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.ordered_tuples",
    ..LAZY
};

/// The pipeline `'networks.dowker'`.
pub(crate) const DOWKER_DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.dowker",
    ..LAZY
};

/// The barcode of a lazy complex with whichever algorithm is configured, its cells as their vertices.
struct LazyRun<'a, K> {
    complex: &'a K,
    max_dim: usize,
    method: Method,
    representatives: bool,
}

impl<K: LazyComplex> AlgorithmVisitor for LazyRun<'_, K> {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let bars = barcode(self.complex, self.max_dim, self.method, algorithm, self.representatives);
        let codec = self.complex.codec();
        SortedBars::new(bars, self.max_dim + 1, |simplex| {
            (simplex.dimension(), codec.vertices(*simplex).to_vec())
        })
    }
}

/// Reduce `complex` in dimensions up to `max_dim`, after checking its simplices of dimension `max_dim + 1` fit.
#[allow(clippy::too_many_arguments)]
fn run<'py, K: LazyComplex>(
    py: Python<'py>,
    complex: &K,
    max_dim: usize,
    method: &str,
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
    defaults: &AlgorithmDefaults,
    labels: Option<Vec<Py<PyAny>>>,
) -> PyResult<Bound<'py, PyAny>> {
    if !fits(complex, max_dim) {
        return Err(PyValueError::new_err(format!(
            "Too many nodes ({}) to index simplices of dimension {}",
            complex.n_vertices(),
            max_dim + 1
        )));
    }
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), defaults)?;
    let bars = py.detach(|| {
        algorithm.run(LazyRun {
            complex,
            max_dim,
            method,
            representatives,
        })
    })?;
    bars.into_python(py, &algorithm, representatives, labels.as_deref())
}

/// Compute the persistent homology of the directed flag complex of a weighted digraph (as flagser), in dimensions
/// `0, ..., max_dim`.
///
/// The directed flag complex has a simplex for each sequence of distinct nodes `(x_0, ..., x_k)` with an edge
/// `x_i -> x_j` whenever `i < j` (a directed clique, ordered by its edges), entering when its last edge does (flagser's
/// `max` filtration; nodes at 0). With `grounded=True` it is grounded as GrPPH is (as grpphati's grounded pipeline with
/// `DirectedFlagComplexHomology`): the digraph's own edges are present from time 0, and the rest of the complex is
/// the shortest-path filtration's, so that the bars in dimension 1 are the digraph's cycles, until they become
/// boundaries; only dimensions 0 and 1 then make sense.
///
/// Call either as `directed_flag(G)` with a networkx `DiGraph`, or as `directed_flag(n_vertices, edges)`.
///
/// Args:
///     graph: Either a networkx `DiGraph` (or `MultiDiGraph`), whose edge weights are read from the `weight` attribute
///         (default 1); or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the weighted edges, as `(source, target, weight)` tuples.
///     max_dim: The maximum homology dimension to compute (at most 1 if `grounded`).
///     weight: Only for a networkx graph: the edge attribute holding the weight (default `'weight'`).
///     filtration: When the edge `a -> b` enters: `'weight'`, its own weight (the lightest of parallel edges, and never
///         if there is none; the default, as flagser), or `'shortest_path'`, the length of the shortest directed path
///         from `a` to `b` (as grpphati; the default, and the only choice, if `grounded`).
///     grounded: Ground the filtration at the digraph's edges (see above).
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'networks.directed_flag'`): columns='merged', cache=True, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar. Cells are tuples of nodes (node
///         labels for a networkx graph), in their order in the simplex.
///
/// Weights must be non-negative and may be infinite (the edge never appears, but grounds). Self-loops are not supported.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(module = "phlite.networks")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, max_dim = 1, *, weight = "weight", filtration = None, grounded = false, method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn directed_flag<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[networkx.DiGraph, builtins.int]", imports = ("builtins", "networkx", "typing")))]
    graph: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Sequence[tuple[builtins.int, builtins.int, builtins.float]]]", imports = ("builtins", "typing")))]
    edges: Option<Vec<(u32, u32, f64)>>,
    max_dim: usize,
    weight: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Literal['weight', 'shortest_path']]", imports = ("typing")))]
    filtration: Option<&str>,
    grounded: bool,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (n_vertices, edges, labels) = read_graph(graph, edges, weight, "directed_flag", u32::MAX - 1)?;
    let complex = if grounded {
        if filtration.is_some_and(|f| f != "shortest_path") {
            return Err(PyValueError::new_err(
                "The grounded directed flag complex is built on the shortest-path filtration",
            ));
        }
        if max_dim > 1 {
            return Err(PyValueError::new_err(
                "Only dimensions 0 and 1 make sense for the grounded directed flag complex (max_dim must be 0 or 1)",
            ));
        }
        let graph = Digraph::shortest_paths(n_vertices, &edges);
        FlagComplex::grounded(Kind::DirectedFlag, graph, edges.iter().map(|&(a, b, _)| (a, b)))
    } else {
        let distances = parse_distances(filtration.unwrap_or("weight"))?;
        FlagComplex::new(Kind::DirectedFlag, Digraph::with_distances(n_vertices, &edges, distances))
    };
    run(py, &complex, max_dim, method, algorithm, representatives, &FLAG_DEFAULTS, labels)
}

/// Compute the persistent homology of the ordered-tuple complex of a weighted digraph (Turner's directed Rips; the
/// survey's GVR-b), in dimensions `0, ..., max_dim`.
///
/// The ordered-tuple complex has a simplex for each sequence of nodes `(x_0, ..., x_k)`, with no node next to itself,
/// with an edge `x_i -> x_j` whenever `i < j` and `x_i != x_j`, entering when its last such edge does. So it is the
/// directed flag complex with repeated nodes allowed: `(a, b, a)` is a simplex as soon as `a -> b` and `b -> a` are
/// edges, so a pair of opposite edges is filled in at once (as grpphati's `OrderedTuplesHomology`, in dimension 1).
/// Faces with a node next to itself are degenerate, and left out of the boundary. The complex grows quickly with the
/// dimension.
///
/// Call either as `ordered_tuples(G)` with a networkx `DiGraph`, or as `ordered_tuples(n_vertices, edges)`.
///
/// Args:
///     graph: Either a networkx `DiGraph` (or `MultiDiGraph`), whose edge weights are read from the `weight` attribute
///         (default 1); or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the weighted edges, as `(source, target, weight)` tuples.
///     max_dim: The maximum homology dimension to compute.
///     weight: Only for a networkx graph: the edge attribute holding the weight (default `'weight'`).
///     filtration: When the edge `a -> b` enters: `'weight'` (the default) or `'shortest_path'`, as for `directed_flag`.
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'networks.ordered_tuples'`): columns='merged', cache=True, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar (cells as for `directed_flag`).
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(module = "phlite.networks")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, max_dim = 1, *, weight = "weight", filtration = "weight", method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn ordered_tuples<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[networkx.DiGraph, builtins.int]", imports = ("builtins", "networkx", "typing")))]
    graph: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Sequence[tuple[builtins.int, builtins.int, builtins.float]]]", imports = ("builtins", "typing")))]
    edges: Option<Vec<(u32, u32, f64)>>,
    max_dim: usize,
    weight: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['weight', 'shortest_path']", imports = ("typing")))]
    filtration: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (n_vertices, edges, labels) = read_graph(graph, edges, weight, "ordered_tuples", u32::MAX - 1)?;
    let distances = parse_distances(filtration)?;
    let complex = FlagComplex::new(Kind::OrderedTuples, Digraph::with_distances(n_vertices, &edges, distances));
    run(py, &complex, max_dim, method, algorithm, representatives, &TUPLES_DEFAULTS, labels)
}

/// Compute the persistent homology of the Dowker sink or source complex of a weighted digraph (Chowdhury and Mémoli),
/// in dimensions `0, ..., max_dim`.
///
/// With `ω(a, b)` the weight of the edge `a -> b` (infinite if there is none, and `ω(a, a) = 0`), a set of nodes is
/// a simplex of the **sink** complex once some node `x` (perhaps one of them) is within `t` of each of them,
/// `ω(v, x) <= t`, so it enters at `min_x max_v ω(v, x)`; of the **source** complex, likewise with `ω(x, v)`. By
/// Dowker duality the two have the same barcode. Every set of nodes with a common sink is a simplex, so Dowker
/// complexes are dense: they suit small digraphs and low dimensions.
///
/// Call either as `dowker(G)` with a networkx `DiGraph`, or as `dowker(n_vertices, edges)`.
///
/// Args:
///     graph: Either a networkx `DiGraph` (or `MultiDiGraph`), whose edge weights are read from the `weight` attribute
///         (default 1); or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the weighted edges, as `(source, target, weight)` tuples.
///     max_dim: The maximum homology dimension to compute.
///     weight: Only for a networkx graph: the edge attribute holding the weight (default `'weight'`).
///     side: `'sink'` (the default) or `'source'`.
///     filtration: What `ω(a, b)` is: `'weight'`, the edge's own weight (the default), or `'shortest_path'`, the length
///         of the shortest directed path from `a` to `b`.
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'networks.dowker'`): columns='merged', cache=True, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar. Cells are tuples of nodes (node
///         labels for a networkx graph), in increasing order of node index.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(module = "phlite.networks")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, max_dim = 1, *, weight = "weight", side = "sink", filtration = "weight", method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn dowker<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[networkx.DiGraph, builtins.int]", imports = ("builtins", "networkx", "typing")))]
    graph: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Sequence[tuple[builtins.int, builtins.int, builtins.float]]]", imports = ("builtins", "typing")))]
    edges: Option<Vec<(u32, u32, f64)>>,
    max_dim: usize,
    weight: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['sink', 'source']", imports = ("typing")))]
    side: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['weight', 'shortest_path']", imports = ("typing")))]
    filtration: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (n_vertices, edges, labels) = read_graph(graph, edges, weight, "dowker", 1 << 16)?;
    let side = match side {
        "sink" => Side::Sink,
        "source" => Side::Source,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown side {other:?}, expected 'sink' or 'source'"
            )))
        }
    };
    let distances: Distances = parse_distances(filtration)?;
    let complex = DowkerComplex::new(&Digraph::with_distances(n_vertices, &edges, distances), side);
    run(py, &complex, max_dim, method, algorithm, representatives, &DOWKER_DEFAULTS, labels)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(directed_flag, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_tuples, m)?)?;
    m.add_function(wrap_pyfunction!(dowker, m)?)
}
