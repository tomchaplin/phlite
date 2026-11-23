//! `phlite.networks.pph`: persistent path homology of a weighted digraph in any dimension, by [`crate::pph`].

use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
    fields::Z2,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{
    algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars,
};

use super::{
    graph::{network_error, read_graph},
    grpph::path_homology,
};
use crate::{
    grpph::build_filtration,
    pph::{pph_barcode_with, Network},
};

/// The pipeline `'networks.pph'`.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.pph",
    columns: Columns::Merged,
    cache: true,
    store_r: false,
    row_ranks: false,
    union_find: None,
    collapse: None,
};

/// Compute the persistent path homology (PPH) of a weighted digraph (Chowdhury and Mémoli), in dimensions
/// `0, ..., max_dim`.
///
/// At time `t` the digraph has the edges entering by `t`, and `PH_p` is the persistent homology of its chain complexes
/// of ∂-invariant paths. With `filtration='weight'` (the default, Chowdhury and Mémoli's) an edge enters at its
/// weight; with `filtration='shortest_path'` the pair `a -> b` enters at the length of the shortest directed path from
/// `a` to `b` (as grpphati's `PPH`, and as GrPPH without the grounding).
///
/// Computed by reducing the boundary of the regular path complex, each path entering once all its edges have (its
/// *allow time*), with a bar `[t(σ), max(t(τ), t(σ)))` for each pair: the algorithm of Chowdhury and Mémoli, with
/// phlite's optimisations (clearing, cohomology, apparent pairs). It grows quickly with the dimension. (In dimensions
/// 0 and 1 of the shortest-path filtration, GrPPH's matrices without the grounding give the same bars faster, so are
/// used instead; their representatives are sums of nodes or of edges.)
///
/// Call either as `pph(G)` with a networkx `DiGraph`, or as `pph(n_vertices, edges)`.
///
/// Args:
///     graph: Either a networkx `DiGraph` (or `MultiDiGraph`), whose edge weights are read from the `weight` attribute
///         (default 1); or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the weighted edges, as `(source, target, weight)` tuples.
///     max_dim: The maximum homology dimension to compute (default 1).
///     weight: Only for a networkx graph: the edge attribute holding the weight (default `'weight'`).
///     filtration: `'weight'` (the default) or `'shortest_path'`, see above.
///     method: `'cohomology'` (the default, and much faster) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'networks.pph'`): columns='merged', cache=True, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar: a sum of allowed regular paths
///         (tuples of nodes), a ∂-invariant cycle from the bar's birth which becomes a boundary at its death.
///
/// Weights must be non-negative and may be infinite (the edge never appears). Self-loops are not supported.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(
    python_overload = r#"
    import builtins
    import typing
    import networkx
    import phlite.core

    @typing.overload
    def pph(
        graph: networkx.DiGraph,
        /,
        max_dim: builtins.int = 1,
        *,
        weight: builtins.str = "weight",
        filtration: typing.Literal['weight', 'shortest_path'] = "weight",
        method: typing.Literal['cohomology', 'homology'] = "cohomology",
        algorithm: typing.Optional[core.Algorithm] = None,
        representatives: builtins.bool = False,
    ) -> core.Barcode: ...
    @typing.overload
    def pph(
        n_vertices: builtins.int,
        edges: typing.Sequence[tuple[builtins.int, builtins.int, builtins.float]],
        /,
        max_dim: builtins.int = 1,
        *,
        filtration: typing.Literal['weight', 'shortest_path'] = "weight",
        method: typing.Literal['cohomology', 'homology'] = "cohomology",
        algorithm: typing.Optional[core.Algorithm] = None,
        representatives: builtins.bool = False,
    ) -> core.Barcode: ...
    "#,
    no_default_overload = true,
    module = "phlite.networks"
)]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, max_dim = 1, *, weight = "weight", filtration = "weight", method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn pph<'py>(
    py: Python<'py>,
    graph: &Bound<'_, PyAny>,
    edges: Option<Vec<(u32, u32, f64)>>,
    max_dim: usize,
    weight: &str,
    filtration: &str,
    method: &str,
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let shortest_path = match filtration {
        "weight" => false,
        "shortest_path" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown filtration {other:?}, expected 'weight' or 'shortest_path'"
            )))
        }
    };
    if shortest_path && max_dim <= 1 {
        // The GrPPH oracle without the grounding computes the same bars (tests/pph_vs_grpph.rs), 2-2.5x faster
        return path_homology(py, graph, edges, weight, method, algorithm, representatives, false, &DEFAULTS);
    }
    let (n_vertices, edges, labels) = read_graph(graph, edges, weight, "pph", u16::MAX as u32)?;
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let network = py.detach(|| {
        if shortest_path {
            let distances = build_filtration(n_vertices, &edges)?;
            let edges: Vec<(u16, u16, f64)> = distances
                .iter()
                .enumerate()
                .flat_map(|(a, row)| {
                    row.iter()
                        .enumerate()
                        .filter_map(move |(b, d)| d.map(|d| (a as u16, b as u16, d.into_inner())))
                })
                .collect();
            Network::from_edges(n_vertices as u16, &edges)
        } else {
            let edges: Vec<(u16, u16, f64)> =
                edges.iter().map(|&(a, b, w)| (a as u16, b as u16, w)).collect();
            Network::from_edges(n_vertices as u16, &edges)
        }
    })
    .map_err(network_error)?;
    if max_dim >= network.max_dimension() {
        return Err(PyValueError::new_err(format!(
            "Too many nodes ({n_vertices}) to encode paths of dimension {}",
            max_dim + 1
        )));
    }
    let bars = py.detach(|| {
        algorithm.run(PPHRun {
            network: &network,
            max_dim,
            method,
            representatives,
            parallel_setup: algorithm.parallel_setup(),
        })
    })?;
    bars.into_python(py, &algorithm, representatives, labels.as_deref())
}

/// Compute the PPH barcode with whichever algorithm is configured.
struct PPHRun<'a> {
    network: &'a Network,
    max_dim: usize,
    method: Method,
    representatives: bool,
    parallel_setup: bool,
}

impl AlgorithmVisitor for PPHRun<'_> {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let bars = pph_barcode_with::<Z2, R>(
            self.network,
            self.max_dim,
            self.method,
            algorithm,
            self.representatives,
            self.parallel_setup,
        );
        SortedBars::new(bars, self.max_dim + 1, |path| {
            (
                path.dimension(),
                self.network.vertices(*path).into_iter().map(u32::from).collect(),
            )
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pph, m)?)
}
