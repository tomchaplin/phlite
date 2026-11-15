//! `phlite.networks.extended`: extended persistence of node values on a graph (see [`crate::extended`]).

use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
    fields::{Z11, Z13, Z2, Z3, Z5, Z7},
};
use phlite_bindings::{algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use super::{flag::read, graph::network_error};
use crate::extended::{extended_persistence, ExtendedBar, Kind};
use crate::NetworkError;

/// The pipeline `'networks.extended'`: an explicit cone complex, so with `phlite.complexes`' defaults.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.extended",
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    row_ranks: true,
    union_find: None,
};

/// The extended persistence of a function on a graph: four diagrams, each a list (by dimension) of `(birth, death)`
/// pairs, sorted.
///
/// * `ordinary`: classes born and dying in the sublevel sets, going up (as ordinary persistence);
/// * `relative`: born and dying going down, in the pairs (graph, superlevel set); dimensions start at 1, as GUDHI's;
/// * `extended_plus`: born going up and dying going down, birth below death (a component, from its minimum to its
///   maximum);
/// * `extended_minus`: likewise with birth above death (a loop, from its highest point to its lowest).
#[gen_stub_pyclass]
#[pyclass(module = "phlite.networks", frozen)]
pub struct ExtendedBarcode {
    #[pyo3(get)]
    ordinary: Vec<Vec<(f64, f64)>>,
    #[pyo3(get)]
    relative: Vec<Vec<(f64, f64)>>,
    #[pyo3(get)]
    extended_plus: Vec<Vec<(f64, f64)>>,
    #[pyo3(get)]
    extended_minus: Vec<Vec<(f64, f64)>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ExtendedBarcode {
    /// The four diagrams in GUDHI's format (`SimplexTree.extended_persistence`): a list of four lists of
    /// `(dimension, (birth, death))`.
    fn to_gudhi(&self) -> Vec<Vec<(usize, (f64, f64))>> {
        [&self.ordinary, &self.relative, &self.extended_plus, &self.extended_minus]
            .iter()
            .map(|diagram| {
                diagram
                    .iter()
                    .enumerate()
                    .flat_map(|(dim, bars)| bars.iter().map(move |&bar| (dim, bar)))
                    .collect()
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        let count = |d: &Vec<Vec<(f64, f64)>>| {
            d.iter().enumerate().filter(|(_, b)| !b.is_empty()).map(|(i, b)| format!("H{i}: {}", b.len())).collect::<Vec<_>>().join(", ")
        };
        format!(
            "ExtendedBarcode(ordinary=[{}], relative=[{}], extended_plus=[{}], extended_minus=[{}])",
            count(&self.ordinary),
            count(&self.relative),
            count(&self.extended_plus),
            count(&self.extended_minus)
        )
    }
}

impl ExtendedBarcode {
    fn new(bars: Vec<ExtendedBar>, top: usize) -> Self {
        // Dimensions 0..=top + 1 (relative bars go up one)
        let empty = || vec![vec![]; top + 2];
        let (mut ordinary, mut relative, mut plus, mut minus) = (empty(), empty(), empty(), empty());
        for bar in bars {
            let diagram = match bar.kind {
                Kind::Ordinary => &mut ordinary,
                Kind::Relative => &mut relative,
                Kind::ExtendedPlus => &mut plus,
                Kind::ExtendedMinus => &mut minus,
            };
            diagram[bar.dimension].push((bar.birth, bar.death));
        }
        for diagram in [&mut ordinary, &mut relative, &mut plus, &mut minus] {
            for bars in diagram.iter_mut() {
                bars.sort_by(|a, b| a.partial_cmp(b).expect("values are finite"));
            }
        }
        Self { ordinary, relative, extended_plus: plus, extended_minus: minus }
    }
}

/// Compute the extended persistence (Cohen-Steiner, Edelsbrunner and Harer) of a function on the nodes of an
/// undirected graph: the lower-star filtration of the node values, going up through the sublevel sets and then down
/// through the pairs (graph, superlevel set), so that every class dies (see `ExtendedBarcode`). As GUDHI's
/// `SimplexTree.extend_filtration` and `extended_persistence`, with exact values (GUDHI's are rescaled, so equal up to
/// rounding), and bars of length zero omitted (as GUDHI's with its default `min_persistence=0`).
///
/// Edge weights are not used: only the node values.
///
/// Call either as `extended(G, node_weight='height')` with a networkx `Graph`, or as
/// `extended(n_vertices, edges, node_weight=values)`.
///
/// Args:
///     graph: Either a networkx `Graph`, or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the edges, as `(u, v)` (or `(u, v, weight)`)
///         tuples.
///     node_weight: The node values: a node attribute name (for a networkx graph), or a sequence with one finite value
///         per node, in the order of `G.nodes` (or `0, ..., n_vertices - 1`).
///     dim: The dimension of the complex: 1 for the graph itself (the default), or more for its flag complex (every
///         clique of at most `dim + 1` nodes).
///     modulus: The prime coefficient field: 2 (the default), 3, 5, 7, 11 or 13.
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'networks.extended'`): columns='pivot', cache=False, store_r=True.
///
/// Returns:
///     An `ExtendedBarcode`: the ordinary, relative, extended+ and extended- diagrams, by dimension.
#[gen_stub_pyfunction(module = "phlite.networks")]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, *, node_weight, dim = 1, modulus = 2, method = "cohomology", algorithm = None))]
#[allow(clippy::too_many_arguments)]
fn extended(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))] graph: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Sequence[typing.Union[typing.Tuple[builtins.int, builtins.int], typing.Tuple[builtins.int, builtins.int, builtins.float]]]]", imports = ("builtins", "typing")))]
    edges: Option<Vec<Bound<'_, PyAny>>>,
    #[gen_stub(override_type(type_repr = "typing.Union[builtins.str, typing.Sequence[builtins.float]]", imports = ("builtins", "typing")))]
    node_weight: Bound<'_, PyAny>,
    dim: usize,
    modulus: u32,
    method: &str,
    algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<ExtendedBarcode> {
    let (n_vertices, edges, values, _labels) = read(graph, edges, None, Some(&node_weight))?;
    let values = values.expect("node_weight is required");
    let edges: Vec<(u32, u32)> = edges.iter().map(|&(u, v, _)| (u, v)).collect();
    if !matches!(modulus, 2 | 3 | 5 | 7 | 11 | 13) {
        return Err(PyValueError::new_err(format!(
            "Unsupported modulus {modulus}; use 2, 3, 5, 7, 11 or 13"
        )));
    }
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py.detach(|| {
        algorithm.run(ExtendedRun { n_vertices, edges: &edges, values: &values, dim, modulus, method })
    })?;
    Ok(ExtendedBarcode::new(bars.map_err(network_error)?, dim))
}

/// Compute extended persistence with whichever algorithm is configured.
struct ExtendedRun<'a> {
    n_vertices: u32,
    edges: &'a [(u32, u32)],
    values: &'a [f64],
    dim: usize,
    modulus: u32,
    method: Method,
}

impl AlgorithmVisitor for ExtendedRun<'_> {
    type Output = Result<Vec<ExtendedBar>, NetworkError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        macro_rules! run {
            ($field:ty) => {
                extended_persistence::<$field, R>(self.n_vertices, self.edges, self.values, self.dim, self.method, algorithm)
            };
        }
        match self.modulus {
            2 => run!(Z2),
            3 => run!(Z3),
            5 => run!(Z5),
            7 => run!(Z7),
            11 => run!(Z11),
            _ => run!(Z13),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExtendedBarcode>()?;
    m.add_function(wrap_pyfunction!(extended, m)?)
}
