//! `phlite.networks.magnitude`: magnitude homology of a digraph, by [`crate::magnitude`].

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use pyo3_stub_gen::derive::*;

use super::graph::{read_graph, validate};
use crate::magnitude::{
    magnitude_integral, magnitude_ranks, total_ranks, MagnitudeGraph, MagnitudeQuery, PairGroups, PairRanks, MODULI,
};
use crate::mpss::mpss_backend;

/// Magnitude homology of a digraph, as defined by Hepworth and Willerton: the ranks of `MH_{k,l}` over `Z/modulus`.
///
/// The chain group `MC_{k,l}` is spanned by the tuples `(x_0, ..., x_k)` of nodes with `x_i ≠ x_{i+1}` and a
/// directed path from each to the next, whose lengths (shortest-path distances, every edge of length 1: weights are
/// ignored) add up to `l`; the differential deletes inner nodes that lie on a geodesic between their neighbours. The
/// complex splits over the pairs `(s, t)` of first and last nodes, and each piece is reduced on its own, in parallel,
/// with the collapsible pairs of cells found first (nearly all of them).
///
/// Call either as `magnitude(G, l_max=...)` with a networkx `DiGraph` (for an undirected graph `H`, pass
/// `H.to_directed()`), or as `magnitude(n_vertices, edges, l_max=...)` with edges as `(source, target)` (or
/// `(source, target, weight)`) tuples.
///
/// Args:
///     graph: Either a networkx `DiGraph`, or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the edges.
///     l_max: The largest length `l` to compute.
///     k_max: The largest degree `k` to compute (default: `l_max`). Chains up to degree `k_max + 1` are listed.
///     pairs: Only these pairs `(s, t)` of nodes (default: every pair). Tuples are only extended while some target is
///         within reach, so a few pairs are much cheaper than all of them.
///     modulus: The characteristic of the coefficient field: 2 (the default), 3, 5, 7, 11, 13 or 251; or a list of
///         them, computed together (each piece is collapsed once). Comparing the ranks over two fields finds torsion:
///         by universal coefficients, if the ranks over `Z/p` and `Z/251` differ in degree `k` or `k + 1`, the integral
///         group `MH_{k,l}` has `p`-torsion (unless it has 251-torsion).
///     by_pair: Whether to return the ranks of each pair separately.
///     integral: Whether to compute over the integers instead (`modulus` is then ignored): each piece is collapsed and
///         eliminated over Z, and what is left (nearly always a few cells) put in Smith normal form.
///
/// Returns:
///     A `MagnitudeHomology`: the ranks, indexed `[l][k]` as a list over `l = 0, ..., l_max` of lists over
///     `k = 0, ..., min(l, k_max)` (summed over the pairs), with `rank(k, l)`, `group(k, l)`, `table()` (also how it
///     prints) and `magnitude_coefficients()`. With `by_pair`, a dict from each pair `(s, t)` with some non-zero rank
///     to its own. With a list of moduli, a dict from each modulus to its result. With `integral`, over the integers:
///     free ranks (`None` if its Smith form overflowed or was too large) and `torsion(k, l)`, the invariant factors of
///     the torsion as `{order: multiplicity}`.
#[gen_stub_pyfunction(
    python_overload = r#"
    import builtins
    import typing
    import networkx

    @typing.overload
    def magnitude(
        graph: networkx.DiGraph,
        /,
        l_max: builtins.int,
        *,
        k_max: typing.Optional[builtins.int] = None,
        pairs: typing.Optional[typing.Sequence[tuple[typing.Any, typing.Any]]] = None,
        modulus: builtins.int | typing.Sequence[builtins.int] = 2,
        by_pair: builtins.bool = False,
        integral: builtins.bool = False,
    ) -> typing.Union[MagnitudeHomology, typing.Dict[typing.Any, MagnitudeHomology]]: ...
    @typing.overload
    def magnitude(
        n_vertices: builtins.int,
        edges: typing.Sequence[tuple[builtins.int, builtins.int] | tuple[builtins.int, builtins.int, builtins.float]],
        /,
        l_max: builtins.int,
        *,
        k_max: typing.Optional[builtins.int] = None,
        pairs: typing.Optional[typing.Sequence[tuple[builtins.int, builtins.int]]] = None,
        modulus: builtins.int | typing.Sequence[builtins.int] = 2,
        by_pair: builtins.bool = False,
        integral: builtins.bool = False,
    ) -> typing.Union[MagnitudeHomology, typing.Dict[typing.Any, MagnitudeHomology]]: ...
    "#,
    no_default_overload = true,
    module = "phlite.networks"
)]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, l_max = None, *, k_max = None, pairs = None, modulus = None, by_pair = false, integral = false))]
#[allow(clippy::too_many_arguments)]
fn magnitude<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    edges: Option<&Bound<'py, PyAny>>,
    l_max: Option<usize>,
    k_max: Option<usize>,
    pairs: Option<&Bound<'py, PyAny>>,
    modulus: Option<&Bound<'py, PyAny>>,
    by_pair: bool,
    integral: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // (2 by default)
    let (moduli, several): (Vec<u64>, bool) = match modulus {
        None => (vec![2], false),
        Some(modulus) => match modulus.extract::<u64>() {
            Ok(m) => (vec![m], false),
            Err(_) => (
                modulus
                    .extract::<Vec<u64>>()
                    .map_err(|_| PyTypeError::new_err("modulus must be an int or a list of ints"))?,
                true,
            ),
        },
    };
    for m in &moduli {
        if !MODULI.contains(m) {
            return Err(PyValueError::new_err(format!("Unsupported modulus {m}, expected one of {MODULI:?}")));
        }
    }
    let l_max = l_max.ok_or_else(|| PyTypeError::new_err("magnitude needs l_max, the largest length to compute"))?;
    let edges = read_edges(edges)?;
    let (n_vertices, edges, labels) = read_graph(graph, edges, "weight", "magnitude", u32::MAX >> 1)?;
    validate(n_vertices, &edges, u32::MAX >> 1)?;
    // Pairs, by node label if read from networkx
    let index_of = |node: &Bound<'py, PyAny>| -> PyResult<u32> {
        match &labels {
            None => node.extract::<u32>().ok().filter(|&v| v < n_vertices).ok_or_else(|| {
                PyValueError::new_err(format!("Pair node {node} is not one of 0..{n_vertices}"))
            }),
            Some(labels) => {
                for (i, label) in labels.iter().enumerate() {
                    if label.bind(py).eq(node)? {
                        return Ok(i as u32);
                    }
                }
                Err(PyValueError::new_err(format!("Pair node {node} is not a node of the graph")))
            }
        }
    };
    let pairs: Option<Vec<(u32, u32)>> = pairs
        .map(|pairs| -> PyResult<Vec<(u32, u32)>> {
            pairs
                .try_iter()?
                .map(|pair| {
                    let pair = pair?;
                    let (s, t): (Bound<'py, PyAny>, Bound<'py, PyAny>) = pair.extract()?;
                    Ok((index_of(&s)?, index_of(&t)?))
                })
                .collect()
        })
        .transpose()?;
    let edges: Vec<(u32, u32)> = edges.into_iter().map(|(a, b, _)| (a, b)).collect();
    let query = MagnitudeQuery { l_max, k_max, pairs };
    if integral {
        let results: Vec<PairGroups> = phlite_bindings::detach(py, || {
            let graph = MagnitudeGraph::new(n_vertices as usize, &edges);
            magnitude_integral(&graph, &query, true)
        });
        let k_cap = k_max.unwrap_or(l_max).min(l_max);
        return integral_result(py, &results, l_max, k_cap, by_pair, &labels);
    }
    let results: Vec<PairRanks> = phlite_bindings::detach(py, || {
        let graph = MagnitudeGraph::new(n_vertices as usize, &edges);
        magnitude_ranks(&graph, &query, &moduli, true)
    });
    let label = |v: u32| -> Bound<'py, PyAny> {
        match &labels {
            None => v.into_pyobject(py).expect("an int").into_any(),
            Some(labels) => labels[v as usize].bind(py).clone(),
        }
    };
    let one = |m: usize| -> PyResult<Bound<'py, PyAny>> {
        if !by_pair {
            return Ok(total_ranks(&results, m, l_max, k_max).into_pyobject(py)?.into_any());
        }
        let out = PyDict::new(py);
        for pair in &results {
            let key = PyTuple::new(py, [label(pair.source), label(pair.target)])?;
            out.set_item(key, pair.ranks[m].clone())?;
        }
        Ok(out.into_any())
    };
    if !several {
        return one(0);
    }
    let out = PyDict::new(py);
    for (m, &modulus) in moduli.iter().enumerate() {
        out.set_item(modulus, one(m)?)?;
    }
    Ok(out.into_any())
}

/// The magnitude-path spectral sequence of a digraph (Asao; Hepworth and Roff) over `Z/modulus` or the integers, in
/// the bigrading of magnitude homology, computed one page at a time: a `phlite.core.SpectralSequence`, whose steps are
/// the lengths `l` and whose degrees are `k` (`ss.next()` gives `E^1`, then `E^2`, ...; printing a page shows it as a
/// table, as gramag's, with its differentials `d^r` from `(l, k)` to `(l - r, k - 1)`).
///
/// The reachability chain complex (tuples of nodes with a directed path from each to the next, with the alternating
/// sum of deletions of every node as differential) is filtered by the length of a tuple; this is its spectral
/// sequence. `E^1` is magnitude homology (as `magnitude` computes), `E^2` is *bigraded path homology*, whose diagonal
/// `E^2_{k,k}` is the path homology of the digraph, and later pages converge to reachability homology. Tuples are
/// listed up to length `l_max`, so an entry `E^r_{k,l}` is known exactly when `l + r <= l_max + 1`, and is `None`
/// (shown as `?`) otherwise. The complex doesn't split over pairs of nodes, so every tuple up to `l_max` is listed:
/// this is for small graphs.
///
/// Over a field the pages are read off the persistence barcode of the length filtration, computed here. Over the
/// integers (`integral=True`) there is no barcode, and each entry is computed exactly when its page is asked for, as
/// a quotient of lattices, `E^r_l = Z^r_l / (Z^{r-1}_{l-1} + d Z^{r-1}_{l+r-1})`, from the tuples of lengths
/// `l - r + 1, ..., l + r - 1` alone, by dense integer elimination: far slower, for a handful of nodes.
///
/// Call either as `magnitude_path_spectral_sequence(G, l_max=...)` with a networkx `DiGraph`, or as
/// `magnitude_path_spectral_sequence(n_vertices, edges, l_max=...)`.
///
/// Args:
///     graph: Either a networkx `DiGraph`, or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the edges, as `(source, target)` tuples.
///     l_max: The largest length to list tuples up to.
///     k_max: The largest degree (default: `l_max`).
///     modulus: The characteristic of the coefficient field: 2 (the default), 3, 5, 7, 11, 13 or 251.
///     integral: Compute over the integers instead (`modulus` is then ignored).
///
/// Returns:
///     A `phlite.core.SpectralSequence`: `ss.page(r)[l, k]` is the dimension of `E^r_{k,l}` (degree `k`, length `l`),
///     or over the integers `(rank, torsion)`, or `None` if not determined by tuples up to `l_max`.
///
/// Raises:
///     OverflowError: If, over the integers, an entry outgrows 128 bits (when its page is computed).
#[gen_stub_pyfunction(
    python_overload = r#"
    import builtins
    import typing
    import networkx
    from phlite import core

    @typing.overload
    def magnitude_path_spectral_sequence(
        graph: networkx.DiGraph,
        /,
        l_max: builtins.int,
        *,
        k_max: typing.Optional[builtins.int] = None,
        modulus: builtins.int = 2,
        integral: builtins.bool = False,
    ) -> core.SpectralSequence: ...
    @typing.overload
    def magnitude_path_spectral_sequence(
        n_vertices: builtins.int,
        edges: typing.Sequence[tuple[builtins.int, builtins.int] | tuple[builtins.int, builtins.int, builtins.float]],
        /,
        l_max: builtins.int,
        *,
        k_max: typing.Optional[builtins.int] = None,
        modulus: builtins.int = 2,
        integral: builtins.bool = False,
    ) -> core.SpectralSequence: ...
    "#,
    no_default_overload = true,
    module = "phlite.networks"
)]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, l_max = None, *, k_max = None, modulus = 2, integral = false))]
fn magnitude_path_spectral_sequence<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    edges: Option<&Bound<'py, PyAny>>,
    l_max: Option<usize>,
    k_max: Option<usize>,
    modulus: u64,
    integral: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use phlite_bindings::spectral::{spectral_sequence, SpectralBackend};
    let l_max = l_max.ok_or_else(|| PyTypeError::new_err("magnitude_path_spectral_sequence needs l_max"))?;
    if !integral && !MODULI.contains(&modulus) {
        return Err(PyValueError::new_err(format!("Unsupported modulus {modulus}, expected one of {MODULI:?}")));
    }
    let edges = read_edges(edges)?;
    let (n_vertices, edges, labels) =
        read_graph(graph, edges, "weight", "magnitude_path_spectral_sequence", u32::MAX >> 1)?;
    let edges: Vec<(u32, u32)> = edges.into_iter().map(|(a, b, _)| (a, b)).collect();
    let k_cap = k_max.unwrap_or(l_max).min(l_max);
    let (backend, tuples) = phlite_bindings::detach(py, || {
        let graph = MagnitudeGraph::new(n_vertices as usize, &edges);
        mpss_backend(&graph, l_max, k_max, (!integral).then_some(modulus))
    });
    let label_ids = match &labels {
        None => None,
        Some(labels) => {
            let ids = PyDict::new(py);
            for (i, label) in labels.iter().enumerate() {
                ids.set_item(label.bind(py), i)?;
            }
            Some(ids.unbind())
        }
    };
    let index = tuples.iter().cloned().enumerate().map(|(i, t)| (t, i)).collect();
    let cells = Box::new(TupleCells { tuples, index, labels, label_ids });
    let backend = SpectralBackend::new(backend, integral)
        .with_row_lengths((0..=l_max).map(|l| k_cap.min(l) + 1).collect())
        .with_cells(cells, (!integral).then_some(modulus));
    let kwargs = PyDict::new(py);
    kwargs.set_item("step_label", "l")?;
    kwargs.set_item("degree_label", "k")?;
    kwargs.set_item("title", "the MPSS")?;
    spectral_sequence(py, backend, &kwargs)
}

/// The Python result of [`magnitude_integral`]: `{"free": ranks[l][k], "torsion": {(k, l): {order: multiplicity}}}`
/// summed over the pairs, or a dict from each pair to its own.
fn integral_result<'py>(
    py: Python<'py>,
    results: &[PairGroups],
    l_max: usize,
    k_cap: usize,
    by_pair: bool,
    labels: &Option<Vec<Py<PyAny>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let summary = |pairs: &[&PairGroups]| -> PyResult<Bound<'py, PyAny>> {
        let mut free: Vec<Vec<Option<usize>>> = (0..=l_max).map(|l| vec![Some(0); k_cap.min(l) + 1]).collect();
        let mut counts: std::collections::BTreeMap<(usize, usize), std::collections::BTreeMap<u128, usize>> =
            Default::default();
        for pair in pairs {
            for (l, row) in pair.groups.iter().enumerate() {
                for (k, group) in row.iter().enumerate() {
                    match group {
                        None => free[l][k] = None,
                        Some((f, t)) => {
                            if let Some(total) = &mut free[l][k] {
                                *total += f;
                            }
                            for &order in t {
                                *counts.entry((k, l)).or_default().entry(order).or_default() += 1;
                            }
                        }
                    }
                }
            }
        }
        let torsion = PyDict::new(py);
        for (key, orders) in counts {
            let orders_dict = PyDict::new(py);
            for (order, count) in orders {
                orders_dict.set_item(order, count)?;
            }
            torsion.set_item(key, orders_dict)?;
        }
        let out = PyDict::new(py);
        out.set_item("free", free)?;
        out.set_item("torsion", torsion)?;
        Ok(out.into_any())
    };
    if !by_pair {
        return summary(&results.iter().collect::<Vec<_>>());
    }
    let label = |v: u32| -> Bound<'py, PyAny> {
        match labels {
            None => v.into_pyobject(py).expect("an int").into_any(),
            Some(labels) => labels[v as usize].bind(py).clone(),
        }
    };
    let out = PyDict::new(py);
    for pair in results {
        out.set_item(PyTuple::new(py, [label(pair.source), label(pair.target)])?, summary(&[pair])?)?;
    }
    Ok(out.into_any())
}

/// Edges given as pairs or weighted triples (weights are ignored).
fn read_edges(edges: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<(u32, u32, f64)>>> {
    edges
        .map(|edges| -> PyResult<Vec<(u32, u32, f64)>> {
            edges
                .try_iter()?
                .map(|edge| {
                    let edge = edge?;
                    if let Ok((a, b)) = edge.extract::<(u32, u32)>() {
                        Ok((a, b, 1.0))
                    } else {
                        edge.extract::<(u32, u32, f64)>().map_err(|_| {
                            PyTypeError::new_err("Edges must be (source, target) or (source, target, weight) tuples")
                        })
                    }
                })
                .collect()
        })
        .transpose()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(magnitude_path_spectral_sequence, m)?)
}

/// The cells of the reachability complex for generators' chains: tuples of nodes (by label, if the graph has them).
struct TupleCells {
    tuples: Vec<Vec<u32>>,
    index: std::collections::HashMap<Vec<u32>, usize>,
    labels: Option<Vec<Py<PyAny>>>,
    label_ids: Option<Py<PyDict>>,
}

impl phlite_bindings::spectral::CellNames for TupleCells {
    fn name<'py>(&self, py: Python<'py>, cell: usize) -> PyResult<Bound<'py, PyAny>> {
        let tuple = &self.tuples[cell];
        match &self.labels {
            None => Ok(PyTuple::new(py, tuple)?.into_any()),
            Some(labels) => Ok(PyTuple::new(py, tuple.iter().map(|&v| labels[v as usize].bind(py)))?.into_any()),
        }
    }

    fn index(&self, py: Python<'_>, cell: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let Ok(items) = cell.try_iter() else {
            return Ok(None);
        };
        let mut tuple = vec![];
        for item in items {
            let item = item?;
            let id = match &self.label_ids {
                None => item.extract::<u32>().ok(),
                Some(ids) => ids.bind(py).get_item(&item)?.map(|i| i.extract::<u32>()).transpose()?,
            };
            match id {
                Some(id) => tuple.push(id),
                None => return Ok(None),
            }
        }
        Ok(self.index.get(&tuple).copied())
    }
}
