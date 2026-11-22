//! `phlite.complexes.zigzag`: zigzag persistence of simplicial complexes, by [`crate::zigzag`].

use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use phlite_bindings::{algorithm_from_py, AlgorithmVisitor};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use super::DEFAULTS;
use crate::zigzag::{timed_zigzag, zigzag_barcode_with, Join, Operation, TimedCell, TimedZigzagError, ZigzagBar, ZigzagError};

/// The engine behind `phlite.complexes.zigzag` and `Zigzag` (which return a `ZigzagBarcode`): the intervals as tuples.
///
/// Compute the zigzag persistence of a sequence of simplicial complexes, each obtained from the last by inserting or
/// deleting one simplex, starting from the empty complex, over Z/2.
///
/// A zigzag `K_0 <-> K_1 <-> ... <-> K_m` (with `K_0` empty and `K_{i+1}` the result of operation `i`) has a barcode
/// of intervals `[birth, death]` (homology alive in `K_birth, ..., K_death`), each end *closed* or *open*: a birth is
/// closed if made by an insertion (open if by a deletion), a death closed if followed by a deletion (open if by an
/// insertion). Computed by FastZigzag (Dey and Hou): the zigzag is turned into an ordinary filtration of a complex of
/// twice its length (the insertions, then cones on the deletions), whose barcode phlite computes as usual and maps
/// back interval by interval. Intervals still alive at the end end at `m` (closed).
///
/// Args:
///     operations: The operations, in order, as `(simplex, insert)` pairs: a simplex as a sequence of integer vertices
///         and `True` to insert it (its faces must be there), `False` to delete it (its cofaces must be gone). A simplex
///         may be inserted again after being deleted. Or, for cells of any kind (cubes, ...), `(cell, insert, faces)`
///         triples: any hashable cells, each inserted with its faces (cells inserted before; none for a vertex), the
///         dimension following from them, over Z/2.
///     algorithm: The reduction algorithm, as for `FilteredComplex.persistence`.
///
/// Returns:
///     The intervals, as `(dimension, birth, death, birth_closed, death_closed)` tuples (positions of complexes, from
///     0 to `m`), sorted.
///
/// Raises:
///     ValueError: If an operation is impossible (inserting a simplex that is there or without its faces, deleting one
///         that isn't there or with a coface).
#[gen_stub_pyfunction(module = "phlite.complexes")]
#[pyfunction]
#[pyo3(name = "_zigzag", signature = (operations, *, algorithm = None, method = "cohomology", union_find = true))]
#[allow(clippy::type_complexity)]
pub(crate) fn zigzag(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "typing.Iterable[typing.Union[tuple[typing.Sequence[builtins.int], builtins.bool], tuple[typing.Any, builtins.bool, typing.Sequence[typing.Any]]]]", imports = ("builtins", "typing")))]
    operations: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))] method: &str,
    union_find: bool,
) -> PyResult<Vec<(usize, usize, usize, bool, bool)>> {
    let method = phlite_bindings::parse_method(method)?;
    let items: Vec<Bound<'_, PyAny>> = operations.try_iter()?.collect::<PyResult<_>>()?;
    // Cells of any kind, with their faces: (cell, insert, faces) triples
    if items.iter().any(|item| item.len().is_ok_and(|n| n == 3)) {
        let ids = pyo3::types::PyDict::new(py);
        let mut dimension_of: Vec<usize> = vec![];
        let mut ops: Vec<Operation<u32>> = vec![];
        for item in &items {
            let cell = item.get_item(0)?;
            let insert: bool = item.get_item(1)?.extract()?;
            let id = match ids.get_item(&cell)? {
                Some(id) => id.extract::<u32>()?,
                None => {
                    let id = dimension_of.len() as u32;
                    ids.set_item(&cell, id)?;
                    dimension_of.push(0);
                    id
                }
            };
            if !insert {
                ops.push(Operation::Delete(id));
                continue;
            }
            let faces: Vec<u32> = if item.len()? == 3 {
                item.get_item(2)?
                    .try_iter()?
                    .map(|face| {
                        let face = face?;
                        ids.get_item(&face)?.ok_or_else(|| {
                            PyValueError::new_err(format!("a face of {cell} ({face}) was never inserted"))
                        })?
                        .extract::<u32>()
                    })
                    .collect::<PyResult<_>>()?
            } else {
                vec![]
            };
            let dims: std::collections::BTreeSet<usize> = faces.iter().map(|&f| dimension_of[f as usize]).collect();
            if dims.len() > 1 {
                return Err(PyValueError::new_err(format!("the faces of {cell} have different dimensions")));
            }
            let dimension = dims.first().map_or(0, |d| d + 1);
            dimension_of[id as usize] = dimension;
            ops.push(Operation::Insert { cell: id, dimension, faces });
        }
        let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
        let bars = py
            .detach(|| algorithm.run(Run { operations: &ops, method, union_find }))?
            .map_err(|e: ZigzagError| PyValueError::new_err(e.to_string()))?;
        let mut out: Vec<_> =
            bars.into_iter().map(|b: ZigzagBar| (b.dimension, b.birth, b.death, b.birth_closed, b.death_closed)).collect();
        out.sort_unstable();
        return Ok(out);
    }
    let mut ops: Vec<Operation<Vec<i64>>> = vec![];
    for item in items {
        let item = Ok::<_, PyErr>(item);
        let (simplex, insert): (Vec<i64>, bool) = item?.extract()?;
        let mut simplex = simplex;
        simplex.sort_unstable();
        if simplex.is_empty() || simplex.windows(2).any(|w| w[0] == w[1]) {
            return Err(PyValueError::new_err(format!("{simplex:?} is not a simplex")));
        }
        ops.push(if insert {
            let faces = if simplex.len() == 1 {
                vec![]
            } else {
                (0..simplex.len())
                    .map(|i| simplex.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v).collect())
                    .collect()
            };
            Operation::Insert { dimension: simplex.len() - 1, cell: simplex, faces }
        } else {
            Operation::Delete(simplex)
        });
    }
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py
        .detach(|| algorithm.run(Run { operations: &ops, method, union_find }))?
        .map_err(|e: ZigzagError| PyValueError::new_err(e.to_string()))?;
    let mut out: Vec<_> =
        bars.into_iter().map(|b: ZigzagBar| (b.dimension, b.birth, b.death, b.birth_closed, b.death_closed)).collect();
    out.sort_unstable();
    Ok(out)
}

/// The engine behind `phlite.complexes.timed_zigzag` (which returns a `TimedZigzagBarcode`).
///
/// Args:
///     cells: `(cell, lifetimes)` pairs, a cell a simplex (a sequence of integer vertices, its faces its facets), or
///         `(cell, lifetimes, faces)` triples for cells of any hashable kind; lifetimes a list of `(start, end)` pairs
///         (the cell is there when `start <= t < end`; `end` may be `inf`).
///     join: `'union'` or `'intersection'`: what the zigzag passes through at each event time.
///     algorithm, method, union_find: As for `zigzag`.
///
/// Returns:
///     `(bars, times)`: each bar `(dimension, birth, death, birth_closed, death_closed, birth_level, death_level)`,
///     and the event times.
#[gen_stub_pyfunction(module = "phlite.complexes")]
#[pyfunction]
#[pyo3(name = "_timed_zigzag", signature = (cells, *, join = "union", algorithm = None, method = "cohomology", union_find = true))]
#[allow(clippy::type_complexity)]
pub(crate) fn timed(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "typing.Iterable[typing.Any]", imports = ("typing")))] cells: &Bound<'_, PyAny>,
    join: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))] method: &str,
    union_find: bool,
) -> PyResult<(Vec<(usize, f64, f64, bool, bool, usize, usize)>, Vec<f64>)> {
    let method = phlite_bindings::parse_method(method)?;
    let join = match join {
        "union" => Join::Union,
        "intersection" => Join::Intersection,
        other => return Err(PyValueError::new_err(format!("Unknown join {other:?}: expected 'union' or 'intersection'"))),
    };
    let items: Vec<Bound<'_, PyAny>> = cells.try_iter()?.collect::<PyResult<_>>()?;
    let general = items.iter().any(|item| item.len().is_ok_and(|n| n == 3));
    let mut timed: Vec<TimedCell<u32>> = Vec::with_capacity(items.len());
    if general {
        let ids = pyo3::types::PyDict::new(py);
        for (i, item) in items.iter().enumerate() {
            if ids.contains(item.get_item(0)?)? {
                return Err(PyValueError::new_err(format!("{} is given twice", item.get_item(0)?)));
            }
            ids.set_item(item.get_item(0)?, i as u32)?;
        }
        let mut faces: Vec<Vec<u32>> = vec![];
        for item in &items {
            let cell = item.get_item(0)?;
            let f: Vec<u32> = if item.len()? == 3 {
                item.get_item(2)?
                    .try_iter()?
                    .map(|face| {
                        let face = face?;
                        ids.get_item(&face)?
                            .ok_or_else(|| PyValueError::new_err(format!("a face of {cell} ({face}) isn't a cell")))?
                            .extract::<u32>()
                    })
                    .collect::<PyResult<_>>()?
            } else {
                vec![]
            };
            faces.push(f);
        }
        // Dimensions from the faces (one more than theirs, which must agree)
        let mut dims: Vec<Option<usize>> = vec![None; items.len()];
        fn dim(c: usize, faces: &[Vec<u32>], dims: &mut [Option<usize>], depth: usize) -> PyResult<usize> {
            if let Some(d) = dims[c] {
                return Ok(d);
            }
            if depth > faces.len() {
                return Err(PyValueError::new_err("the faces go round in a circle"));
            }
            let mut d: Option<usize> = None;
            for &f in &faces[c] {
                let fd = dim(f as usize, faces, dims, depth + 1)?;
                if d.is_some_and(|d| d != fd) {
                    return Err(PyValueError::new_err(format!("the faces of cell {c} have different dimensions")));
                }
                d = Some(fd);
            }
            let d = d.map_or(0, |d| d + 1);
            dims[c] = Some(d);
            Ok(d)
        }
        for (i, item) in items.iter().enumerate() {
            let dimension = dim(i, &faces, &mut dims, 0)?;
            let lifetimes: Vec<(f64, f64)> = item.get_item(1)?.extract()?;
            timed.push(TimedCell { cell: i as u32, dimension, faces: std::mem::take(&mut faces[i]), lifetimes });
        }
    } else {
        let mut index: std::collections::HashMap<Vec<i64>, u32> = std::collections::HashMap::new();
        let mut simplices: Vec<(Vec<i64>, Vec<(f64, f64)>)> = vec![];
        for item in &items {
            let (mut simplex, lifetimes): (Vec<i64>, Vec<(f64, f64)>) = item.extract()?;
            simplex.sort_unstable();
            if simplex.is_empty() || simplex.windows(2).any(|w| w[0] == w[1]) {
                return Err(PyValueError::new_err(format!("{simplex:?} is not a simplex")));
            }
            if index.insert(simplex.clone(), simplices.len() as u32).is_some() {
                return Err(PyValueError::new_err(format!("{simplex:?} is given twice")));
            }
            simplices.push((simplex, lifetimes));
        }
        for (i, (simplex, lifetimes)) in simplices.into_iter().enumerate() {
            let faces = crate::zigzag::simplex_facets(&simplex)
                .into_iter()
                .map(|f| index.get(&f).copied().ok_or_else(|| PyValueError::new_err(format!("{f:?}, a face of {simplex:?}, isn't a cell"))))
                .collect::<PyResult<_>>()?;
            timed.push(TimedCell { cell: i as u32, dimension: simplex.len() - 1, faces, lifetimes });
        }
    }
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let (bars, times) = py
        .detach(|| algorithm.run(TimedRun { cells: &timed, join, method, union_find }))?
        .map_err(|e: TimedZigzagError| PyValueError::new_err(e.to_string()))?;
    let out = bars
        .into_iter()
        .map(|b| (b.dimension, b.birth, b.death, b.birth_closed, b.death_closed, b.levels.birth, b.levels.death))
        .collect();
    Ok((out, times))
}

struct TimedRun<'a> {
    cells: &'a [TimedCell<u32>],
    join: Join,
    method: phlite::barcode::Method,
    union_find: bool,
}

impl AlgorithmVisitor for TimedRun<'_> {
    type Output = Result<(Vec<crate::zigzag::TimedBar>, Vec<f64>), TimedZigzagError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        timed_zigzag(self.cells, self.join, self.method, algorithm, self.union_find)
    }
}

struct Run<'a, K> {
    operations: &'a [Operation<K>],
    method: phlite::barcode::Method,
    union_find: bool,
}

impl<K: std::hash::Hash + Eq + Clone + Send + Sync> AlgorithmVisitor for Run<'_, K> {
    type Output = Result<Vec<ZigzagBar>, ZigzagError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        zigzag_barcode_with(self.operations, self.method, algorithm, self.union_find)
    }
}
