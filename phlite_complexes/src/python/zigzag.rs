//! `phlite.complexes.zigzag`: zigzag persistence of simplicial complexes, by [`crate::zigzag`].

use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use phlite_bindings::{algorithm_from_py, AlgorithmVisitor};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use super::DEFAULTS;
use crate::zigzag::{zigzag_barcode, Operation, ZigzagBar, ZigzagError};

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
///         may be inserted again after being deleted.
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
#[pyo3(name = "_zigzag", signature = (operations, *, algorithm = None))]
#[allow(clippy::type_complexity)]
pub(crate) fn zigzag(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "typing.Iterable[tuple[typing.Sequence[builtins.int], builtins.bool]]", imports = ("builtins", "typing")))]
    operations: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<(usize, usize, usize, bool, bool)>> {
    let mut ops: Vec<Operation<Vec<i64>>> = vec![];
    for item in operations.try_iter()? {
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
        .detach(|| algorithm.run(Run { operations: &ops }))?
        .map_err(|e: ZigzagError| PyValueError::new_err(e.to_string()))?;
    let mut out: Vec<_> =
        bars.into_iter().map(|b: ZigzagBar| (b.dimension, b.birth, b.death, b.birth_closed, b.death_closed)).collect();
    out.sort_unstable();
    Ok(out)
}

struct Run<'a> {
    operations: &'a [Operation<Vec<i64>>],
}

impl AlgorithmVisitor for Run<'_> {
    type Output = Result<Vec<ZigzagBar>, ZigzagError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        zigzag_barcode(self.operations, algorithm)
    }
}
