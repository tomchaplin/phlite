//! An implementation of the standard algorithm with no optimisations.
//!
//! Run it with `Standard::new().reduce(matrix)`, see [`ReductionAlgorithm`].

use std::hash::Hash;

use crate::columns::{
    ColumnCoefficient, ColumnFiltration, ColumnRepresentation, ColumnRow, MergedColumns,
    WorkingColumn,
};
use crate::matrices::combinators::product;
use crate::matrices::implementors::MapVecMatrix;
use crate::matrices::SquareMatrix;
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::reduction::{Diagram, Reduction, ReductionAlgorithm};

// TODO: Convert to a MapVecMatrix so that we can use a common diagram read off

/// The columns of `V`, keyed by column
type VColumns<CF, ColT> = FxHashMap<ColT, Vec<(CF, ColT)>>;

/// The type of `V` returned by [`Standard`] - reduction columns are stored behind a hash map.
pub type StandardReductionMatrix<CF, ColT> = MapVecMatrix<'static, CF, ColT, ColT>;

/// The standard reduction algorithm, with no optimisations.
///
/// Columns are reduced according to their order in the attached [`ColBasis`].
/// Every column is built in full and its reduction column stored, and apparent pairs are not recognised,
/// so the column basis must contain every cell (unlike for [`Clearing`](crate::reduction::Clearing)).
/// The reduction matrix V is returned as a [`StandardReductionMatrix`].
///
/// The algorithm is generic over how it represents the column being reduced, see [`ColumnRepresentation`];
/// the default is [`MergedColumns`]. Choose another with [`with_columns`](Standard::with_columns).
#[derive(Debug, Clone, Copy, Default)]
pub struct Standard<C = MergedColumns> {
    columns: C,
}

impl Standard {
    /// The standard algorithm. The only option is the column representation.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: ColumnRepresentation> Standard<C> {
    /// Represent the columns being reduced with `columns` instead, see [`ColumnRepresentation`].
    pub fn with_columns<C2: ColumnRepresentation>(self, columns: C2) -> Standard<C2> {
        Standard { columns }
    }
}

impl<M, C> ReductionAlgorithm<M> for Standard<C>
where
    M: MatrixOracle + HasRowFiltration + HasColBasis + SquareMatrix,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow + 'static,
    C: ColumnRepresentation,
{
    type V = StandardReductionMatrix<M::CoefficientField, M::ColT>;

    fn reduce(self, boundary: M) -> Reduction<Self::V, M::ColT> {
        let v = standard_algo(&boundary, &self.columns);
        let diagram = read_off_diagram(&boundary, &v);
        Reduction { v, diagram }
    }
}

fn standard_algo<M, C>(
    boundary: M,
    columns: &C,
) -> StandardReductionMatrix<M::CoefficientField, M::ColT>
where
    M: MatrixOracle + HasRowFiltration + HasColBasis + SquareMatrix,
    M::CoefficientField: Invertible + ColumnCoefficient,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow,
    C: ColumnRepresentation,
{
    // There is only one "dimension", so one cache for the whole reduction
    let cache = C::Cache::default();
    // Each V_i, including its diagonal
    let mut v: VColumns<M::CoefficientField, M::ColT> = FxHashMap::default();

    // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
    let mut low_inverse: FxHashMap<M::RowT, (M::ColT, M::CoefficientField)> = FxHashMap::default();

    for i in 0..boundary.basis().size() {
        // Reduce column i

        let basis_element = boundary.basis().element(i);

        let mut v_i = (&boundary).with_trivial_filtration().empty_bhcol();
        v_i.add_tuple((M::CoefficientField::one(), basis_element.clone(), ()));
        let mut r_i = C::Column::default();
        columns.add_matrix_column(
            &cache,
            &boundary,
            &mut r_i,
            basis_element.clone(),
            M::CoefficientField::one(),
        );

        let pivot = loop {
            let Some(pivot_entry) = r_i.pop_pivot() else {
                // Column reduced to 0 -> found cycle -> move onto next column
                break None;
            };

            // Check if there is a column with the same pivot
            let Some((j_basis_element, j_coeff)) = low_inverse.get(&pivot_entry.row_index) else {
                // Cannot reduce further -> found boundary -> save pivot
                break Some(pivot_entry);
            };

            // If so then we add a multiple of R_j = D V_j to cancel out the pivot, which we push back in to be cancelled
            let col_multiple = pivot_entry.coeff.additive_inverse() * (j_coeff.mult_inverse());
            r_i.add_entry(pivot_entry);

            // V_j includes its diagonal
            for (coeff, k) in &v[j_basis_element] {
                let multiple = *coeff * col_multiple;
                columns.add_matrix_column(&cache, &boundary, &mut r_i, k.clone(), multiple);
                v_i.add_tuple((multiple, k.clone(), ()));
            }
        };

        // Save pivot if we have one
        if let Some(pivot_entry) = pivot {
            low_inverse.insert(
                pivot_entry.row_index,
                (basis_element.clone(), pivot_entry.coeff),
            );
        };

        // Save V
        v.insert(
            basis_element,
            v_i.drain_sorted()
                .map(|entry| (entry.coeff, entry.row_index))
                .collect(),
        );
    }

    MapVecMatrix::from(v)
}

/// Reads off the pairings from the reduced matrix to construct the persistence diagram.
///
/// The result doesn't depend on the order in which columns are visited:
/// the essential classes are the columns that reduce to zero and are not the pivot of any other column.
fn read_off_diagram<M>(
    boundary: M,
    reduction_matrix: &StandardReductionMatrix<M::CoefficientField, M::ColT>,
) -> Diagram<M::ColT>
where
    M: SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let r = product(&boundary, &reduction_matrix);

    let mut essential = FxHashSet::default();
    let mut pairings = FxHashSet::default();
    let mut pivot_rows = FxHashSet::default();

    let basis = boundary.basis();
    for i in 0..basis.size() {
        let basis_element = basis.element(i);
        match r.build_bhcol(basis_element.clone()).pop_pivot() {
            None => {
                essential.insert(basis_element);
            }
            Some(piv) => {
                pivot_rows.insert(piv.row_index.clone());
                pairings.insert((piv.row_index, basis_element));
            }
        }
    }
    // A column that is the pivot of another is the birth of a finite bar
    essential.retain(|elem| !pivot_rows.contains(elem));

    Diagram {
        essential,
        pairings,
    }
}
