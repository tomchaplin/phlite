//! An implementation of the standard algorithm with the clearing optimisation.
//!
//! To run this algorithm you should attempt to construct the V matrix by calling [`ClearedReductionMatrix::build_with_diagram`].

use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use crate::columns::{BHCol, ColumnEntry};
use crate::matrices::combinators::product;
use crate::matrices::{SplitByDimension, SquareMatrix};
use crate::matrix_col_product;
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};

use crate::reduction::Diagram;

use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone)]
#[allow(clippy::module_name_repetitions)]
enum ReductionColumn<CF, ColT> {
    /// This variant is used when `(i, j)` is found as a pair, in which case column `i` can be reduced by setting the column `V_i = R_j`, we store j here.
    Cleared(ColT),
    /// This variant is used when the column is not cleared and hence must be reduced by the main loop.
    /// The inner `Vec` represents the sum of columns used in the reduction, minus the starting column (which corresponds to the +1 on the diagonal of `V`).
    Reduced(Vec<(CF, ColT)>),
}

type ReductionCols<'a, ColT, CoefficientField> =
    Cow<'a, FxHashMap<ColT, ReductionColumn<CoefficientField, ColT>>>;

#[derive(Clone)]
/// The reduction matrix produced by the clearing algorithm.
///
/// Since some columns in this matrix are just copies of columns in the original boundary matrix (D) this must hold onto a reference to your matrix.
/// When you request a column of this matrix, if the column was reduced by clearing then the corresponding column from D is returned, otherwise the column was reduced conventially so the stored column of V is returned.
pub struct ClearedReductionMatrix<'a, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    boundary: M,
    reduction_columns: ReductionCols<'a, M::ColT, M::CoefficientField>,
}

impl<'a, M> ClearedReductionMatrix<'a, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    fn build_from_ref(
        boundary: M,
        reduction_columns: &'a FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
    ) -> Self {
        Self {
            boundary,
            reduction_columns: Cow::Borrowed(reduction_columns),
        }
    }

    /// Provides a more efficient check for whether a given column represents a cycle in the reduced matrix R.
    ///
    /// A column that was cleared by the clearing optimisation is necessarily a cycle.
    /// Otherwise, if the column was conventionally reduced then we have to build that column in R.
    pub fn col_is_cycle(&self, col: M::ColT) -> bool {
        let v_col = self.reduction_columns.get(&col).unwrap();
        match v_col {
            ReductionColumn::Cleared(_) => true,
            ReductionColumn::Reduced(_) => {
                let r_matrix = product(&self.boundary, &self);
                let mut r_col = r_matrix.build_bhcol(col);
                r_col.pop_pivot().is_none()
            }
        }
    }
}

impl<M> ClearedReductionMatrix<'static, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    fn build_from_owned(
        boundary: M,
        reduction_columns: FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
    ) -> Self {
        Self {
            boundary,
            reduction_columns: Cow::Owned(reduction_columns),
        }
    }
}

impl<'a, M> MatrixOracle for ClearedReductionMatrix<'a, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::ColT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // TODO: Check that this doesn't actually Clone!
        let reduction_col = self.reduction_columns.get(&col).unwrap();

        // TODO: Is there a way to do this without Box?

        let output_iter: Box<dyn Iterator<Item = (M::CoefficientField, M::RowT)>> =
            match reduction_col {
                ReductionColumn::Cleared(death_idx) => {
                    // This returns the death_idx column of R = D V
                    let v_j = self.column(death_idx.clone());
                    // v_j should be of the Reduced variant
                    Box::new(matrix_col_product!(self.boundary, v_j))
                    //Box::new(vec.iter().copied())
                }
                ReductionColumn::Reduced(vec) => Box::new(
                    // We don't store the diagonal so we have to chain +1 on the diagonal to the output
                    vec.iter()
                        .cloned()
                        .chain(iter::once((M::CoefficientField::one(), col))),
                ),
            };

        output_iter
    }
}

impl<M> HasColBasis for ClearedReductionMatrix<'_, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type BasisT = M::BasisT;
    type BasisRef<'b>
        = M::BasisRef<'b>
    where
        Self: 'b;

    fn basis(&self) -> Self::BasisRef<'_> {
        self.boundary.basis()
    }
}

struct CRMBuilder<M, DimIter, RowT, CF> {
    boundary: M,
    dimension_order: DimIter,
    diagram: Diagram<RowT>,
    reduction_columns: FxHashMap<RowT, ReductionColumn<CF, RowT>>,
}

impl<M, DimIter> CRMBuilder<M, DimIter, M::RowT, M::CoefficientField>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    <M as HasColBasis>::BasisT: SplitByDimension,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
    DimIter: Iterator<Item = usize>,
    ColumnEntry<M::FiltrationT, M::RowT, M::CoefficientField>: Clone,
{
    fn init(boundary: M, dimension_order: DimIter) -> Self {
        let essential = FxHashSet::default();
        let pairings = FxHashSet::default();
        let diagram = Diagram {
            essential,
            pairings,
        };

        let reduction_columns: FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>> =
            FxHashMap::default();
        Self {
            boundary,
            dimension_order,
            diagram,
            reduction_columns,
        }
    }

    fn reduce_column(
        &self,
        low_inverse: &FxHashMap<M::RowT, (M::ColT, M::CoefficientField)>,
        r_i: &mut BHCol<M::FiltrationT, M::RowT, M::CoefficientField>,
        v_i: &mut BHCol<(), M::RowT, M::CoefficientField>,
    ) {
        loop {
            let Some(pivot_entry) = r_i.clone_pivot() else {
                // Column reduced to 0 -> found cycle -> move onto next column
                return;
            };

            // Check if there is a column with the same pivot
            let Some((j_basis_element, j_coeff)) = low_inverse.get(&pivot_entry.row_index) else {
                // Cannot reduce further -> found boundary -> break and save pivot
                return;
            };

            // TODO: Over "fields" such as f64 this might not cancel to 0
            //       We need some way to enforce that the pivot is deleted
            // If so then we add a multiple of that column to cancel out the pivot in r_i
            let col_multiple = pivot_entry.coeff.additive_inverse() * (j_coeff.mult_inverse());

            // Get references to V and R as reduced so far
            let v_matrix =
                ClearedReductionMatrix::build_from_ref(&self.boundary, &self.reduction_columns);
            let v_matrix = v_matrix.with_trivial_filtration();
            let r_matrix = product(&self.boundary, &v_matrix);

            // Add the multiple of that column to r_i and v_i
            r_i.add_entries(
                r_matrix
                    .column_with_filtration(j_basis_element.clone())
                    .map(|entry| (entry * col_multiple)),
            );

            v_i.add_entries(
                v_matrix
                    .column_with_filtration(j_basis_element.clone())
                    .map(|entry| entry * col_multiple),
            );
        }
    }

    fn save_column(
        &mut self,
        basis_element: M::ColT,
        low_inverse: &mut FxHashMap<M::RowT, (M::ColT, M::CoefficientField)>,
        r_i: &BHCol<M::FiltrationT, M::RowT, M::CoefficientField>,
        mut v_i: BHCol<(), M::RowT, M::CoefficientField>,
    ) {
        // If we have a pivot
        if let Some(pivot_entry) = r_i.peek_pivot().cloned() {
            // NOTE: Safe to call peek_pivot because we only ever break after calling clone_pivot
            // Save it to low inverse
            low_inverse.insert(
                pivot_entry.row_index.clone(),
                (basis_element.clone(), pivot_entry.coeff),
            );

            // and clear out the birth column
            self.reduction_columns.insert(
                pivot_entry.row_index.clone(),
                ReductionColumn::Cleared(basis_element.clone()),
            );

            // Update diagram
            // Don't need to remove any essential because we assume the dimension_order
            // is provided sensibly so that we see pairings first
            self.diagram
                .pairings
                .insert((pivot_entry.row_index, basis_element.clone()));
        } else {
            // Update diagram
            self.diagram.essential.insert(basis_element.clone());
        }

        // Then save v_i to reduction matrix
        // TODO: Add option to only store this column when pivot is Some
        // Because otherwise we will never need the column again during reduction
        self.reduction_columns.insert(
            basis_element,
            ReductionColumn::Reduced(
                v_i.drain_sorted()
                    .map(|entry| (entry.coeff, entry.row_index))
                    .collect(),
            ),
        );
    }

    fn reduce_dimension(&mut self, dim: usize) {
        // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
        // We build a new one for each dimension
        let mut low_inverse: FxHashMap<M::RowT, (M::ColT, M::CoefficientField)> =
            FxHashMap::default();

        let basis_size = self.boundary.basis().in_dimension(dim).size();

        'column_loop: for i in 0..basis_size {
            // Reduce column i
            let basis_element = (&self.boundary)
                .sub_matrix_in_dimension(dim)
                .basis()
                .element(i);

            // First check whether already cleared
            if self.reduction_columns.contains_key(&basis_element) {
                continue 'column_loop;
            }

            // Otheewise clear and update the builder accordingly
            let mut v_i = {
                let self_borrow = &self.boundary;
                self_borrow.with_trivial_filtration().empty_bhcol()
            };
            let mut r_i = {
                let self_borrow = &self.boundary;
                self_borrow.build_bhcol(basis_element.clone())
            };
            self.reduce_column(&low_inverse, &mut r_i, &mut v_i);
            self.save_column(basis_element, &mut low_inverse, &r_i, v_i);
        }
    }

    fn reduce_all_dimension(&mut self) {
        // TODO: Surely there's a nicer way to write this iteration?
        while let Some(dim) = self.dimension_order.next() {
            self.reduce_dimension(dim);
        }
    }
}

// TODO: Experiment with different Hashers, maybe nohash_hash? Make generic over ColT hashser?
// TODO: Experiment with occasionally consolidating v_i and r_i
//       This will take some time but reduce memory usage - maybe make configurable?
impl<M> ClearedReductionMatrix<'static, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    <M as HasColBasis>::BasisT: SplitByDimension,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
    ColumnEntry<M::FiltrationT, M::RowT, M::CoefficientField>: Clone,
{
    /// Run the clearing algorithm and return the reduction matrix.
    /// Note:
    /// * The persistence diagram is also returned alongside since it is more efficient to compute during reduction.
    /// * The `boundary` matrix is consumed so you probably want to a pass a reference to your matrix.
    ///
    /// You must also specify a `dimension_order` - this should be a contiguous range of dimensions that you would like to reduce.
    /// The order is important and should follow the dimension order of your chain complex.
    /// That is, if you have a coboundary matrix then `dimension_order` should be `0..=max_dim` whereas if you have a boundary matrix then `dimension_order` should be `(0..=max_dim).rev()`.
    /// This enables the clearing optimisation to work and is cruical to returning the correct diagram.
    pub fn build_with_diagram<DimIter>(
        boundary: M,
        dimension_order: DimIter,
    ) -> (Self, Diagram<M::ColT>)
    where
        DimIter: IntoIterator<Item = usize>,
    {
        let mut builder = CRMBuilder::init(boundary, dimension_order.into_iter());
        builder.reduce_all_dimension();
        let v =
            ClearedReductionMatrix::build_from_owned(builder.boundary, builder.reduction_columns);
        let diagram = builder.diagram;
        (v, diagram)
    }
}
