use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use crate::columns::BHCol;
use crate::matrices::adaptors::WithTrivialFiltration;
use crate::matrices::combinators::product;
use crate::matrices::implementors::MapVecMatrix;
use crate::matrices::{MatrixRef, SplitByDimension, SquareMatrix};
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};
use crate::{matrix_col_product, PhliteError};

use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone)]
pub enum ReductionColumn<CF, ColT> {
    Cleared(ColT), // This gets set when (i, j) is found as a pair in which case column i can be reduced by R_j, we store j here
    Reduced(Vec<(CF, ColT)>), // The sum of columns required to reduce (minus the +1 with self index)
}

#[derive(Clone)]
pub struct ClearedReductionMatrix<'a, M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    boundary: M,
    reduction_columns: Cow<'a, FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>>,
}

impl<'a, M> ClearedReductionMatrix<'a, M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
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

    pub fn col_is_cycle(&self, col: M::ColT) -> Result<bool, PhliteError> {
        let v_col = self
            .reduction_columns
            .get(&col)
            .ok_or(PhliteError::NotInDomain)?;
        match v_col {
            ReductionColumn::Cleared(_) => Ok(true),
            ReductionColumn::Reduced(_) => {
                let r_matrix = product(&self.boundary, &self);
                let mut r_col = r_matrix.build_bhcol(col)?;
                Ok(r_col.pop_pivot().is_none())
            }
        }
    }
}

impl<M> ClearedReductionMatrix<'static, M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
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
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::ColT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        // TODO: Check that this doesn't actually Clone!
        let reduction_col = self
            .reduction_columns
            .get(&col)
            .ok_or(PhliteError::NotInDomain)?;

        // TODO: Is there a way to do this without Box?

        let output_iter: Box<dyn Iterator<Item = (M::CoefficientField, M::RowT)>> =
            match reduction_col {
                ReductionColumn::Cleared(death_idx) => {
                    // This returns the death_idx column of R = D V
                    let v_j = self.column(*death_idx)?;
                    // v_j should be of the Reduced variant
                    Box::new(matrix_col_product!(self.boundary, v_j))
                    //Box::new(vec.iter().copied())
                }
                ReductionColumn::Reduced(vec) => Box::new(
                    // We don't store the diagonal so we have to chain +1 on the diagonal to the output
                    vec.iter()
                        .copied()
                        .chain(iter::once((M::CoefficientField::one(), col))),
                ),
            };

        Ok(output_iter)
    }
}

impl<'a, M> HasColBasis for ClearedReductionMatrix<'a, M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type BasisT = M::BasisT;

    fn basis(&self) -> &Self::BasisT {
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
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    <M as HasColBasis>::BasisT: SplitByDimension,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
    DimIter: Iterator<Item = usize>,
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
        r_i: &mut BHCol<M>,
        v_i: &mut BHCol<WithTrivialFiltration<M>>,
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

            // If so then we add a multiple of that column to cancel out the pivot in r_i
            let col_multiple = pivot_entry.coeff.additive_inverse() * (j_coeff.inverse());

            // Get references to V and R as reduced so far
            let v_matrix =
                ClearedReductionMatrix::build_from_ref(self.boundary, &self.reduction_columns);
            let v_matrix = v_matrix.with_trivial_filtration();
            let r_matrix = product(self.boundary, &v_matrix);

            // Add the multiple of that column to r_i and v_i
            r_i.add_entries(
                r_matrix
                    .column_with_filtration(*j_basis_element)
                    .unwrap()
                    .map(|entry| (entry * col_multiple)),
            );

            v_i.add_entries(
                v_matrix
                    .column_with_filtration(*j_basis_element)
                    .unwrap()
                    .map(|entry| entry * col_multiple),
            );
        }
    }

    fn save_column(
        &mut self,
        basis_element: M::ColT,
        low_inverse: &mut FxHashMap<M::RowT, (M::ColT, M::CoefficientField)>,
        r_i: BHCol<M>,
        mut v_i: BHCol<WithTrivialFiltration<M>>,
    ) {
        // If we have a pivot
        if let Some(pivot_entry) = r_i.peek_pivot().cloned() {
            // NOTE: Safe to call peek_pivot because we only ever break after calling clone_pivot
            // Save it to low inverse
            low_inverse.insert(pivot_entry.row_index, (basis_element, pivot_entry.coeff));

            // and clear out the birth column
            self.reduction_columns.insert(
                pivot_entry.row_index,
                ReductionColumn::Cleared(basis_element),
            );

            // Update diagram
            // Don't need to remove any essential because we assume the dimension_order
            // is provided sensibly so that we see pairings first
            self.diagram
                .pairings
                .insert((pivot_entry.row_index, basis_element));
        } else {
            // Update diagram
            self.diagram.essential.insert(basis_element);
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

        let basis_size = self.boundary.sub_matrix_in_dimension(dim).basis().size();

        'column_loop: for i in 0..basis_size {
            // Reduce column i
            let basis_element = self
                .boundary
                .sub_matrix_in_dimension(dim)
                .basis()
                .element(i);

            // First check whether already cleared
            if self.reduction_columns.contains_key(&basis_element) {
                continue 'column_loop;
            }

            // Otheewise clear and update the builder accordingly
            let mut v_i = self.boundary.with_trivial_filtration().empty_bhcol();
            let mut r_i = self.boundary.build_bhcol(basis_element).unwrap();
            self.reduce_column(&low_inverse, &mut r_i, &mut v_i);
            self.save_column(basis_element, &mut low_inverse, r_i, v_i);
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
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    <M as HasColBasis>::BasisT: SplitByDimension,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
{
    pub fn build_with_diagram<DimIter>(
        boundary: M,
        dimension_order: DimIter,
    ) -> (Self, Diagram<M::ColT>)
    where
        DimIter: Iterator<Item = usize>,
    {
        let mut builder = CRMBuilder::init(boundary, dimension_order);
        builder.reduce_all_dimension();
        let v =
            ClearedReductionMatrix::build_from_owned(builder.boundary, builder.reduction_columns);
        let diagram = builder.diagram;
        (v, diagram)
    }
}

// TODO: Should this be just a vec to make construction quicker?
#[derive(Debug, Clone)]
pub struct Diagram<T> {
    pub essential: FxHashSet<T>,
    pub pairings: FxHashSet<(T, T)>,
}

// TODO: Convert to a MapVecMatrix so that we can use a common diagram read off

pub type StandardReductionMatrix<CF, ColT> = MapVecMatrix<'static, CF, ColT, ColT>;

/// If your operator goes up in column order (e.g. coboundary) then you will need to set `reverse_order=True`.
pub fn standard_algo_with_diagram<M>(
    boundary: M,
    reverse_order: bool,
) -> (
    StandardReductionMatrix<M::CoefficientField, M::ColT>,
    Diagram<M::ColT>,
)
where
    M: SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let v = standard_algo(&boundary);
    let diagram = read_off_diagram(&boundary, &v, reverse_order);
    (v, diagram)
}

/// If your operator goes up in column order (e.g. coboundary) then you will need to set `reverse_order=True`.
pub fn read_off_diagram<M>(
    boundary: M,
    reduction_matrix: &StandardReductionMatrix<M::CoefficientField, M::ColT>,
    reverse_order: bool,
) -> Diagram<M::ColT>
where
    M: SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let r = product(&boundary, &reduction_matrix);

    let mut essential = FxHashSet::default();
    let mut pairings = FxHashSet::default();

    let col_iter: Box<dyn Iterator<Item = usize>> = if reverse_order {
        Box::new((0..boundary.basis().size()).rev())
    } else {
        Box::new(0..boundary.basis().size())
    };

    for i in col_iter {
        let basis_element = boundary.basis().element(i);
        let mut r_i = r.build_bhcol(basis_element).unwrap();
        match r_i.pop_pivot() {
            None => {
                essential.insert(basis_element);
            }
            Some(piv) => {
                pairings.insert((piv.row_index, basis_element));
                essential.remove(&piv.row_index);
            }
        }
        if r_i.pop_pivot().is_none() {}
    }

    let diagram = Diagram {
        essential,
        pairings,
    };

    diagram
}

pub fn standard_algo<M>(boundary: M) -> StandardReductionMatrix<M::CoefficientField, M::ColT>
where
    M: MatrixOracle + HasRowFiltration + HasColBasis + SquareMatrix,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let mut v = FxHashMap::default();

    // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
    let mut low_inverse: FxHashMap<M::RowT, (M::ColT, M::CoefficientField)> = FxHashMap::default();

    for i in 0..boundary.basis().size() {
        // Reduce column i

        let basis_element = boundary.basis().element(i);

        let mut v_i = boundary.with_trivial_filtration().empty_bhcol();
        v_i.add_tuple((M::CoefficientField::one(), basis_element, ()));
        let mut r_i = boundary.build_bhcol(basis_element).unwrap();

        'reduction: loop {
            let Some(pivot_entry) = r_i.pop_pivot() else {
                // Column reduced to 0 -> found cycle -> move onto next column
                break 'reduction;
            };

            let pivot_j = pivot_entry.row_index;
            let pivot_coeff = pivot_entry.coeff;

            // Push the pivot back in to keep r_col coorect
            r_i.push(pivot_entry);

            // Check if there is a column with the same pivot
            let Some((j_basis_element, j_coeff)) = low_inverse.get(&pivot_j) else {
                // Cannot reduce further -> found boundary -> break and save pivot
                break 'reduction;
            };

            // If so then we add a multiple of that column to cancel out the pivot in r_col
            let col_multiple = pivot_coeff.additive_inverse() * (j_coeff.inverse());

            let v_matrix = MapVecMatrix::from(&v);
            let v_matrix = v_matrix.with_trivial_filtration();
            let r_matrix = product(&boundary, &v_matrix);

            // Add the multiple of that column
            r_i.add_entries(
                r_matrix
                    .column_with_filtration(*j_basis_element)
                    .unwrap()
                    .map(|entry| entry * col_multiple),
            );

            // Update V
            v_i.add_entries(
                v_matrix
                    .column_with_filtration(*j_basis_element)
                    .unwrap()
                    .map(|entry| entry * col_multiple),
            )
        }

        // Save pivot if we have one
        if let Some(pivot_entry) = r_i.pop_pivot() {
            low_inverse.insert(pivot_entry.row_index, (basis_element, pivot_entry.coeff));
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

#[cfg(test)]
mod tests {

    use crate::matrices::{
        combinators::product, implementors::simple_Z2_matrix, MatrixOracle, MatrixRef,
    };

    use super::standard_algo;

    #[test]
    fn test_inefficient_reduction() {
        let matrix_d = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
            vec![3, 4, 5],
        ]);
        let matrix_v = standard_algo(matrix_d.with_trivial_filtration());
        let matrix_r = product(&matrix_d, &matrix_v);
        let true_matrix_r = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![],
            vec![3, 4, 5],
            vec![],
        ]);

        assert!((0..=7).all(|idx| matrix_r.eq_on_col(&true_matrix_r, idx)))
    }
}
