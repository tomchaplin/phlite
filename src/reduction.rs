use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use crate::matrices::combinators::product;
use crate::matrices::implementors::MapVecMatrix;
use crate::matrices::{MatrixRef, SplitByDimension, SquareMatrix};
use crate::{
    columns::ColumnEntry,
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

// TODO: Experiment with how to represent cleared columns - by death idx or store all of R_j?
// TODO: Experiment with different Hashers, maybe nohash_hash? Make generic over ColT hashser?
impl<M> ClearedReductionMatrix<'static, M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    <M as HasColBasis>::BasisT: SplitByDimension,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
{
    pub fn build_with_diagram(
        boundary: M,
        dimension_order: impl Iterator<Item = usize>,
    ) -> (Self, Diagram<M::ColT>) {
        let mut essential = FxHashSet::default();
        let mut pairings = FxHashSet::default();

        let mut reduction_columns: FxHashMap<
            M::ColT,
            ReductionColumn<M::CoefficientField, M::ColT>,
        > = FxHashMap::default();

        for dim in dimension_order {
            let sub_matrix = boundary.sub_matrix_in_dimension(dim);
            // Reduce submatrix

            // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
            let mut low_inverse: FxHashMap<M::RowT, (M::ColT, M::CoefficientField)> =
                FxHashMap::default();

            'column_loop: for i in 0..sub_matrix.basis().size() {
                // Reduce column i
                let basis_element = sub_matrix.basis().element(i);

                // First check whether already cleared
                if reduction_columns.contains_key(&basis_element) {
                    continue 'column_loop;
                }

                let mut v_i = boundary.with_trivial_filtration().empty_bhcol();
                let mut r_i = sub_matrix.build_bhcol(basis_element).unwrap();

                'reduction: loop {
                    // TODO: Work out why this is so slow for last column :()
                    let Some(pivot_entry) = r_i.clone_pivot() else {
                        // Column reduced to 0 -> found cycle -> move onto next column
                        break 'reduction;
                        // TODO: In this case, we won't need this column of V again so no point storing!
                        // Unless we want representatives
                    };

                    // Check if there is a column with the same pivot
                    let Some((other_col_basis_element, other_col_coeff)) =
                        low_inverse.get(&pivot_entry.row_index)
                    else {
                        // Cannot reduce further -> found boundary -> break and save pivot
                        break 'reduction;
                    };

                    // If so then we add a multiple of that column to cancel out the pivot in r_col
                    let col_multiple =
                        pivot_entry.coeff.additive_inverse() * (other_col_coeff.inverse());

                    // Get references to V and R as reduced so far
                    let v_matrix =
                        ClearedReductionMatrix::build_from_ref(boundary, &reduction_columns);
                    let r_matrix = product(boundary, &v_matrix);

                    // TODO : Make this nicer

                    // Add the multiple of that column
                    r_i.add_entries(
                        r_matrix
                            .column_with_filtration(*other_col_basis_element)
                            .unwrap()
                            .map(|e| e.unwrap())
                            .map(|entry| {
                                ColumnEntry::from((
                                    entry.coeff * col_multiple,
                                    entry.row_index,
                                    entry.filtration_value,
                                ))
                            }),
                    );

                    let v_col = v_matrix.column(*other_col_basis_element).unwrap();
                    // Update V
                    v_i.add_entries(v_col.map(|(coeff, row_index)| {
                        ColumnEntry::from((coeff * col_multiple, row_index, ()))
                    }));
                }

                // If we have a pivot
                if let Some(pivot_entry) = r_i.peek_pivot().cloned() {
                    // NOTE: Safe to call peek_pivot because we only ever break after calling clone_pivot
                    // Save it to low inverse
                    low_inverse.insert(pivot_entry.row_index, (basis_element, pivot_entry.coeff));

                    // and clear out the birth column
                    reduction_columns.insert(
                        pivot_entry.row_index,
                        ReductionColumn::Cleared(basis_element),
                    );

                    // and update diagram
                    // Don't need to remove any essential because we assume the dimension_order
                    // is provided sensibly so that we see pairings first
                    pairings.insert((pivot_entry.row_index, basis_element));
                } else {
                    // update diagram
                    essential.insert(basis_element);
                }

                // TODO: Add option to only store this column when pivot is Some
                // Because otherwise we will never need the column again during reduction
                // Then save v_i to reduction matrix
                reduction_columns.insert(
                    basis_element,
                    ReductionColumn::Reduced(
                        v_i.drain_sorted()
                            .map(|entry| (entry.coeff, entry.row_index))
                            .collect(),
                    ),
                );
            }
            println!("Finished reducing dimension {dim}");
        }

        let diagram = Diagram {
            essential,
            pairings,
        };
        let v = ClearedReductionMatrix::build_from_owned(boundary, reduction_columns);
        (v, diagram)
    }
}

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
        v_i.add_entries(iter::once(ColumnEntry::from((
            M::CoefficientField::one(),
            basis_element,
            (),
        ))));
        let mut r_col = boundary.build_bhcol(basis_element).unwrap();

        'reduction: loop {
            let Some(pivot_entry) = r_col.pop_pivot() else {
                // Column reduced to 0 -> found cycle -> move onto next column
                break 'reduction;
            };

            let pivot_row_index = pivot_entry.row_index;
            let pivot_coeff = pivot_entry.coeff;

            // Push the pivot back in to keep r_col coorect
            r_col.push(pivot_entry);

            // Check if there is a column with the same pivot
            let Some((other_col_basis_element, other_col_coeff)) =
                low_inverse.get(&pivot_row_index)
            else {
                // Cannot reduce further -> found boundary -> break and save pivot
                break 'reduction;
            };

            // If so then we add a multiple of that column to cancel out the pivot in r_col
            let col_multiple = pivot_coeff.additive_inverse() * (other_col_coeff.inverse());

            let v_matrix = MapVecMatrix::from(&v);
            let r_matrix = product(&boundary, &v_matrix);

            // Add the multiple of that column
            r_col.add_entries(
                r_matrix
                    .column_with_filtration(*other_col_basis_element)
                    .unwrap()
                    .map(|e| e.unwrap())
                    .map(|entry| {
                        ColumnEntry::from((
                            entry.coeff * col_multiple,
                            entry.row_index,
                            entry.filtration_value,
                        ))
                    }),
            );

            // Update V
            v_i.add_entries(
                v_matrix
                    .column(*other_col_basis_element)
                    .unwrap()
                    .map(|(coeff, row_index)| ColumnEntry::from((coeff, row_index, ()))),
            )
        }

        // Save pivot if we have one
        if let Some(pivot_entry) = r_col.pop_pivot() {
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

    use ordered_float::NotNan;

    use crate::{
        fields::Z2,
        filtrations::rips::cohomology::RipsCoboundaryAllDims,
        matrices::{
            combinators::product, implementors::simple_Z2_matrix, HasRowFiltration, MatrixOracle,
            MatrixRef,
        },
        reduction::ClearedReductionMatrix,
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

    fn distance_matrix() -> Vec<Vec<NotNan<f64>>> {
        let zero = NotNan::new(0.0).unwrap();
        let one = NotNan::new(1.0).unwrap();
        let sqrt2 = NotNan::new(2.0_f64.sqrt()).unwrap();
        let distance_matrix: Vec<Vec<NotNan<f64>>> = vec![
            vec![zero, one, sqrt2, one],
            vec![one, zero, one, sqrt2],
            vec![sqrt2, one, zero, one],
            vec![one, sqrt2, one, zero],
        ];
        distance_matrix
    }

    fn big_distance_matrix() -> Vec<Vec<NotNan<f64>>> {
        let matrix = vec![
            vec![
                0.00, 1.94, 1.80, 1.29, 1.27, 1.27, 1.68, 2.00, 1.31, 1.73, 1.79, 1.91, 1.98, 0.98,
                1.14, 1.13, 0.44, 0.91, 0.36, 0.52,
            ],
            vec![
                1.94, 0.00, 1.27, 1.79, 1.20, 1.20, 0.66, 0.52, 1.78, 1.38, 0.44, 0.12, 0.73, 1.46,
                1.33, 1.34, 1.79, 1.95, 2.00, 1.75,
            ],
            vec![
                1.80, 1.27, 0.00, 0.81, 1.94, 1.94, 1.71, 0.82, 0.80, 0.15, 1.57, 1.36, 0.62, 2.00,
                1.98, 1.98, 1.95, 1.21, 1.62, 1.97,
            ],
            vec![
                1.29, 1.79, 0.81, 0.00, 1.97, 1.97, 1.98, 1.49, 0.02, 0.67, 1.94, 1.84, 1.34, 1.87,
                1.93, 1.93, 1.60, 0.46, 1.00, 1.65,
            ],
            vec![
                1.27, 1.20, 1.94, 1.97, 0.00, 0.00, 0.61, 1.58, 1.97, 1.97, 0.83, 1.10, 1.70, 0.35,
                0.16, 0.17, 0.90, 1.83, 1.52, 0.82,
            ],
            vec![
                1.27, 1.20, 1.94, 1.97, 0.00, 0.00, 0.61, 1.58, 1.97, 1.97, 0.83, 1.10, 1.70, 0.35,
                0.16, 0.17, 0.90, 1.83, 1.52, 0.82,
            ],
            vec![
                1.68, 0.66, 1.71, 1.98, 0.61, 0.61, 0.00, 1.13, 1.98, 1.78, 0.24, 0.55, 1.30, 0.93,
                0.76, 0.77, 1.40, 1.99, 1.85, 1.34,
            ],
            vec![
                2.00, 0.52, 0.82, 1.49, 1.58, 1.58, 1.13, 0.00, 1.48, 0.95, 0.93, 0.64, 0.21, 1.77,
                1.67, 1.68, 1.96, 1.76, 1.96, 1.94,
            ],
            vec![
                1.31, 1.78, 0.80, 0.02, 1.97, 1.97, 1.98, 1.48, 0.00, 0.66, 1.94, 1.83, 1.33, 1.88,
                1.94, 1.93, 1.61, 0.48, 1.01, 1.65,
            ],
            vec![
                1.73, 1.38, 0.15, 0.67, 1.97, 1.97, 1.78, 0.95, 0.66, 0.00, 1.66, 1.47, 0.76, 2.00,
                1.99, 1.99, 1.91, 1.09, 1.53, 1.93,
            ],
            vec![
                1.79, 0.44, 1.57, 1.94, 0.83, 0.83, 0.24, 0.93, 1.94, 1.66, 0.00, 0.32, 1.11, 1.13,
                0.97, 0.98, 1.56, 2.00, 1.92, 1.50,
            ],
            vec![
                1.91, 0.12, 1.36, 1.84, 1.10, 1.10, 0.55, 0.64, 1.83, 1.47, 0.32, 0.00, 0.84, 1.38,
                1.24, 1.24, 1.74, 1.97, 1.99, 1.69,
            ],
            vec![
                1.98, 0.73, 0.62, 1.34, 1.70, 1.70, 1.30, 0.21, 1.33, 0.76, 1.11, 0.84, 0.00, 1.86,
                1.78, 1.78, 1.99, 1.65, 1.90, 1.98,
            ],
            vec![
                0.98, 1.46, 2.00, 1.87, 0.35, 0.35, 0.93, 1.77, 1.88, 2.00, 1.13, 1.38, 1.86, 0.00,
                0.19, 0.18, 0.57, 1.66, 1.27, 0.49,
            ],
            vec![
                1.14, 1.33, 1.98, 1.93, 0.16, 0.16, 0.76, 1.67, 1.94, 1.99, 0.97, 1.24, 1.78, 0.19,
                0.00, 0.01, 0.75, 1.76, 1.41, 0.67,
            ],
            vec![
                1.13, 1.34, 1.98, 1.93, 0.17, 0.17, 0.77, 1.68, 1.93, 1.99, 0.98, 1.24, 1.78, 0.18,
                0.01, 0.00, 0.74, 1.76, 1.41, 0.66,
            ],
            vec![
                0.44, 1.79, 1.95, 1.60, 0.90, 0.90, 1.40, 1.96, 1.61, 1.91, 1.56, 1.74, 1.99, 0.57,
                0.75, 0.74, 0.00, 1.28, 0.78, 0.08,
            ],
            vec![
                0.91, 1.95, 1.21, 0.46, 1.83, 1.83, 1.99, 1.76, 0.48, 1.09, 2.00, 1.97, 1.65, 1.66,
                1.76, 1.76, 1.28, 0.00, 0.57, 1.34,
            ],
            vec![
                0.36, 2.00, 1.62, 1.00, 1.52, 1.52, 1.85, 1.96, 1.01, 1.53, 1.92, 1.99, 1.90, 1.27,
                1.41, 1.41, 0.78, 0.57, 0.00, 0.86,
            ],
            vec![
                0.52, 1.75, 1.97, 1.65, 0.82, 0.82, 1.34, 1.94, 1.65, 1.93, 1.50, 1.69, 1.98, 0.49,
                0.67, 0.66, 0.08, 1.34, 0.86, 0.00,
            ],
        ];

        matrix
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|entry| NotNan::new(entry).unwrap())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_clearing() {
        let distance_matrix = distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 1;

        // Compute column basis
        let coboundary = RipsCoboundaryAllDims::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix, in increasing dimension
        let (v, diagram) = ClearedReductionMatrix::build_with_diagram(&coboundary, 0..=max_dim);
        let _r = product(&coboundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = coboundary.filtration_value(*idx).unwrap().0;
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.1.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).unwrap().0;
            let death_f = coboundary.filtration_value(tup.0).unwrap().0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        // Ignored 2-dimensional void
        assert_eq!(diagram.pairings.len(), 6);
        assert_eq!(diagram.essential.len(), 1);
    }

    #[test]
    fn test_big_clearing() {
        let distance_matrix = big_distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 2;

        // Compute column basis
        let coboundary = RipsCoboundaryAllDims::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix, in increasing dimension
        let (v, diagram) = ClearedReductionMatrix::build_with_diagram(&coboundary, 0..=max_dim);
        let _r = product(&coboundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = coboundary.filtration_value(*idx).unwrap().0;
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.1.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).unwrap().0.into_inner();
            let death_f = coboundary.filtration_value(tup.0).unwrap().0.into_inner();
            let difference = death_f - birth_f;
            if difference > 0.01 {
                println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
            }
        }
    }
}
