// ======== Tests ==============================================

use std::cmp::Reverse;

use crate::fields::{NonZeroCoefficient, Z2};
use crate::matrices::adaptors::consolidate;
use crate::matrices::combinators::product;
use crate::matrices::MatrixOracle;
use crate::matrices::{implementors::simple_Z2_matrix, HasRowFiltration, MatrixRef};

use crate::columns::BHCol;

use super::{ColBasis, HasColBasis};

#[test]
fn test_matrix_product() {
    let matrix_d = simple_Z2_matrix(vec![
        vec![],
        vec![],
        vec![],
        vec![0, 1],
        vec![1, 2],
        vec![0, 2],
        vec![3, 4, 5],
    ]);
    let matrix_v = simple_Z2_matrix(vec![
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![3, 4, 5],
        vec![6],
    ]);

    let true_matrix_r = simple_Z2_matrix(vec![
        vec![],
        vec![],
        vec![],
        vec![0, 1],
        vec![1, 2],
        vec![],
        vec![3, 4, 5],
    ]);

    let matrix_r = product(&matrix_d, &matrix_v);

    assert!((0..=6).all(|i| matrix_r.eq_on_col(&true_matrix_r, i)))
}

#[test]
fn test_matrix_bhcol_interface() {
    let base_matrix = simple_Z2_matrix(vec![
        vec![],
        vec![],
        vec![],
        vec![0, 1],
        vec![1, 2],
        vec![0, 2],
        vec![3, 4, 5],
    ]);
    let matrix = base_matrix.with_filtration(|idx| Ok(idx * 10));

    let add = |column: &mut BHCol<_>, index| {
        column.add_entries(matrix.column_with_filtration(index).unwrap());
    };

    let mut column = matrix.empty_bhcol();
    add(&mut column, 5);
    add(&mut column, 4);
    assert_eq!(
        column.pop_pivot().map(|e| e.into()),
        Some((Z2::one(), 1, 10))
    );
    assert_eq!(
        column.pop_pivot().map(|e| e.into()),
        Some((Z2::one(), 0, 0))
    );
    assert_eq!(column.pop_pivot(), None);

    // Opposite filtration
    let opp_matrix =
        matrix.with_filtration(|idx| Ok(-(matrix.filtration_value(idx).unwrap() as isize)));
    let opp_add = |column: &mut BHCol<_>, index| {
        column.add_entries(opp_matrix.column_with_filtration(index).unwrap());
    };
    let mut column = opp_matrix.empty_bhcol();
    opp_add(&mut column, 5);
    opp_add(&mut column, 4);
    assert_eq!(
        column.pop_pivot().map(|e| e.into()),
        Some((Z2::one(), 0, 0))
    );
    assert_eq!(
        column.pop_pivot().map(|e| e.into()),
        Some((Z2::one(), 1, -10))
    );
    assert_eq!(column.pop_pivot(), None);

    // Extra tests

    let mut column = matrix.empty_bhcol();
    add(&mut column, 5);
    add(&mut column, 4);
    add(&mut column, 3);
    assert_eq!(column.pop_pivot(), None);

    let mut column = matrix.empty_bhcol();
    add(&mut column, 6);
    add(&mut column, 6);
    assert_eq!(column.to_sorted_vec().len(), 0);
}

#[test]
fn test_consolidate() {
    let mat = simple_Z2_matrix(vec![vec![0], vec![0, 1]]);
    // Working over Z^2 so M^2 = Id

    let mat4 = product(product(&mat, &mat), product(&mat, &mat));

    let col1: Vec<_> = mat4.column(1).unwrap().collect();

    let col2: Vec<_> = consolidate(&mat4).column(1).unwrap().collect();

    // Lots of entries adding up
    assert_eq!(col1.len(), 5);
    // Consolidated down to single entry
    assert_eq!(col2.len(), 1);
}

#[test]
fn test_projection() {
    let base_matrix = simple_Z2_matrix(vec![
        vec![4, 3, 12],
        vec![5, 9, 4],
        vec![0, 1, 0],
        vec![1, 2, 4, 4],
    ]);

    // Dumb way to build - better to make custom oracle
    let mut projection_cols = vec![vec![]; 13];
    projection_cols[4] = vec![4];
    let projection_matrix = simple_Z2_matrix(projection_cols);

    let projected = product(&projection_matrix, &base_matrix);

    let true_matrix = simple_Z2_matrix(vec![vec![4], vec![4], vec![], vec![]]);

    assert!((0..4).all(|i| projected.eq_on_col(&true_matrix, i)))
}

#[test]
fn test_basis_reverse() {
    let base_matrix = simple_Z2_matrix(vec![
        vec![4, 3, 12],
        vec![5, 9, 4],
        vec![0, 1, 0],
        vec![1, 2, 4, 4],
    ]);
    let base_matrix = base_matrix.with_basis(vec![0, 1, 3]);
    let rev_matrix = base_matrix.reverse();

    let base_elem = base_matrix.basis().element(0);
    assert_eq!(base_elem, 0);
    let base_elem = base_matrix.basis().element(1);
    assert_eq!(base_elem, 1);
    let base_elem = base_matrix.basis().element(2);
    assert_eq!(base_elem, 3);

    let rev_elem = rev_matrix.basis().element(0);
    assert_eq!(rev_elem, Reverse(3));
    let rev_elem = rev_matrix.basis().element(1);
    assert_eq!(rev_elem, Reverse(1));
    let rev_elem = rev_matrix.basis().element(2);
    assert_eq!(rev_elem, Reverse(0));

    let unrev_matrix = (&rev_matrix).unreverse();
    let unrev_elem = unrev_matrix.basis().element(0);
    assert_eq!(unrev_elem, 0);
    let unrev_elem = unrev_matrix.basis().element(1);
    assert_eq!(unrev_elem, 1);
    let unrev_elem = unrev_matrix.basis().element(2);
    assert_eq!(unrev_elem, 3);

    let forward_col: Vec<_> = base_matrix
        .with_trivial_filtration()
        .build_bhcol(base_matrix.basis().element(0))
        .unwrap()
        .to_sorted_vec();
    println!("{:?}", forward_col);
    let rev_col: Vec<_> = rev_matrix
        .with_trivial_filtration()
        .build_bhcol(rev_matrix.basis().element(2))
        .unwrap()
        .to_sorted_vec();
    println!("{:?}", rev_col);
    let unrev_col = unrev_matrix
        .with_trivial_filtration()
        .build_bhcol(unrev_matrix.basis().element(0))
        .unwrap()
        .to_sorted_vec();
    println!("{:?}", unrev_col);
}
