//! The tutorial in the phlite book, steps 1–4: persistent homology of a 3D greyscale volume, as a cubical complex
//! computed lazily from its voxels. See `docs/book/src/tutorial/`.
//!
//! ```text
//! cargo run --release -p phlite --example volume_basic [size]
//! ```

#[path = "volume/volume.rs"]
mod volume;

use std::time::Instant;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar},
    fields::{NonZeroCoefficient, Z2},
    matrices::{BasisElement, HasRowFiltration, MatrixOracle, SplitBasis},
    reduction::{Clearing, Standard},
};
use volume::{smooth_noise, Volume};

// ANCHOR: cell
/// A cell of the volume's cubical complex, by its number (see `Volume`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell(pub u64);

// Mark `Cell` as a label for rows and columns. Its order (derived: by number) breaks ties between equal values.
impl BasisElement for Cell {}
// ANCHOR_END: cell

// ANCHOR: boundary
/// Step 1: the boundary matrix, as an oracle. Nothing is stored: a column is computed when it is asked for.
#[derive(Clone, Copy)]
pub struct Boundary<'a>(pub &'a Volume);

impl MatrixOracle for Boundary<'_> {
    type CoefficientField = Z2;
    type ColT = Cell;
    type RowT = Cell;

    fn column(&self, col: Cell) -> impl Iterator<Item = (Z2, Cell)> {
        self.0.faces(col.0).map(|face| (Z2::one(), Cell(face)))
    }
}
// ANCHOR_END: boundary

// ANCHOR: basis_by_number
/// The columns to reduce: every cell up to dimension `max_dim`, by dimension, each in the order of the rows. With no
/// filtration that is just the order on labels: by number.
fn basis_by_number(volume: &Volume, max_dim: usize) -> SplitBasis<Cell> {
    let mut dims: Vec<Vec<Cell>> = vec![vec![]; max_dim + 1];
    for cell in 0..volume.n_cells() {
        let dim = volume.dimension(cell);
        if dim <= max_dim {
            dims[dim].push(Cell(cell));
        }
    }
    SplitBasis::from_dimensions(dims)
}
// ANCHOR_END: basis_by_number

// ANCHOR: basis
/// As `basis_by_number`, but in filtration order: by value, then number.
fn basis(volume: &Volume, max_dim: usize) -> SplitBasis<Cell> {
    let mut dims: Vec<Vec<Cell>> = vec![vec![]; max_dim + 1];
    for cell in 0..volume.n_cells() {
        let dim = volume.dimension(cell);
        if dim <= max_dim {
            dims[dim].push(Cell(cell));
        }
    }
    for cells in &mut dims {
        cells.sort_by_key(|&cell| (NotNan::new(volume.value(cell.0)).unwrap(), cell));
    }
    SplitBasis::from_dimensions(dims)
}
// ANCHOR_END: basis

// ANCHOR: filtration
/// Step 2: the filtration on rows (cells), read off the volume. Implementing it on the oracle, rather than attaching
/// a closure with `with_filtration`, leaves room to override the optional hooks later.
impl HasRowFiltration for Boundary<'_> {
    type FiltrationT = NotNan<f32>;

    fn filtration_value(&self, row: Cell) -> NotNan<f32> {
        NotNan::new(self.0.value(row.0)).unwrap()
    }
}
// ANCHOR_END: filtration

// ANCHOR: coboundary
/// Step 3: the coboundary matrix (the transpose of the boundary), with the same filtration.
#[derive(Clone, Copy)]
pub struct Coboundary<'a>(pub &'a Volume);

impl MatrixOracle for Coboundary<'_> {
    type CoefficientField = Z2;
    type ColT = Cell;
    type RowT = Cell;

    fn column(&self, col: Cell) -> impl Iterator<Item = (Z2, Cell)> {
        self.0.cofaces(col.0).map(|coface| (Z2::one(), Cell(coface)))
    }
}

impl HasRowFiltration for Coboundary<'_> {
    type FiltrationT = NotNan<f32>;

    fn filtration_value(&self, row: Cell) -> NotNan<f32> {
        NotNan::new(self.0.value(row.0)).unwrap()
    }
}
// ANCHOR_END: coboundary

/// The number of bars in each dimension.
fn summary<F>(volume: &Volume, bars: &[Bar<Cell, F, Z2>]) -> String {
    let mut counts = [0; 4];
    for bar in bars {
        counts[volume.dimension(bar.birth.0)] += 1;
    }
    format!("{} bars in H0, {} in H1 and {} in H2", counts[0], counts[1], counts[2])
}

/// Bars as (birth value, death value), sorted, to compare barcodes computed in different ways.
fn values(bars: &[Bar<Cell, NotNan<f32>, Z2>]) -> Vec<(NotNan<f32>, Option<NotNan<f32>>)> {
    let mut out: Vec<_> = bars
        .iter()
        .map(|bar| (bar.birth_value, bar.death.map(|(_, value)| value)))
        .collect();
    out.sort();
    out
}

fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    println!("{label:<44} {:>9.1} ms", start.elapsed().as_secs_f64() * 1e3);
    out
}

fn main() {
    let size: usize = std::env::args()
        .nth(1)
        .map_or(24, |s| s.parse().expect("a volume size"));
    let volume = smooth_noise(size, 1);
    println!("A {size}³ volume: {} cells\n", volume.n_cells());

    // ANCHOR: step1
    // Step 1: no filtration (every cell at time 0): every finite bar has length zero, so only the essential ones are
    // left, which count the holes of a solid cube: one component, and nothing else
    let d = Boundary(&volume).with_trivial_filtration();
    let d = d.with_basis(basis_by_number(&volume, 3));
    let betti = barcode::homology(&d, Algorithm::Standard(Standard::new()), false);
    println!("No filtration: {}", summary(&volume, &betti));
    // ANCHOR_END: step1

    // ANCHOR: step2
    // Step 2: the lower-star filtration, by homology, with the standard algorithm.
    // Every cell, cubes included, is a column.
    let d = Boundary(&volume).with_basis(basis(&volume, 3));
    let standard = timed("homology, standard algorithm", || {
        barcode::homology(&d, Algorithm::Standard(Standard::new()), false)
    });
    println!("    {}", summary(&volume, &standard));
    // ANCHOR_END: step2

    // ANCHOR: step3
    // Step 3: cohomology with clearing. Columns stop at the squares (for H0, H1 and H2): the cubes only ever appear
    // as rows, cofaces of squares, and are never enumerated
    let delta = Coboundary(&volume).with_basis(basis(&volume, 2));
    let cohomology = timed("cohomology, clearing", || {
        barcode::cohomology(&delta, Algorithm::Clearing(Clearing::new()))
    });
    // ANCHOR_END: step3

    // ANCHOR: step4
    // Step 4: attach the transpose, so that apparent pairs need no reduction, and zero-length ones aren't recorded
    let delta_t = (&delta).with_transpose(Boundary(&volume));
    let apparent = timed("cohomology, clearing, apparent pairs", || {
        barcode::cohomology(&delta_t, Algorithm::Clearing(Clearing::new()))
    });
    // ANCHOR_END: step4

    assert_eq!(values(&standard), values(&cohomology));
    assert_eq!(values(&standard), values(&apparent));
    println!("\nThe same barcode every way");
}
