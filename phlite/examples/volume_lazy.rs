//! The tutorial in the phlite book, steps 5–7: the volume's cubical complex from `volume_basic.rs`, now with the
//! hooks that let phlite skip almost all of it. See `docs/book/src/tutorial/`.
//!
//! ```text
//! cargo run --release -p phlite --example volume_lazy --features parallel [size]
//! ```

#[path = "volume/volume.rs"]
mod volume;

use std::time::Instant;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar},
    columns::{ColumnEntry, HeapColumns},
    fields::{NonZeroCoefficient, Z2},
    matrices::{BasisElement, HasRowFiltration, MatrixOracle, SplitBasis},
    reduction::Clearing,
};
use volume::{peak_memory_mb, reset_peak_memory, smooth_noise, Volume};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell(pub u64);

impl BasisElement for Cell {}

/// The boundary matrix (only needed as the coboundary's transpose, for representatives).
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

// ANCHOR: order
/// The filtration order on cells: by value, then number.
fn key(volume: &Volume, cell: u64) -> (NotNan<f32>, u64) {
    (NotNan::new(volume.value(cell)).unwrap(), cell)
}

/// The oldest coface of a cell (the first column in its row of the boundary), and its youngest face (its pivot).
fn oldest_coface(volume: &Volume, cell: u64) -> Option<u64> {
    volume.cofaces(cell).min_by_key(|&c| key(volume, c))
}

fn youngest_face(volume: &Volume, cell: u64) -> Option<u64> {
    volume.faces(cell).max_by_key(|&f| key(volume, f))
}
// ANCHOR_END: order

// ANCHOR: hooks
impl HasRowFiltration for Coboundary<'_> {
    type FiltrationT = NotNan<f32>;

    fn filtration_value(&self, row: Cell) -> NotNan<f32> {
        NotNan::new(self.0.value(row.0)).unwrap()
    }

    // Step 5, emergent pairs: cohomology's pivot of a column is its *smallest* coface. No coface comes before the cell
    // itself, and cofaces are enumerated in increasing order of their numbers, so the first coface with the cell's own
    // value is the smallest: stop there. Most columns need nothing else, so this usually looks at one or two cofaces.
    fn column_min(&self, col: Cell) -> Option<ColumnEntry<NotNan<f32>, Cell, Z2>> {
        let own = self.filtration_value(col);
        let mut best: Option<ColumnEntry<NotNan<f32>, Cell, Z2>> = None;
        for coface in self.0.cofaces(col.0) {
            let entry: ColumnEntry<_, _, _> = (Z2::one(), Cell(coface), self.filtration_value(Cell(coface))).into();
            if entry.filtration_value == own {
                return Some(entry);
            }
            if best.as_ref().is_none_or(|b| entry < *b) {
                best = Some(entry);
            }
        }
        best
    }

    // Step 5, apparent pairs without the transpose: the columns with an entry in row `r` of the coboundary are the
    // faces of `r`, so the first and last are its oldest and youngest faces
    fn first_column_in_row(&self, row: Cell) -> Option<Cell> {
        self.0.faces(row.0).min_by_key(|&f| key(self.0, f)).map(Cell)
    }

    fn last_column_in_row(&self, row: Cell) -> Option<Cell> {
        youngest_face(self.0, row.0).map(Cell)
    }
}
// ANCHOR_END: hooks

// ANCHOR: reduction_basis
/// Step 6: whether a cell is in a zero-length apparent pair, as either its birth or its death. Such cells needn't be
/// in the basis at all: the reduction finds them again as owners (through `first_column_in_row`) when it needs them.
fn in_zero_apparent_pair(volume: &Volume, cell: u64) -> bool {
    let value = volume.value(cell);
    // A birth: its oldest coface has the same value, and the cell is that coface's youngest face
    let birth = oldest_coface(volume, cell)
        .is_some_and(|coface| volume.value(coface) == value && youngest_face(volume, coface) == Some(cell));
    // A death: its youngest face has the same value, and the cell is that face's oldest coface
    let death = youngest_face(volume, cell)
        .is_some_and(|face| volume.value(face) == value && oldest_coface(volume, face) == Some(cell));
    birth || death
}

/// The columns to reduce, up to dimension `max_dim`, without the cells of zero-length apparent pairs. Cells are
/// streamed (never all held at once), so memory goes on the cells that are kept.
fn reduction_basis(volume: &Volume, max_dim: usize) -> SplitBasis<Cell> {
    let mut dims: Vec<Vec<Cell>> = vec![vec![]; max_dim + 1];
    for cell in 0..volume.n_cells() {
        let dim = volume.dimension(cell);
        if dim <= max_dim && !in_zero_apparent_pair(volume, cell) {
            dims[dim].push(Cell(cell));
        }
    }
    for cells in &mut dims {
        cells.sort_by_key(|&cell| key(volume, cell.0));
    }
    SplitBasis::from_dimensions(dims)
}
// ANCHOR_END: reduction_basis

/// Every cell up to dimension `max_dim` (as in `volume_basic.rs`).
fn full_basis(volume: &Volume, max_dim: usize) -> SplitBasis<Cell> {
    let mut dims: Vec<Vec<Cell>> = vec![vec![]; max_dim + 1];
    for cell in 0..volume.n_cells() {
        let dim = volume.dimension(cell);
        if dim <= max_dim {
            dims[dim].push(Cell(cell));
        }
    }
    for cells in &mut dims {
        cells.sort_by_key(|&cell| key(volume, cell.0));
    }
    SplitBasis::from_dimensions(dims)
}

fn basis_size(basis: &SplitBasis<Cell>) -> usize {
    use phlite::matrices::ColBasis;
    basis.size()
}

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
    println!("{label:<50} {:>9.1} ms", start.elapsed().as_secs_f64() * 1e3);
    out
}

fn main() {
    let size: usize = std::env::args()
        .nth(1)
        .map_or(96, |s| s.parse().expect("a volume size"));
    let volume = smooth_noise(size, 1);
    println!(
        "A {size}³ volume: {} MB of voxels, {} cells\n",
        volume.voxels.len() * 4 / (1 << 20),
        volume.n_cells()
    );

    // ANCHOR: step5
    // Step 5: the hooks find pivots (emergent pairs) and apparent pairs without building columns or the transpose.
    // Columns are cheap to compute, so recompute them rather than caching, and keep the working column as a heap
    let algorithm = Clearing::new().with_columns(HeapColumns).cache_columns(false);
    reset_peak_memory();
    let full = timed("build every cell up to dimension 2", || full_basis(&volume, 2));
    let delta_full = Coboundary(&volume).with_basis(full);
    let hooks = timed("cohomology, clearing, hooks", || {
        barcode::cohomology(&delta_full, Algorithm::Clearing(algorithm.clone()))
    });
    // ANCHOR_END: step5
    let full_size = basis_size(phlite::matrices::HasColBasis::basis(&delta_full));
    let full_peak = peak_memory_mb();
    drop(delta_full);
    reset_peak_memory();

    // ANCHOR: step6
    // Step 6: leave the cells of zero-length apparent pairs out of the basis altogether
    let reduced = timed("build the reduction basis", || reduction_basis(&volume, 2));
    let reduced_size = basis_size(&reduced);
    let delta = Coboundary(&volume).with_basis(reduced);
    let lazy = timed("cohomology, clearing, reduction basis", || {
        barcode::cohomology(&delta, Algorithm::Clearing(algorithm.clone()))
    });
    println!(
        "    basis: {reduced_size} cells instead of {full_size} ({:.1}%)",
        100.0 * reduced_size as f64 / full_size as f64
    );
    if let (Some(full), Some(lazy)) = (full_peak, peak_memory_mb()) {
        println!("    peak memory: {lazy:.0} MB instead of {full:.0} MB");
    }
    // ANCHOR_END: step6

    // ANCHOR: step7
    // Step 7: representatives, by the involution, which needs the transpose (the boundary)
    let delta_t = (&delta).with_transpose(Boundary(&volume));
    let with_reps = timed("... with representatives", || {
        barcode::cohomology_with_involution(&delta_t, Algorithm::Clearing(algorithm.clone()))
    });
    let longest = with_reps
        .iter()
        .filter(|bar| volume.dimension(bar.birth.0) == 2 && bar.death.is_some())
        .max_by_key(|bar| bar.death.unwrap().1 - bar.birth_value)
        .expect("a finite cavity");
    println!(
        "    the longest finite H2 bar, [{}, {}), is a closed surface of {} squares",
        longest.birth_value,
        longest.death.unwrap().1,
        longest.representative.as_ref().unwrap().len()
    );

    // ... and all cores
    #[cfg(feature = "parallel")]
    let parallel = timed("cohomology, parallel clearing", || {
        let parallel = Clearing::new()
            .with_columns(HeapColumns)
            .cache_columns(false)
            .parallel();
        barcode::cohomology(&delta, Algorithm::ParallelClearing(parallel))
    });
    // ANCHOR_END: step7

    assert_eq!(values(&hooks), values(&lazy));
    assert_eq!(values(&hooks), values(&with_reps));
    #[cfg(feature = "parallel")]
    assert_eq!(values(&hooks), values(&parallel));
    println!("\n{} bars, the same every way", lazy.len());
}
