//! Zigzag persistence of videos (and of any sequence of images or volumes): frame `i`'s sublevel set at a threshold,
//! as a cubical complex `K_i`, in the zigzag `K_0 -> K_0 ∪ K_1 <- K_1 -> K_1 ∪ K_2 <- ...`, whose barcode tracks
//! features (components, loops, voids) from frame to frame: a blob that drifts keeps its bar, two that merge end one.
//! The unions are of the complexes, so a feature need only overlap itself (share a cell) from one frame to the next
//! to keep its bar.
//!
//! Each frame's complex is a [`CubicalCut`]: [`crate::cubical::periodic::explicit_cubical`] (T- or V-construction,
//! any axes periodic) cut at the threshold, a [`ZigzagComplex`] with cells keyed by their position on the doubled
//! grid, so a cell present in consecutive frames is the same cell. The zigzag is a [`ComplexZigzag`], computed by
//! FastZigzag ([`phlite_complexes::zigzag`]) on phlite's reductions. Frames can also be joined by intersections
//! (`K_0 <- K_0 ∩ K_1 -> K_1 ...`, [`Join::Intersection`]), and any sequence of [`CubicalCut`]s (each frame at its
//! own threshold, or slices of a volume) pushed onto a [`ComplexZigzag`] directly.

use std::collections::HashMap;

use phlite::barcode::Algorithm;
use phlite::columns::ColumnRepresentation;
use phlite_complexes::zigzag::{ComplexZigzag, Join, LevelBar, ZigzagComplex, ZigzagError};

use crate::cubical::periodic::{explicit_cubical, ExplicitCubical};
use crate::cubical::{Construction, Filtration};

/// An interval of a video's zigzag barcode, in levels: frame `i` is level `2 i`, the join of frames `i` and `i + 1`
/// level `2 i + 1` (see [`LevelBar`]).
pub type VideoBar = LevelBar;

/// The cubical complex of an image (or volume) cut at a threshold, as a term of a zigzag ([`ZigzagComplex`]): cells
/// keyed by their position on the doubled grid, so the same cube in two images of one shape is the same cell.
#[derive(Debug, Clone)]
pub struct CubicalCut {
    complex: ExplicitCubical,
    /// The cells kept (positions in `complex`)
    kept: Vec<usize>,
    /// Grid index to position in `complex`
    position: HashMap<usize, usize>,
}

impl CubicalCut {
    /// The cells of `pixels` (in C order of `shape`) with value at most `threshold` for sublevel sets, at least for
    /// superlevel ones, by `construction`, with the `periodic` axes wrapped round.
    ///
    /// # Panics
    /// If `pixels` doesn't fill `shape`, or the shape has more axes than [`crate::cubical::MAX_AXES`].
    pub fn new(
        pixels: &[f64],
        shape: &[usize],
        periodic: &[bool],
        threshold: f64,
        construction: Construction,
        filtration: Filtration,
    ) -> Self {
        assert_eq!(pixels.len(), shape.iter().product::<usize>(), "the pixels fill the shape");
        let cut = match filtration {
            Filtration::Sublevel => threshold,
            Filtration::Superlevel => -threshold,
        };
        let complex = explicit_cubical(pixels, shape, periodic, construction, filtration);
        let kept: Vec<usize> = (0..complex.values.len()).filter(|&c| complex.values[c] <= cut).collect();
        let position = kept.iter().map(|&c| (complex.grid[c], c)).collect();
        Self { complex, kept, position }
    }

    /// The number of cells.
    pub fn len(&self) -> usize {
        self.kept.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }
}

impl ZigzagComplex for CubicalCut {
    type Key = usize;
    fn cells(&self) -> Vec<(usize, usize)> {
        self.kept.iter().map(|&c| (self.complex.grid[c], self.complex.dimensions[c])).collect()
    }
    fn faces(&self, cell: &usize) -> Vec<usize> {
        let c = self.position[cell];
        let ExplicitCubical { faces, indptr, grid, .. } = &self.complex;
        faces[indptr[c]..indptr[c + 1]].iter().map(|&(f, _)| grid[f]).collect()
    }
}

/// The zigzag barcode of a video: `frames` (each a frame of `shape`, in C order) cut at `threshold` as cubical
/// complexes ([`CubicalCut`]) joined frame to frame by `join`, in dimensions up to `max_dim`, in levels.
///
/// # Errors
/// If FastZigzag finds an impossible operation (it can't: each frame's cut is a subcomplex).
///
/// # Panics
/// If `frames` isn't a whole number of frames of `shape`, or the shape has more axes than
/// [`crate::cubical::MAX_AXES`].
#[allow(clippy::too_many_arguments)]
pub fn video_zigzag<R: ColumnRepresentation>(
    frames: &[f64],
    shape: &[usize],
    periodic: &[bool],
    threshold: f64,
    construction: Construction,
    filtration: Filtration,
    max_dim: usize,
    join: Join,
    algorithm: Algorithm<'_, R>,
) -> Result<Vec<VideoBar>, ZigzagError> {
    let per_frame: usize = shape.iter().product();
    assert!(per_frame > 0 && frames.len().is_multiple_of(per_frame), "a whole number of frames");
    let mut zigzag = ComplexZigzag::new(join);
    for frame in frames.chunks(per_frame) {
        zigzag.push(&CubicalCut::new(frame, shape, periodic, threshold, construction, filtration));
    }
    let mut bars = zigzag.barcode(algorithm)?;
    bars.retain(|b| b.dimension <= max_dim);
    Ok(bars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::Clearing;

    /// A blob that moves keeps one H0 bar; one that splits in two gains a bar; a ring has an H1 bar while it lasts.
    #[test]
    fn blobs_and_rings() {
        let shape = [5, 7];
        let frame = |on: &[(usize, usize)]| {
            let mut f = vec![1.0; 35];
            for &(r, c) in on {
                f[r * 7 + c] = 0.0;
            }
            f
        };
        let run = |frames: Vec<Vec<f64>>| {
            video_zigzag(
                &frames.concat(),
                &shape,
                &[false, false],
                0.5,
                Construction::V,
                Filtration::Sublevel,
                1,
                Join::Union,
                Algorithm::Clearing(Clearing::new()),
            )
            .unwrap()
        };
        // Moving right one pixel at a time (overlapping): one component throughout
        let moving = run((0..4).map(|i| frame(&[(2, i), (2, i + 1)])).collect());
        assert_eq!(moving, vec![VideoBar { dimension: 0, birth: 0, death: 6 }]);
        // Splitting: one bar, then a second from the last frame
        let splitting = run(vec![frame(&[(2, 2), (2, 3), (2, 4)]), frame(&[(2, 2), (2, 4)])]);
        assert_eq!(
            splitting,
            vec![VideoBar { dimension: 0, birth: 0, death: 2 }, VideoBar { dimension: 0, birth: 2, death: 2 }]
        );
        // A ring in the middle frame only, between frames of its centre alone: the unions are of the complexes (the
        // centre and the ring, unconnected), so the loop lives from the first union to the second
        let ring = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 4), (3, 2), (3, 3), (3, 4)];
        let ringed = run(vec![frame(&[(2, 3)]), frame(&ring), frame(&[(2, 3)])]);
        assert!(ringed.contains(&VideoBar { dimension: 1, birth: 1, death: 3 }), "{ringed:?}");
        assert_eq!(ringed.iter().filter(|b| b.dimension == 0).count(), 3);
    }
}
