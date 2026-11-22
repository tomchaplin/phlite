//! Zigzag persistence of videos (and of any sequence of images or volumes): frame `i`'s sublevel set at a threshold,
//! as a cubical complex `K_i`, in the zigzag `K_0 -> K_0 ∪ K_1 <- K_1 -> K_1 ∪ K_2 <- ...`, whose barcode tracks
//! features (components, loops, voids) from frame to frame: a blob that drifts keeps its bar, two that merge end one.
//! The unions are of the complexes, so a feature need only overlap itself (share a cell) from one frame to the next
//! to keep its bar.
//!
//! Each frame's complex is [`crate::cubical::periodic::explicit_cubical`] (T- or V-construction, any axes periodic)
//! cut at the threshold, cells keyed by their position on the doubled grid, so a cell present in consecutive frames is
//! the same cell; the zigzag is computed by FastZigzag ([`phlite_complexes::zigzag`]) on phlite's reductions.

use std::collections::{BTreeSet, HashMap};

use phlite::barcode::Algorithm;
use phlite::columns::ColumnRepresentation;
use phlite_complexes::zigzag::{zigzag_barcode, Operation, ZigzagError};

use crate::cubical::periodic::explicit_cubical;
use crate::cubical::{Construction, Filtration};

/// An interval of a video's zigzag barcode, in *levels*: frame `i` is level `2 i`, the union of frames `i` and `i + 1`
/// level `2 i + 1`. The class is alive at every level from `birth` to `death` (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VideoBar {
    pub dimension: usize,
    pub birth: usize,
    pub death: usize,
}

/// The zigzag barcode of a video: `frames` (each with `pixels_per_frame` pixels in C order of `shape`) cut at
/// `threshold` (pixels at most `threshold` for sublevel sets, at least for superlevel ones), in dimensions up to
/// `max_dim`.
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
    algorithm: Algorithm<'_, R>,
) -> Result<Vec<VideoBar>, ZigzagError> {
    let per_frame: usize = shape.iter().product();
    assert!(per_frame > 0 && frames.len().is_multiple_of(per_frame), "a whole number of frames");
    let cut = match filtration {
        Filtration::Sublevel => threshold,
        Filtration::Superlevel => -threshold,
    };
    let mut ops: Vec<Operation<usize>> = vec![];
    // The position of each level (frame i at 2i, unions between)
    let mut levels: Vec<(usize, usize)> = vec![];
    let mut current: BTreeSet<(usize, usize)> = BTreeSet::new(); // (dimension, grid index)
    for (i, frame) in frames.chunks(per_frame).enumerate() {
        let complex = explicit_cubical(frame, shape, periodic, construction, filtration);
        let position: HashMap<usize, usize> = complex.grid.iter().enumerate().map(|(c, &g)| (g, c)).collect();
        let next: BTreeSet<(usize, usize)> = (0..complex.values.len())
            .filter(|&c| complex.values[c] <= cut)
            .map(|c| (complex.dimensions[c], complex.grid[c]))
            .collect();
        // Insert what is new, faces first (by dimension)
        for &(dimension, cell) in next.difference(&current) {
            let c = position[&cell];
            let faces: Vec<usize> =
                complex.faces[complex.indptr[c]..complex.indptr[c + 1]].iter().map(|&(f, _)| complex.grid[f]).collect();
            ops.push(Operation::Insert { cell, dimension, faces });
        }
        if i > 0 {
            levels.push((2 * i - 1, ops.len()));
        }
        // Delete what is gone, cofaces first
        for &(_, cell) in current.difference(&next).collect::<Vec<_>>().into_iter().rev() {
            ops.push(Operation::Delete(cell));
        }
        levels.push((2 * i, ops.len()));
        current = next;
    }
    let bars = zigzag_barcode(&ops, algorithm)?;
    let mut out = vec![];
    for bar in bars.into_iter().filter(|b| b.dimension <= max_dim) {
        let mut inside = levels.iter().filter(|&&(_, p)| bar.birth <= p && p <= bar.death).map(|&(l, _)| l);
        if let Some(birth) = inside.next() {
            let death = inside.last().unwrap_or(birth);
            out.push(VideoBar { dimension: bar.dimension, birth, death });
        }
    }
    out.sort_unstable();
    Ok(out)
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
