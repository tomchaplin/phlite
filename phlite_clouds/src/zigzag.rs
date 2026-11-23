//! Zigzag persistence of point clouds that change: sliding windows over a time series, points that move, points
//! that arrive and leave. Each *frame* is a set of labelled points; its complex is built by any [`PointFiltration`]
//! (Rips in any metric, Čech, alpha, Delaunay–Čech, witness, or a closure of your own) cut at a threshold, with the
//! vertices relabelled so that a point keeps its name from frame to frame; consecutive complexes are joined by unions
//! or intersections and the zigzag computed by FastZigzag ([`phlite_complexes::zigzag::ComplexZigzag`]).
//!
//! ```
//! use phlite::barcode::Algorithm;
//! use phlite::reduction::Clearing;
//! use phlite_clouds::zigzag::{cloud_zigzag, sliding_windows, Rips};
//! use phlite_complexes::zigzag::Join;
//!
//! // Twelve points round a circle, in windows of nine: every window keeps the loop open, every union closes it
//! let points: Vec<Vec<f64>> = (0..12)
//!     .map(|i| {
//!         let t = i as f64 * std::f64::consts::TAU / 12.0;
//!         vec![t.cos(), t.sin()]
//!     })
//!     .collect();
//! let frames = sliding_windows(&points, 9, 1);
//! let bars = cloud_zigzag(&frames, &Rips::default(), 0.6, 1, Join::Union, Algorithm::Clearing(Clearing::new()))?;
//! assert!(bars.iter().all(|b| b.dimension == 0));
//! # Ok::<(), phlite_clouds::CloudError>(())
//! ```

use ordered_float::NotNan;
use phlite::barcode::Algorithm;
use phlite::columns::ColumnRepresentation;
use phlite_complexes::zigzag::{ComplexZigzag, Join, LevelBar, SimplicialComplex};

use crate::alpha::{filtered_delaunay, vertices, DelaunayFiltration};
use crate::cech::cech_simplices;
use crate::error::{check_points, CloudError};
use crate::rips::explicit::cliques_within;
use crate::rips::{Distances, Metric, SparseDistances};
use crate::witness::{witness_simplices, Witness};

/// A filtration of a frame's points by simplices: what each frame of a [`cloud_zigzag`] is cut from.
///
/// Implemented by [`Rips`], [`Cech`], [`Alpha`], [`DelaunayCech`] and [`WitnessFiltration`] (from the frame's
/// points), [`RipsDistances`] (from a distance matrix on the labels), and by any closure
/// `Fn(&Frame, usize, f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError>` (a weighted Rips filtration, say).
pub trait PointFiltration {
    /// The simplices of dimension at most `top` with value at most `threshold`, each with its value and vertices
    /// (indices into the frame, not labels), every face of each included.
    ///
    /// # Errors
    /// If the points can't be filtered (too few coordinates for an alpha complex, a non-finite coordinate, ...).
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError>;
}

impl<F> PointFiltration for F
where
    F: Fn(&Frame, usize, f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError>,
{
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        self(frame, top, threshold)
    }
}

fn rips_of<D: Distances + ?Sized>(graph: &D, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
    let t = NotNan::new(threshold).map_err(|_| CloudError::BadDistance { i: 0, j: 0, distance: threshold })?;
    let cliques = cliques_within(graph, top, t, usize::MAX).expect("no budget");
    Ok(cliques.into_iter().map(|(v, s)| (v.into_inner(), s)).collect())
}

/// A distance matrix restricted to some of its points (by label), births and all.
struct Restricted<'a, D> {
    distances: &'a D,
    labels: &'a [u32],
}

impl<D: Distances> Distances for Restricted<'_, D> {
    const HAS_BIRTHS: bool = D::HAS_BIRTHS;
    fn n_points(&self) -> usize {
        self.labels.len()
    }
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        self.distances.distance(self.labels[i] as usize, self.labels[j] as usize)
    }
    fn birth(&self, v: usize) -> NotNan<f64> {
        self.distances.birth(self.labels[v] as usize)
    }
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        (from..self.labels.len()).filter(move |&w| w != v)
    }
    fn degree(&self, _v: usize) -> usize {
        self.labels.len()
    }
}

/// The Vietoris–Rips filtration of a distance matrix (any [`Distances`]: dense, sparse, weighted) on the labels:
/// each frame's complex is that of the distances between its points' labels (the frame's points are ignored).
#[derive(Debug, Clone)]
pub struct RipsDistances<D> {
    pub distances: D,
}

impl<D: Distances> PointFiltration for RipsDistances<D> {
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        let n = self.distances.n_points();
        if let Some(&l) = frame.labels.iter().find(|&&l| l as usize >= n) {
            return Err(CloudError::PointOutOfRange { i: l as usize, j: l as usize, n_points: n });
        }
        let sub = Restricted { distances: &self.distances, labels: &frame.labels };
        rips_of(&sub, top, threshold)
    }
}

/// The Vietoris–Rips filtration in a metric: a simplex at the largest distance between its vertices.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rips {
    pub metric: Metric,
}

impl PointFiltration for Rips {
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        if frame.points.is_empty() {
            return Ok(vec![]);
        }
        let t = NotNan::new(threshold).map_err(|_| CloudError::BadDistance { i: 0, j: 0, distance: threshold })?;
        rips_of(&SparseDistances::from_points(&frame.points, self.metric, t)?, top, threshold)
    }
}

/// The Čech filtration: a simplex at the radius of its vertices' minimal enclosing ball.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cech;

impl PointFiltration for Cech {
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        let points = &frame.points;
        if points.is_empty() {
            return Ok(vec![]);
        }
        check_points(points)?;
        Ok(cech_simplices(points, top, Some(threshold)).into_iter().map(|(r2, s)| (r2.sqrt(), s)).collect())
    }
}

fn delaunay_simplices(
    points: &[Vec<f64>],
    filtration: DelaunayFiltration,
    top: usize,
    threshold: f64,
) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
    if points.is_empty() {
        return Ok(vec![]);
    }
    let complex = filtered_delaunay(points, filtration)?;
    let mut out = vec![];
    for (simplices, values) in complex.simplices.iter().zip(&complex.values).take(top + 1) {
        for (s, &r2) in simplices.iter().zip(values) {
            let r = r2.max(0.0).sqrt();
            if r <= threshold {
                out.push((r, vertices(s).to_vec()));
            }
        }
    }
    Ok(out)
}

/// The alpha filtration of points in the plane or in space: each Delaunay simplex at its alpha radius (repeated
/// points are merged into the first, so only it is a vertex).
#[derive(Debug, Clone, Copy, Default)]
pub struct Alpha;

impl PointFiltration for Alpha {
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        delaunay_simplices(&frame.points, DelaunayFiltration::Alpha, top, threshold)
    }
}

/// The Delaunay–Čech filtration of points in the plane or in space: each Delaunay simplex at the radius of its
/// minimal enclosing ball.
#[derive(Debug, Clone, Copy, Default)]
pub struct DelaunayCech;

impl PointFiltration for DelaunayCech {
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        delaunay_simplices(&frame.points, DelaunayFiltration::DelaunayCech, top, threshold)
    }
}

/// The witness filtration with each frame's points as landmarks, witnessed by a fixed set of `witnesses`; values as
/// [`witness_simplices`]' (`alpha²`).
#[derive(Debug, Clone)]
pub struct WitnessFiltration {
    pub witnesses: Vec<Vec<f64>>,
    pub kind: Witness,
}

impl PointFiltration for WitnessFiltration {
    fn simplices(&self, frame: &Frame, top: usize, threshold: f64) -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
        let points = &frame.points;
        if points.is_empty() {
            return Ok(vec![]);
        }
        check_points(points)?;
        Ok(witness_simplices(points, &self.witnesses, self.kind, top, threshold))
    }
}

/// A frame of a point-cloud zigzag: points, each with a label naming it in every frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Frame {
    pub labels: Vec<u32>,
    pub points: Vec<Vec<f64>>,
}

impl Frame {
    /// The points labelled by `labels`, taken from `points`.
    pub fn select(points: &[Vec<f64>], labels: impl IntoIterator<Item = u32>) -> Self {
        let labels: Vec<u32> = labels.into_iter().collect();
        let points = labels.iter().map(|&l| points[l as usize].clone()).collect();
        Self { labels, points }
    }
}

/// Windows of `window` consecutive points, `step` apart (points labelled by their index), as long as they fit.
///
/// # Panics
/// If `step` is 0.
pub fn sliding_windows(points: &[Vec<f64>], window: usize, step: usize) -> Vec<Frame> {
    assert!(step > 0, "a positive step");
    let window = window.min(points.len());
    (0..=points.len() - window)
        .step_by(step)
        .map(|start| Frame::select(points, start as u32..(start + window) as u32))
        .collect()
}

/// The same points at different times (point `j` labelled `j` in every frame): moving points.
pub fn moving_points(times: &[Vec<Vec<f64>>]) -> Vec<Frame> {
    times.iter().map(|points| Frame { labels: (0..points.len() as u32).collect(), points: points.clone() }).collect()
}

/// A frame's complex by `filtration` at `threshold` (simplices up to dimension `top`), on the frame's labels.
///
/// # Errors
/// As the filtration's.
pub fn frame_complex<F: PointFiltration + ?Sized>(
    frame: &Frame,
    filtration: &F,
    top: usize,
    threshold: f64,
) -> Result<SimplicialComplex<u32>, CloudError> {
    let simplices = filtration.simplices(frame, top, threshold)?;
    Ok(SimplicialComplex::new(simplices.into_iter().map(|(_, s)| {
        let mut s: Vec<u32> = s.into_iter().map(|v| frame.labels[v as usize]).collect();
        s.sort_unstable();
        s
    })))
}

/// The zigzag barcode, in levels (frame `i` at `2 i`, see [`LevelBar`]), of `frames`, each frame's complex by
/// `filtration` at `threshold` (simplices up to dimension `max_dim + 1`), consecutive frames joined by `join`.
///
/// # Errors
/// As the filtration's.
pub fn cloud_zigzag<F: PointFiltration + ?Sized, R: ColumnRepresentation>(
    frames: &[Frame],
    filtration: &F,
    threshold: f64,
    max_dim: usize,
    join: Join,
    algorithm: Algorithm<'_, R>,
) -> Result<Vec<LevelBar>, CloudError> {
    let mut zigzag = ComplexZigzag::new(join);
    for frame in frames {
        zigzag.push(&frame_complex(frame, filtration, max_dim + 1, threshold)?);
    }
    let mut bars = zigzag.barcode(algorithm).expect("each frame's complex is closed under faces");
    bars.retain(|b| b.dimension <= max_dim);
    Ok(bars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::Clearing;

    fn circle(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / n as f64;
                vec![t.cos(), t.sin()]
            })
            .collect()
    }

    fn run<F: PointFiltration>(frames: &[Frame], f: &F, t: f64, join: Join) -> Vec<LevelBar> {
        cloud_zigzag(frames, f, t, 1, join, Algorithm::Clearing(Clearing::new())).unwrap()
    }

    /// A circle whose points all stay: one H1 bar throughout, whichever filtration
    #[test]
    fn every_filtration_sees_the_loop() {
        let points = circle(12);
        let frames = moving_points(&[points.clone(), points.clone(), points]);
        // Neighbours are 2 sin(π/12) ≈ 0.518 apart, so Rips at 0.6 is the 12-cycle; Čech and alpha at radius 0.3
        for bars in [
            run(&frames, &Rips::default(), 0.6, Join::Union),
            run(&frames, &Cech, 0.3, Join::Union),
            run(&frames, &Alpha, 0.3, Join::Union),
            run(&frames, &DelaunayCech, 0.3, Join::Intersection),
        ] {
            assert_eq!(bars, vec![LevelBar { dimension: 0, birth: 0, death: 4 }, LevelBar { dimension: 1, birth: 0, death: 4 }]);
        }
    }

    /// Sliding windows round a circle: each window an arc, the union of consecutive ones an arc too, so no loop.
    /// Frames each missing one point (an arc each), different ones: their union is the whole circle, a loop at each
    /// union only
    #[test]
    fn arcs_round_a_circle() {
        let points = circle(12);
        let bars = run(&sliding_windows(&points, 6, 1), &Rips::default(), 0.6, Join::Union);
        assert_eq!(bars, vec![LevelBar { dimension: 0, birth: 0, death: 12 }]);
        // The same from the distance matrix
        let matrix: Vec<Vec<NotNan<f64>>> =
            points.iter().map(|p| points.iter().map(|q| Metric::Euclidean.distance(p, q)).collect()).collect();
        let frames = sliding_windows(&points, 11, 1);
        assert_eq!(
            run(&frames, &RipsDistances { distances: matrix }, 0.6, Join::Union),
            run(&frames, &Rips::default(), 0.6, Join::Union)
        );
        let missing = |m: u32| Frame::select(&points, (0..12).filter(move |&v| v != m));
        let frames = [missing(0), missing(6), missing(0)];
        let bars = run(&frames, &Rips::default(), 0.6, Join::Union);
        let loops: Vec<_> = bars.iter().filter(|b| b.dimension == 1).copied().collect();
        assert_eq!(loops, vec![LevelBar { dimension: 1, birth: 1, death: 1 }, LevelBar { dimension: 1, birth: 3, death: 3 }]);
        // Intersections: no loops, and two arcs at each intersection
        let bars = run(&frames, &Rips::default(), 0.6, Join::Intersection);
        assert!(bars.iter().all(|b| b.dimension == 0));
        assert_eq!(bars.iter().filter(|b| b.birth <= 1 && 1 <= b.death).count(), 2);
    }

    /// A closure works as a filtration
    #[test]
    fn closures() {
        let points = circle(8);
        let frames = sliding_windows(&points, 8, 1);
        let only_vertices =
            |f: &Frame, _top: usize, _t: f64| -> Result<Vec<(f64, Vec<u32>)>, CloudError> {
                Ok((0..f.points.len() as u32).map(|v| (0.0, vec![v])).collect())
            };
        let bars = run(&frames, &only_vertices, 1.0, Join::Union);
        assert_eq!(bars.len(), 8);
    }
}
