//! Alpha persistent homology of points in the plane or in space.
//!
//! The alpha complex at radius `r` is the nerve of the balls of radius `r` around the points, each cut down to its
//! point's Voronoi cell: a subcomplex of the Delaunay triangulation with the homotopy type of the union of the balls
//! (so it gives the Čech filtration's barcode), but with far fewer simplices than Rips or Čech.
//!
//! [`alpha_barcode`] computes its barcode; [`alpha_complex`] gives the filtered complex itself. In Python, this is
//! `phlite.clouds.alpha`.
//!
//! # How it is put together
//!
//! * [`delaunay`] triangulates the points: by [`spade`] in the plane, and in space by [`axiolid_tetrahedralize`],
//!   which decides every orientation and in-sphere test exactly and breaks ties (cospherical points) by a symbolic
//!   perturbation. Repeated points are merged, and points spanning fewer dimensions than they have coordinates
//!   are triangulated exactly in their own span: collinear points as a path, and coplanar points in space by adding
//!   an apex off their plane and keeping the triangles opposite it.
//! * Every face of the triangulation gets its *alpha value*, the squared radius at which it appears (as in
//!   Edelsbrunner's definition and GUDHI's implementation), from the top dimension down: a simplex appears at the
//!   squared radius of its smallest circumsphere, if that sphere has no other vertex of the triangulation strictly
//!   inside it (it is *Gabriel*; checking the vertices opposite it in its cofacets is enough), and otherwise when its
//!   first cofacet appears. The test of whether a vertex is inside a smallest circumsphere is exact (written without
//!   division, and evaluated in floating point, double-double or big integers, as a bound on its rounding error
//!   requires); radii are accurate to an ulp or two the same way; and a face whose radius exactly equals that
//!   inherited from a cofacet gets exactly that value, so a bar between them has length zero.
//! * The filtered complex is reduced as an explicit complex by [`phlite_complexes`], with its cells numbered by
//!   dimension, so that a face always enters before a coface with the same value.

use std::collections::HashMap;

use crate::error::{check_points, CloudError};

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Bar, Method},
    columns::ColumnRepresentation,
    fields::Z2,
};
use phlite_complexes::complex::{Cell, Chain, Complex, RawChain};

/// A simplex: its vertices (indices of input points) in increasing order, padded with [`u32::MAX`].
pub type Simplex = [u32; 4];

const PAD: u32 = u32::MAX;

/// The number of vertices of `simplex`.
#[inline]
fn n_vertices(simplex: &Simplex) -> usize {
    simplex.iter().take_while(|&&v| v != PAD).count()
}

/// The vertices of `simplex`.
#[inline]
pub fn vertices(simplex: &Simplex) -> &[u32] {
    &simplex[..n_vertices(simplex)]
}

/// `simplex` without its `i`-th vertex.
#[inline]
fn facet(simplex: &Simplex, k: usize, i: usize) -> Simplex {
    let mut out = [PAD; 4];
    let mut j = 0;
    for (position, &v) in simplex[..k].iter().enumerate() {
        if position != i {
            out[j] = v;
            j += 1;
        }
    }
    out
}

// ====== Delaunay triangulations ===========================================

/// A Delaunay triangulation of a set of points, by its top-dimensional simplices.
#[derive(Debug, Clone)]
pub struct Delaunay {
    /// The dimension of the triangulation: that of the points' affine hull (0 to 3).
    pub dimension: usize,
    /// The top-dimensional simplices (`dimension + 1` vertices each, which are input point indices).
    pub simplices: Vec<Simplex>,
    /// For each input point, the input point it was merged into: itself, or the first earlier point with the same
    /// coordinates.
    pub vertex_of: Vec<u32>,
}

/// Triangulate points in the plane (given by their coordinates, labelled `labels`) with spade.
fn delaunay_2d(coords: &[[f64; 2]], labels: &[u32]) -> Result<(usize, Vec<Simplex>), CloudError> {
    use spade::{DelaunayTriangulation, Point2, Triangulation};
    let mut triangulation: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    let mut label_of = HashMap::new();
    for (p, &label) in coords.iter().zip(labels) {
        let handle = triangulation
            .insert(Point2::new(p[0], p[1]))
            .map_err(|e| CloudError::Triangulation(format!("{e:?}")))?;
        label_of.entry(handle).or_insert(label);
    }
    let sorted = |mut vs: Vec<u32>| {
        vs.sort_unstable();
        let mut simplex = [PAD; 4];
        simplex[..vs.len()].copy_from_slice(&vs);
        simplex
    };
    let triangles: Vec<Simplex> = triangulation
        .inner_faces()
        .map(|face| sorted(face.vertices().iter().map(|v| label_of[&v.fix()]).collect()))
        .collect();
    if !triangles.is_empty() {
        return Ok((2, triangles));
    }
    // Collinear: a path
    let edges: Vec<Simplex> = triangulation
        .undirected_edges()
        .map(|edge| sorted(edge.vertices().iter().map(|v| label_of[&v.fix()]).collect()))
        .collect();
    Ok((1, edges))
}

/// The "triangulation" of collinear points (given by their coordinates, labelled `labels`): a path, in order along
/// their line, which for collinear points is their order by coordinates (so exact).
fn path(coords: &[Vec<f64>], labels: &[u32]) -> (usize, Vec<Simplex>) {
    let mut order: Vec<usize> = (0..coords.len()).collect();
    order.sort_by(|&i, &j| {
        coords[i]
            .iter()
            .zip(&coords[j])
            .map(|(a, b)| a.total_cmp(b))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let edges = order
        .windows(2)
        .map(|pair| {
            let (a, b) = (labels[pair[0]], labels[pair[1]]);
            [a.min(b), a.max(b), PAD, PAD]
        })
        .collect();
    (1, edges)
}

fn coord3(p: &[f64]) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: p[0],
        y: p[1],
        z: p.get(2).copied().unwrap_or(0.0),
    }
}

/// Whether `a`, `b` and `p` are collinear, decided exactly: the three projections onto coordinate planes are all
/// collinear.
fn collinear(a: &[f64], b: &[f64], p: &[f64]) -> bool {
    let (a, b, p) = (coord3(a), coord3(b), coord3(p));
    let project = |c: robust::Coord3D<f64>, i: usize, j: usize| {
        let c = [c.x, c.y, c.z];
        robust::Coord { x: c[i], y: c[j] }
    };
    [(0, 1), (1, 2), (0, 2)]
        .iter()
        .all(|&(i, j)| robust::orient2d(project(a, i, j), project(b, i, j), project(p, i, j)) == 0.0)
}

/// A point off the line through the first two points (distinct), if there is one.
fn off_the_line(coords: &[Vec<f64>]) -> Option<usize> {
    (2..coords.len()).find(|&k| !collinear(&coords[0], &coords[1], &coords[k]))
}

/// Coordinates the triangulators accept, with the same Delaunay triangulation: scaled by a power of two (exactly) so
/// that the largest has magnitude in `[1, 2)`, and those below `2^-90` of that flushed to zero (axiolid's exact
/// predicates refuse magnitudes below `2^-100`, and spade's below `2^-142`). Points that flushing makes equal are
/// merged, as repeated points are.
fn normalise(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    scale_and_flush(points, true)
}

/// The points with the coordinates [`normalise`] would flush set to zero, but in their own scale: the points the
/// alpha values are computed from, so that they and the triangulation agree.
pub fn flush(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    scale_and_flush(points, false)
}

fn scale_and_flush(points: &[Vec<f64>], scale: bool) -> Vec<Vec<f64>> {
    let largest = points.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    if largest == 0.0 {
        return points.to_vec();
    }
    // In two steps, since 2^-exponent alone may not be representable (for subnormal or huge coordinates)
    let exponent = largest.log2().floor() as i32;
    let (half, rest) = (exponent / 2, exponent - exponent / 2);
    let (a, b) = (2f64.powi(-half), 2f64.powi(-rest));
    let tiny = 2f64.powi(-90);
    points
        .iter()
        .map(|p| {
            p.iter()
                .map(|&x| {
                    let y = x * a * b;
                    if y.abs() < tiny {
                        0.0
                    } else if scale {
                        y
                    } else {
                        x
                    }
                })
                .collect()
        })
        .collect()
}

/// The Delaunay triangulation of `points`, each a slice of 2 or 3 finite coordinates (all of the same length).
///
/// # Errors
/// If there are no points, they don't all have 2 or all have 3 coordinates, a coordinate is infinite or `NaN`, or (a
/// bug) the triangulation fails.
pub fn delaunay(points: &[Vec<f64>]) -> Result<Delaunay, CloudError> {
    if points.is_empty() {
        return Err(CloudError::NoPoints);
    }
    let dim = points[0].len();
    if !(2..=3).contains(&dim) {
        return Err(CloudError::UnsupportedDimension { dimension: dim });
    }
    check_points(points)?;
    let n = points.len();
    let normalised = normalise(points);
    // Merge repeated points into the first (comparing coordinates exactly, with -0 equal to 0)
    let mut first = HashMap::new();
    let mut vertex_of = Vec::with_capacity(n);
    let mut distinct: Vec<u32> = vec![];
    for (i, p) in normalised.iter().enumerate() {
        let key: Vec<u64> = p.iter().map(|&x| (x + 0.0).to_bits()).collect();
        let v = *first.entry(key).or_insert_with(|| {
            distinct.push(i as u32);
            i as u32
        });
        vertex_of.push(v);
    }
    let coords: Vec<Vec<f64>> = distinct
        .iter()
        .map(|&i| normalised[i as usize].clone())
        .collect();
    let (dimension, simplices) = if distinct.len() == 1 {
        (0, vec![[distinct[0], PAD, PAD, PAD]])
    } else if off_the_line(&coords).is_none() {
        path(&coords, &distinct)
    } else if dim == 2 {
        let planar: Vec<[f64; 2]> = coords.iter().map(|p| [p[0], p[1]]).collect();
        delaunay_2d(&planar, &distinct)?
    } else {
        delaunay_3d(&coords, &distinct)?
    };
    Ok(Delaunay {
        dimension,
        simplices,
        vertex_of,
    })
}

/// Triangulate points in space (distinct, not all collinear), falling back to their plane if they are coplanar.
fn delaunay_3d(coords: &[Vec<f64>], labels: &[u32]) -> Result<(usize, Vec<Simplex>), CloudError> {
    use axiolid_tetrahedralize::{Delaunay3, Delaunay3Error};
    use glam::DVec3;
    let mut points: Vec<DVec3> = coords.iter().map(|p| DVec3::new(p[0], p[1], p[2])).collect();
    let sorted = |tet: [usize; 4]| {
        let mut vs = tet.map(|i| labels[i]);
        vs.sort_unstable();
        vs
    };
    match Delaunay3::from_points(&points) {
        Ok(triangulation) => Ok((3, triangulation.tetrahedra().tetrahedra.into_iter().map(sorted).collect())),
        Err(Delaunay3Error::NotFullDimensional { .. }) => {
            // Coplanar. Add an apex off their plane and triangulate in space: every tetrahedron then has the apex as
            // a vertex, and its sphere meets the plane in exactly the circumcircle of the opposite triangle, so these
            // triangles are the points' Delaunay triangulation in their plane, decided exactly (ties broken by the
            // same symbolic perturbation) rather than in rounded coordinates on the plane
            let (a, b) = (coord3(&coords[0]), coord3(&coords[1]));
            let c = coord3(&coords[off_the_line(coords).expect("not collinear")]);
            // The coordinates are normalised to at most 2 in magnitude, so a unit step is well within range; one of
            // the three axes leaves the plane, exactly
            let apex = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                .iter()
                .map(|e| DVec3::new(a.x + e[0], a.y + e[1], a.z + e[2]))
                .find(|p| robust::orient3d(a, b, c, robust::Coord3D { x: p.x, y: p.y, z: p.z }) != 0.0)
                .expect("some axis leaves a plane");
            let apex_index = points.len();
            points.push(apex);
            let triangulation = Delaunay3::from_points(&points)
                .map_err(|e| CloudError::Triangulation(format!("{e:?}")))?;
            let triangles = triangulation
                .tetrahedra()
                .tetrahedra
                .into_iter()
                .filter_map(|tet| {
                    let mut vs: Vec<u32> = tet.iter().filter(|&&i| i != apex_index).map(|&i| labels[i]).collect();
                    (vs.len() == 3).then(|| {
                        vs.sort_unstable();
                        [vs[0], vs[1], vs[2], PAD]
                    })
                })
                .collect();
            Ok((2, triangles))
        }
        Err(e) => Err(CloudError::Triangulation(format!("{e:?}"))),
    }
}

// ====== Circumspheres =====================================================

/// The arithmetic the in-sphere test needs, so that one formula serves floating point, error bounds and exact
/// integers.
trait Ring:
    Clone
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    fn zero() -> Self;
}

impl Ring for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl Ring for num_bigint::BigInt {
    fn zero() -> Self {
        num_bigint::BigInt::from(0)
    }
}

/// A bound on the absolute value of a quantity computed in floating point: evaluating a formula on `Magnitude`s of
/// its inputs (where subtraction adds) bounds every intermediate value, and so the rounding error.
#[derive(Clone, Copy)]
struct Magnitude(f64);

impl std::ops::Add for Magnitude {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Magnitude(self.0 + other.0)
    }
}

impl std::ops::Sub for Magnitude {
    type Output = Self;
    // The magnitude of a difference is at most the sum of the magnitudes
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: Self) -> Self {
        Magnitude(self.0 + other.0)
    }
}

impl std::ops::Mul for Magnitude {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Magnitude(self.0 * other.0)
    }
}

impl Ring for Magnitude {
    fn zero() -> Self {
        Magnitude(0.0)
    }
}

/// A number as the unevaluated sum of two floats (about 32 significant digits): enough to evaluate the formulas here
/// accurately in all but the most degenerate cases, at a fraction of the cost of big integers. Each operation's error
/// is at most a small multiple of `2^-104` times the magnitudes of its operands.
#[derive(Clone, Copy)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    fn new(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    #[inline]
    fn two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        Self { hi: s, lo: e }
    }

    #[inline]
    fn quick_two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        Self { hi: s, lo: b - (s - a) }
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

impl std::ops::Add for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        let s = Self::two_sum(self.hi, other.hi);
        Self::quick_two_sum(s.hi, s.lo + self.lo + other.lo)
    }
}

impl std::ops::Sub for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self + DoubleDouble {
            hi: -other.hi,
            lo: -other.lo,
        }
    }
}

impl std::ops::Mul for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        let p = self.hi * other.hi;
        let e = self.hi.mul_add(other.hi, -p) + (self.hi * other.lo + self.lo * other.hi);
        Self::quick_two_sum(p, e)
    }
}

impl Ring for DoubleDouble {
    fn zero() -> Self {
        DoubleDouble::new(0.0)
    }
}

/// The relative error, per unit of magnitude, of the few dozen operations in these formulas: generous multiples of
/// the unit roundoff of floats and of double-doubles.
const FLOAT_ERROR: f64 = 128.0 * f64::EPSILON;
const DOUBLE_DOUBLE_ERROR: f64 = 256.0 * f64::EPSILON * f64::EPSILON;

/// How each kind of number forms the difference `a - b` of two coordinates.
trait Difference: Ring {
    fn difference(a: f64, b: f64) -> Self;
}

/// Correctly rounded: a relative error of at most `f64::EPSILON / 2`, which the error bounds allow for.
impl Difference for f64 {
    fn difference(a: f64, b: f64) -> Self {
        a - b
    }
}

/// Bounds the rounded difference.
impl Difference for Magnitude {
    fn difference(a: f64, b: f64) -> Self {
        Magnitude((a - b).abs())
    }
}

/// Exact: the rounded difference and its rounding error.
impl Difference for DoubleDouble {
    fn difference(a: f64, b: f64) -> Self {
        DoubleDouble::two_sum(a, -b)
    }
}

fn dot_r<T: Ring>(a: &[T; 3], b: &[T; 3]) -> T {
    a[0].clone() * b[0].clone() + a[1].clone() * b[1].clone() + a[2].clone() * b[2].clone()
}

fn sub_r<T: Ring>(a: &[T; 3], b: &[T; 3]) -> [T; 3] {
    [
        a[0].clone() - b[0].clone(),
        a[1].clone() - b[1].clone(),
        a[2].clone() - b[2].clone(),
    ]
}

/// The smallest circumsphere of the simplex with vertices `p[0], ..., p[k]` (`k <= 3`), in the form needed for both
/// its squared radius and the in-sphere test, without dividing.
///
/// With `v_i = p_i - p_0`, its centre is `p_0 + sum_i lambda_i v_i` where `G lambda = B / 2` for the Gram matrix
/// `G_ij = v_i . v_j` and `B_i = |v_i|^2`; by Cramer's rule `lambda = adj(G) B / (2 det G)`. Returns `v`,
/// `adj(G) B` and `det G`.
fn sphere_terms<T: Ring>(p: &[[T; 3]]) -> ([[T; 3]; 3], [T; 3], T) {
    let k = p.len() - 1;
    let zero3 = || [T::zero(), T::zero(), T::zero()];
    let mut v = [zero3(), zero3(), zero3()];
    for i in 0..k {
        v[i] = sub_r(&p[i + 1], &p[0]);
    }
    let g = |i: usize, j: usize| dot_r(&v[i], &v[j]);
    let b: Vec<T> = (0..k).map(|i| g(i, i)).collect();
    let (weights, det) = match k {
        0 => (zero3(), T::zero()),
        1 => ([b[0].clone(), T::zero(), T::zero()], g(0, 0)),
        2 => (
            [
                g(1, 1) * b[0].clone() - g(0, 1) * b[1].clone(),
                g(0, 0) * b[1].clone() - g(1, 0) * b[0].clone(),
                T::zero(),
            ],
            g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0),
        ),
        _ => {
            // The adjugate of the (symmetric) Gram matrix
            let a = [
                [
                    g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1),
                    g(0, 2) * g(2, 1) - g(0, 1) * g(2, 2),
                    g(0, 1) * g(1, 2) - g(0, 2) * g(1, 1),
                ],
                [
                    g(1, 2) * g(2, 0) - g(1, 0) * g(2, 2),
                    g(0, 0) * g(2, 2) - g(0, 2) * g(2, 0),
                    g(0, 2) * g(1, 0) - g(0, 0) * g(1, 2),
                ],
                [
                    g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0),
                    g(0, 1) * g(2, 0) - g(0, 0) * g(2, 1),
                    g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0),
                ],
            ];
            let det = g(0, 0) * a[0][0].clone() + g(0, 1) * a[1][0].clone() + g(0, 2) * a[2][0].clone();
            let row = |r: usize| {
                a[r][0].clone() * b[0].clone() + a[r][1].clone() * b[1].clone() + a[r][2].clone() * b[2].clone()
            };
            ([row(0), row(1), row(2)], det)
        }
    };
    (v, weights, det)
}

/// The two sides of the in-sphere test: `q` is strictly inside the smallest circumsphere of `p` exactly when the
/// first is larger. (With `w = q - p_0`: `|w - c|^2 < |c|^2`, that is `w . c > |w|^2 / 2`, that is, multiplying by
/// `2 det G > 0`, `sum_i (adj(G) B)_i (w . v_i) > det G |w|^2`.)
fn insphere_sides<T: Ring>(p: &[[T; 3]], q: &[T; 3]) -> (T, T) {
    let k = p.len() - 1;
    let (v, weights, det) = sphere_terms(p);
    let w = sub_r(q, &p[0]);
    let mut lhs = T::zero();
    for i in 0..k {
        lhs = lhs + weights[i].clone() * dot_r(&w, &v[i]);
    }
    (lhs, det * dot_r(&w, &w))
}

/// `x` as an integer times `2^exponent`, with the integer an `i64` (exactly).
fn split_float(x: f64) -> (i64, i32) {
    if x == 0.0 {
        return (0, 0);
    }
    let bits = x.to_bits();
    let sign = if bits >> 63 == 1 { -1 } else { 1 };
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = (bits & ((1 << 52) - 1)) as i64;
    let (mantissa, exponent) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    };
    (sign * mantissa, exponent)
}

/// The smallest circumsphere of a simplex (its vertices' coordinates, padded with zeros to three).
struct Circumsphere {
    points: Vec<[f64; 3]>,
}

impl Circumsphere {
    /// The vertices are put in a canonical order (by coordinates), so that the rounding of what is computed from them
    /// doesn't depend on how the points are numbered.
    fn new(points: &[&[f64]]) -> Self {
        let pad = |p: &[f64]| {
            let mut out = [0.0; 3];
            out[..p.len()].copy_from_slice(p);
            out
        };
        let mut points: Vec<[f64; 3]> = points.iter().map(|p| pad(p)).collect();
        points.sort_by(|a, b| {
            a[0].total_cmp(&b[0])
                .then(a[1].total_cmp(&b[1]))
                .then(a[2].total_cmp(&b[2]))
        });
        Self { points }
    }

    /// The squared radius exactly, as `(numerator, denominator, least)`: it is `numerator / denominator * 4^least`
    /// (with the denominator positive), or `None` for a flat simplex.
    fn exact_squared_radius(&self) -> Option<(num_bigint::BigInt, num_bigint::BigInt, i32)> {
        let k = self.points.len() - 1;
        let (points, least) = self.exact_points(&[]);
        if k == 0 {
            return Some((num_bigint::BigInt::from(0), num_bigint::BigInt::from(1), least));
        }
        // Every coordinate is an integer times 2^least: the numerator scales by 2^(least (2k + 2)) and the
        // determinant by 2^(least 2k), so the squared radius by 4^least
        let (v, weights, det) = sphere_terms(&points);
        let num = (0..k).fold(num_bigint::BigInt::from(0), |sum, i| sum + weights[i].clone() * dot_r(&v[i], &v[i]));
        (det.sign() == num_bigint::Sign::Plus).then(|| (num, det << 2usize, least))
    }

    /// Whether the two spheres have exactly the same squared radius.
    fn same_squared_radius(&self, other: &Circumsphere) -> bool {
        match (self.exact_squared_radius(), other.exact_squared_radius()) {
            (Some((n1, d1, l1)), Some((n2, d2, l2))) => {
                let least = l1.min(l2);
                let lhs = (n1 * d2) << (2 * (l1 - least)) as usize;
                let rhs = (n2 * d1) << (2 * (l2 - least)) as usize;
                lhs == rhs
            }
            _ => false,
        }
    }

    /// The vertices relative to the first, and `q` too: subtracting two floats is correctly rounded, so these carry
    /// a relative error of at most `f64::EPSILON / 2` each, and their magnitudes bound what the formulas compute from
    /// them (far more tightly than the coordinates' own magnitudes, for a small simplex far from the origin).
    fn relative<T: Difference>(&self, q: Option<&[f64; 3]>) -> (Vec<[T; 3]>, Option<[T; 3]>) {
        let p0 = self.points[0];
        let shift = |p: &[f64; 3]| {
            [
                T::difference(p[0], p0[0]),
                T::difference(p[1], p0[1]),
                T::difference(p[2], p0[2]),
            ]
        };
        (self.points.iter().map(shift).collect(), q.map(shift))
    }

    /// The squared radius, `|c|^2 = sum_i lambda_i (c . v_i) = sum_i (adj(G) B)_i B_i / (4 det G)`, to within an
    /// ulp or two: in floating point, or double-double, when the error bound allows, else exactly (very thin
    /// simplices, whose Gram determinants cancel badly).
    fn squared_radius(&self) -> f64 {
        fn terms<T: Ring>(points: &[[T; 3]]) -> (T, T) {
            let k = points.len() - 1;
            let (v, weights, det) = sphere_terms(points);
            let num = (0..k).fold(T::zero(), |sum, i| sum + weights[i].clone() * dot_r(&v[i], &v[i]));
            (num, det)
        }
        if self.points.len() == 1 {
            return 0.0;
        }
        let (bounds, _) = self.relative::<Magnitude>(None);
        let (num_bound, det_bound) = terms(&bounds);
        let (points, _) = self.relative::<f64>(None);
        let (num, det) = terms(&points);
        if FLOAT_ERROR * num_bound.0 <= 1e-15 * num.abs() && FLOAT_ERROR * det_bound.0 <= 1e-15 * det.abs() {
            return (num / (4.0 * det)).max(0.0);
        }
        let (points, _) = self.relative::<DoubleDouble>(None);
        let (num, det) = terms(&points);
        let (num, det) = (num.to_f64(), det.to_f64());
        if DOUBLE_DOUBLE_ERROR * num_bound.0 <= 1e-17 * num.abs()
            && DOUBLE_DOUBLE_ERROR * det_bound.0 <= 1e-17 * det.abs()
        {
            return (num / (4.0 * det)).max(0.0);
        }
        let Some((num, den, least)) = self.exact_squared_radius() else {
            return 0.0;
        };
        if num.sign() == num_bigint::Sign::Minus {
            return 0.0;
        }
        // The quotient to 64 significant bits, then rounded once to a float
        let shift = (64 + den.bits() as i64 - num.bits() as i64).max(0);
        let quotient = (num << shift as usize) / den;
        let mantissa = num_traits::ToPrimitive::to_f64(&quotient).unwrap_or(f64::INFINITY);
        let scale = 2 * least as i64 - shift;
        mantissa * 2f64.powi(scale.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// The vertices and then `extra`'s coordinates as integers times a common `2^least` (exactly), and `least`.
    fn exact_points(&self, extra: &[f64]) -> (Vec<[num_bigint::BigInt; 3]>, i32) {
        let all: Vec<(i64, i32)> = self
            .points
            .iter()
            .flatten()
            .chain(extra)
            .map(|&x| split_float(x))
            .collect();
        let least = all.iter().filter(|(m, _)| *m != 0).map(|&(_, e)| e).min().unwrap_or(0);
        let integers: Vec<num_bigint::BigInt> = all
            .iter()
            .map(|&(m, e)| num_bigint::BigInt::from(m) << ((e - least).max(0) as usize))
            .collect();
        let points = integers
            .chunks(3)
            .map(|c| [c[0].clone(), c[1].clone(), c[2].clone()])
            .collect();
        (points, least)
    }

    /// Which side of the sphere `q` is on: `Greater` strictly inside, `Equal` on it, `Less` strictly outside. Decided
    /// exactly: in floating point, or double-double, when the rounding error (bounded by evaluating the same formula
    /// on magnitudes) can't change the answer, else with big integers.
    fn side(&self, q: &[f64]) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let mut q3 = [0.0; 3];
        q3[..q.len()].copy_from_slice(q);
        let (bounds, q_bound) = self.relative::<Magnitude>(Some(&q3));
        let (lhs_bound, rhs_bound) = insphere_sides(&bounds, &q_bound.unwrap());
        let magnitude = lhs_bound.0 + rhs_bound.0;
        let (points, q_rel) = self.relative::<f64>(Some(&q3));
        let (lhs, rhs) = insphere_sides(&points, &q_rel.unwrap());
        if (lhs - rhs).abs() > FLOAT_ERROR * magnitude {
            return if lhs > rhs { Ordering::Greater } else { Ordering::Less };
        }
        let (points, q_rel) = self.relative::<DoubleDouble>(Some(&q3));
        let (lhs, rhs) = insphere_sides(&points, &q_rel.unwrap());
        let difference = (lhs - rhs).to_f64();
        if difference.abs() > DOUBLE_DOUBLE_ERROR * magnitude {
            return if difference > 0.0 { Ordering::Greater } else { Ordering::Less };
        }
        // Exactly: every coordinate as an integer, times a common power of two (which scales both sides alike)
        let (mut points, _) = self.exact_points(&q3);
        let q = points.pop().unwrap();
        let (lhs, rhs) = insphere_sides(&points, &q);
        lhs.cmp(&rhs)
    }

    /// Whether `q` is strictly inside the sphere (exactly).
    fn contains(&self, q: &[f64]) -> bool {
        self.side(q) == std::cmp::Ordering::Greater
    }

    /// Whether `q` is inside the sphere or on it (exactly).
    fn encloses(&self, q: &[f64]) -> bool {
        self.side(q) != std::cmp::Ordering::Less
    }
}

// ====== Filtrations of the Delaunay triangulation ===========================

/// A filtration of the Delaunay triangulation, by the squared radius of a ball for each simplex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelaunayFiltration {
    /// The alpha filtration: a simplex appears when the balls about its vertices, each cut down to its Voronoi cell,
    /// first meet (the radius of its smallest empty circumsphere, see [`alpha_complex`]).
    Alpha,
    /// The Delaunay–Čech filtration: a simplex appears when the balls about its vertices first meet (the radius of
    /// its minimal enclosing ball), as GUDHI's `DelaunayCechComplex`.
    DelaunayCech,
}

/// A filtration of the Delaunay triangulation of a set of points (alpha or Delaunay–Čech): every simplex, by
/// dimension, with the squared radius at which it appears.
#[derive(Debug, Clone)]
pub struct AlphaComplex {
    /// The simplices of each dimension, sorted (by vertices).
    pub simplices: Vec<Vec<Simplex>>,
    /// The squared radius at which each simplex appears, matching `simplices`.
    pub values: Vec<Vec<f64>>,
    /// The facets of each simplex of dimension `d >= 1`, as indices into `simplices[d - 1]`.
    pub facets: Vec<Vec<[u32; 4]>>,
    /// For each input point, the point it was merged into (itself, unless it repeats an earlier point).
    pub vertex_of: Vec<u32>,
    /// The index of the first cell of each dimension, when the cells are numbered by dimension (and the number of
    /// cells, last).
    offsets: Vec<usize>,
}

/// The alpha filtration of `points` (each a slice of 2 or 3 coordinates).
pub fn alpha_complex(points: &[Vec<f64>]) -> Result<AlphaComplex, CloudError> {
    filtered_delaunay(points, DelaunayFiltration::Alpha)
}

/// The Delaunay–Čech filtration of `points` (each a slice of 2 or 3 coordinates): each simplex of the Delaunay
/// triangulation at the squared radius of its minimal enclosing ball.
pub fn delaunay_cech_complex(points: &[Vec<f64>]) -> Result<AlphaComplex, CloudError> {
    filtered_delaunay(points, DelaunayFiltration::DelaunayCech)
}

/// The Delaunay triangulation of `points`, filtered by `filtration`.
pub fn filtered_delaunay(
    points: &[Vec<f64>],
    filtration: DelaunayFiltration,
) -> Result<AlphaComplex, CloudError> {
    // Coordinates negligible next to the largest are zero, for the triangulation and the values alike
    let triangulation = delaunay(points)?;
    let flushed = flush(points);
    let points = flushed.as_slice();
    let top = triangulation.dimension;
    // Every face of every top simplex, by dimension
    let mut simplices: Vec<Vec<Simplex>> = vec![vec![]; top + 1];
    simplices[top] = triangulation.simplices;
    simplices[top].sort_unstable();
    simplices[top].dedup();
    for dim in (1..=top).rev() {
        let k = dim + 1;
        let mut faces: Vec<Simplex> = simplices[dim]
            .iter()
            .flat_map(|s| (0..k).map(move |i| facet(s, k, i)))
            .collect();
        faces.sort_unstable();
        faces.dedup();
        simplices[dim - 1] = faces;
    }
    // Facets as indices
    let mut facets: Vec<Vec<[u32; 4]>> = vec![vec![]; top + 1];
    for dim in 1..=top {
        let k = dim + 1;
        facets[dim] = simplices[dim]
            .iter()
            .map(|s| {
                let mut out = [PAD; 4];
                for (i, slot) in out.iter_mut().enumerate().take(k) {
                    let f = facet(s, k, i);
                    *slot = simplices[dim - 1].binary_search(&f).expect("facet listed") as u32;
                }
                out
            })
            .collect();
    }
    let values = match filtration {
        DelaunayFiltration::Alpha => alpha_values(points, &simplices, &facets),
        DelaunayFiltration::DelaunayCech => cech_values(points, &simplices, &facets),
    };
    let mut offsets = vec![0];
    for dim in &simplices {
        offsets.push(offsets[offsets.len() - 1] + dim.len());
    }
    Ok(AlphaComplex {
        simplices,
        values,
        facets,
        vertex_of: triangulation.vertex_of,
        offsets,
    })
}

/// The smallest circumsphere of a simplex of `points`.
fn sphere_of(points: &[Vec<f64>], simplex: &Simplex) -> Circumsphere {
    let vs: Vec<&[f64]> = vertices(simplex).iter().map(|&v| points[v as usize].as_slice()).collect();
    Circumsphere::new(&vs)
}

/// Alpha values, from the top dimension down.
fn alpha_values(points: &[Vec<f64>], simplices: &[Vec<Simplex>], facets: &[Vec<[u32; 4]>]) -> Vec<Vec<f64>> {
    let top = simplices.len() - 1;
    let coords = |v: u32| points[v as usize].as_slice();
    let sphere = |s: &Simplex| sphere_of(points, s);
    let mut values: Vec<Vec<f64>> = vec![vec![]; top + 1];
    // Every value is the squared radius of some simplex's smallest circumsphere: which (dimension and index), so that
    // a tie between a simplex's own radius and that inherited from a cofacet can be decided exactly
    let mut sources: Vec<Vec<(u8, u32)>> = vec![vec![]; top + 1];
    values[top] = simplices[top]
        .iter()
        .map(|s| if top == 0 { 0.0 } else { sphere(s).squared_radius() })
        .collect();
    sources[top] = (0..simplices[top].len() as u32).map(|i| (top as u8, i)).collect();
    for dim in (0..top).rev() {
        let n = simplices[dim].len();
        if dim == 0 {
            values[0] = vec![0.0; n];
            break;
        }
        let spheres: Vec<Circumsphere> = simplices[dim].iter().map(sphere).collect();
        let mut gabriel = vec![true; n];
        // The first cofacet to appear: its value and its value's source
        let mut first_cofacet = vec![(f64::INFINITY, (0u8, 0u32)); n];
        let k = dim + 2;
        for (cofacet, s) in simplices[dim + 1].iter().enumerate() {
            let value = values[dim + 1][cofacet];
            for i in 0..k {
                let f = facets[dim + 1][cofacet][i] as usize;
                if value < first_cofacet[f].0 {
                    first_cofacet[f] = (value, sources[dim + 1][cofacet]);
                }
                // The vertex of the cofacet opposite this facet
                if gabriel[f] && spheres[f].contains(coords(s[i])) {
                    gabriel[f] = false;
                }
            }
        }
        let mut dim_values = Vec::with_capacity(n);
        let mut dim_sources = Vec::with_capacity(n);
        for f in 0..n {
            let (inherited, inherited_source) = first_cofacet[f];
            let (value, source) = if !gabriel[f] {
                (inherited, inherited_source)
            } else {
                // Its own radius, which is never more than its cofacets' (exactly): if rounding makes the two
                // indistinguishable, decide exactly whether they are equal, so that a zero-length bar stays zero
                let own = spheres[f].squared_radius();
                let close = (own - inherited).abs() <= 1e-12 * inherited.abs();
                let (source_dim, source_index) = inherited_source;
                if inherited.is_finite()
                    && (own >= inherited
                        || close
                            && spheres[f].same_squared_radius(&sphere(
                                &simplices[source_dim as usize][source_index as usize],
                            )))
                {
                    (inherited, inherited_source)
                } else {
                    (own, (dim as u8, f as u32))
                }
            };
            dim_values.push(value);
            dim_sources.push(source);
        }
        values[dim] = dim_values;
        sources[dim] = dim_sources;
    }
    values
}

/// Delaunay–Čech values: the squared radius of each simplex's minimal enclosing ball. For at most four points, that
/// is the smallest circumsphere of some subset of them which encloses them all (exactly; and of all such, the
/// smallest).
fn cech_values(points: &[Vec<f64>], simplices: &[Vec<Simplex>], facets: &[Vec<[u32; 4]>]) -> Vec<Vec<f64>> {
    let top = simplices.len() - 1;
    let coords = |v: u32| points[v as usize].as_slice();
    let mut values: Vec<Vec<f64>> = simplices
        .iter()
        .enumerate()
        .map(|(dim, simplices)| {
            if dim == 0 {
                return vec![0.0; simplices.len()];
            }
            simplices
                .iter()
                .map(|s| {
                    let vs = vertices(s);
                    let k = vs.len();
                    (1u32..(1 << k))
                        .filter(|subset| subset.count_ones() >= 2)
                        .filter_map(|subset| {
                            let chosen: Vec<&[f64]> =
                                (0..k).filter(|i| subset >> i & 1 == 1).map(|i| coords(vs[i])).collect();
                            let ball = Circumsphere::new(&chosen);
                            (0..k)
                                .filter(|i| subset >> i & 1 == 0)
                                .all(|i| ball.encloses(coords(vs[i])))
                                .then(|| ball.squared_radius())
                        })
                        .fold(f64::INFINITY, f64::min)
                })
                .collect()
        })
        .collect();
    // Exactly, a face's ball is never larger than its cofacets'; keep it so through rounding
    for dim in (1..=top).rev() {
        for (cofacet, f) in facets[dim].iter().enumerate() {
            let value = values[dim][cofacet];
            for &face in &f[..dim + 1] {
                let face_value = &mut values[dim - 1][face as usize];
                *face_value = face_value.min(value);
            }
        }
    }
    values
}

impl AlphaComplex {
    /// The dimension of the complex (0 to 3).
    pub fn dimension(&self) -> usize {
        self.simplices.len() - 1
    }

    /// The number of simplices.
    pub fn n_simplices(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    /// The dimension and simplex of a cell of [`filtered`](Self::filtered)'s complex.
    #[inline]
    pub fn cell(&self, cell: Cell) -> (usize, &Simplex) {
        let index = cell.0 as usize;
        let dim = self.offsets.partition_point(|&o| o <= index) - 1;
        (dim, &self.simplices[dim][index - self.offsets[dim]])
    }

    /// The filtered complex over Z/2, with values the radii (or with `squared`, the squared radii), and cells
    /// numbered by dimension (so that a face enters before a coface with the same value).
    pub fn filtered(&self, squared: bool) -> Complex<Z2> {
        let offsets = &self.offsets;
        let mut raw = RawChain {
            indptr: vec![0],
            ..RawChain::default()
        };
        let mut values = Vec::with_capacity(self.n_simplices());
        for (dim, simplices) in self.simplices.iter().enumerate() {
            for (i, _) in simplices.iter().enumerate() {
                raw.dimensions.push(dim as i64);
                if dim > 0 {
                    raw.indices.extend(
                        self.facets[dim][i][..dim + 1]
                            .iter()
                            .map(|&f| (offsets[dim - 1] + f as usize) as i64),
                    );
                }
                raw.indptr.push(raw.indices.len() as i64);
                let value = self.values[dim][i];
                values.push(if squared { value } else { value.sqrt() });
            }
        }
        Chain::<Z2>::from_raw(raw)
            .and_then(|chain| chain.filter(Some(values)))
            .expect("the alpha complex is a valid filtered complex")
    }
}

/// A bar in the alpha barcode, with cells of [`AlphaComplex::filtered`]'s complex.
pub type AlphaBar = Bar<Cell, NotNan<f64>, Z2>;

/// The alpha barcode of `points` (each a slice of 2 or 3 coordinates) in dimensions `0..=max_dim` (by default, all
/// below the complex's own dimension, the only ones that can be non-zero), with values the radii (or with `squared`,
/// the squared radii), together with the complex (to turn the bars' cells into simplices, see
/// [`AlphaComplex::cell`]).
///
/// Zero-length bars are omitted, and bars are returned in no particular order.
pub fn alpha_barcode<R: ColumnRepresentation>(
    points: &[Vec<f64>],
    max_dim: Option<usize>,
    squared: bool,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Result<(AlphaComplex, Vec<AlphaBar>), CloudError> {
    delaunay_barcode(
        points,
        DelaunayFiltration::Alpha,
        max_dim,
        squared,
        method,
        algorithm,
        representatives,
    )
}

/// As [`alpha_barcode`], for either filtration of the Delaunay triangulation.
#[allow(clippy::too_many_arguments)]
pub fn delaunay_barcode<R: ColumnRepresentation>(
    points: &[Vec<f64>],
    filtration: DelaunayFiltration,
    max_dim: Option<usize>,
    squared: bool,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Result<(AlphaComplex, Vec<AlphaBar>), CloudError> {
    let complex = filtered_delaunay(points, filtration)?;
    let max_dim = max_dim.unwrap_or(complex.dimension().saturating_sub(1));
    let filtered = complex.filtered(squared);
    let bars = if max_dim > complex.dimension() {
        // Nothing above the complex's dimension
        filtered.barcode(None, method, algorithm, representatives)
    } else {
        filtered.barcode(Some(max_dim), method, algorithm, representatives)
    };
    Ok((complex, bars))
}

#[cfg(test)]
mod tests;
