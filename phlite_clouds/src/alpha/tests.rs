use super::*;
use phlite::{columns::PivotColumns, reduction::Clearing};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn random_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..n)
        .map(|_| (0..dim).map(|_| lcg(&mut state)).collect())
        .collect()
}

#[test]
fn circumspheres() {
    let (a, b, c) = (vec![0.0, 0.0], vec![2.0, 0.0], vec![0.0, 2.0]);
    let triangle = Circumsphere::new(&[&a, &b, &c]);
    assert_eq!(triangle.squared_radius(), 2.0);
    assert!(triangle.contains(&[1.0, 1.0]));
    assert!(!triangle.contains(&[2.0, 2.0]), "on the circle is not inside");
    let edge = Circumsphere::new(&[&a, &b]);
    assert_eq!(edge.squared_radius(), 1.0);
    assert!(edge.contains(&[1.0, 0.5]));
    assert!(!edge.contains(&[1.0, 1.0]));
    // A regular tetrahedron's corners of a cube: circumradius^2 = 3/4 of the side^2
    let p: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
    ];
    let refs: Vec<&[f64]> = p.iter().map(Vec::as_slice).collect();
    let tet = Circumsphere::new(&refs);
    assert!((tet.squared_radius() - 0.75).abs() < 1e-12);
    assert!(tet.contains(&[0.5, 0.5, 0.5]));
    assert!(!tet.contains(&[1.0, 1.0, 1.0]), "a corner of the cube is on the sphere");
    // A triangle in space: the smallest circumsphere is centred in its plane
    let refs: Vec<&[f64]> = p[..3].iter().map(Vec::as_slice).collect();
    let triangle = Circumsphere::new(&refs);
    assert!((triangle.squared_radius() - 2.0 / 3.0).abs() < 1e-12);
}

/// A thin triangle, whose Gram determinant cancels badly: its squared circumradius must still be accurate (here to
/// GUDHI's exact value, 23.765625409713838; the floating point formula gave 23.7656256503)
#[test]
fn squared_radii_of_thin_simplices_are_accurate() {
    let (a, b, q) = (vec![9.75, 0.0], vec![0.0, 0.0011819995706900954], vec![-5.960464477539063e-08, 0.0]);
    let r2 = Circumsphere::new(&[&a, &q, &b]).squared_radius();
    assert!((r2 - 23.765625409713838).abs() <= 4.0 * f64::EPSILON * r2, "{r2}");
    // Every angle is acute, so no vertex is inside the opposite edge's smallest circle
    assert!(!Circumsphere::new(&[&a, &b]).contains(&q));
    // A sliver tetrahedron (four nearly cocircular points), whose Gram determinant (1e-18) vanishes in floating
    // point: its centre is (1/2, 1/2, 1e-9 / 2), so r^2 = 1/2 + 1e-18 / 4, which rounds to 1/2
    let flat: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1e-9],
    ];
    let refs: Vec<&[f64]> = flat.iter().map(Vec::as_slice).collect();
    assert_eq!(Circumsphere::new(&refs).squared_radius(), 0.5);
}

#[test]
fn in_sphere_tests_are_exact() {
    // On the circle is not inside, exactly
    let (a, b) = (vec![0.1, 0.0], vec![0.0, 0.1]);
    let sphere = Circumsphere::new(&[&a, &b]);
    assert!(!sphere.contains(&[0.0, 0.0]) && !sphere.contains(&[0.1, 0.1]));
    assert!(sphere.contains(&[0.05, 0.05]));
}

#[test]
fn triangulations_of_degenerate_points() {
    // A cubical grid: every cube's corners are cospherical, and each cube splits into 6 tetrahedra
    let grid: Vec<Vec<f64>> = (0..5)
        .flat_map(|x| {
            (0..5).flat_map(move |y| (0..5).map(move |z| vec![x as f64, y as f64, z as f64]))
        })
        .collect();
    let d = delaunay(&grid).unwrap();
    assert_eq!((d.dimension, d.simplices.len()), (3, 4 * 4 * 4 * 6));
    // Coplanar points in space, a square grid: triangulated in their plane. (Coordinates in the plane are rounded,
    // so the boundary's collinear points may gain slivers, which only appear together with their long edge.)
    let plane: Vec<Vec<f64>> = (0..4)
        .flat_map(|x| (0..4).map(move |y| vec![x as f64, y as f64, x as f64 + y as f64]))
        .collect();
    let d = delaunay(&plane).unwrap();
    assert_eq!(d.dimension, 2);
    assert!(d.simplices.len() >= 3 * 3 * 2);
    // The same barcode as the same points given in the plane (one loop per square of the grid)
    let summary = |points: &[Vec<f64>]| {
        let algorithm = Algorithm::Clearing(Clearing::new().with_columns(PivotColumns));
        let (complex, bars) =
            alpha_barcode(points, None, true, Method::Cohomology, algorithm, false).unwrap();
        let mut out: Vec<(usize, f64, f64)> = bars
            .iter()
            .map(|bar| {
                let death = bar.death.map_or(f64::INFINITY, |(_, d)| d.into_inner());
                (complex.cell(bar.birth).0, bar.birth_value.into_inner(), death)
            })
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    };
    let flat: Vec<Vec<f64>> = (0..4)
        .flat_map(|x| (0..4).map(move |y| vec![x as f64, 2.0 * y as f64]))
        .collect();
    let in_space: Vec<Vec<f64>> = flat.iter().map(|p| vec![p[0], p[1], 0.0]).collect();
    let (a, b) = (summary(&flat), summary(&in_space));
    assert_eq!(a.iter().filter(|bar| bar.0 == 1).count(), 9);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.0, y.0);
        assert!((x.1 - y.1).abs() < 1e-9 && (x.2 == y.2 || (x.2 - y.2).abs() < 1e-9), "{x:?} {y:?}");
    }
    // Collinear points, in space and in the plane, out of order: a path
    for points in [
        vec![vec![0.0, 0.0, 0.0], vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 1.0]],
        vec![vec![3.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0], vec![0.0, 0.0]],
    ] {
        let d = delaunay(&points).unwrap();
        assert_eq!(d.dimension, 1);
        assert_eq!(d.simplices.len(), points.len() - 1);
    }
    // Repeated points are merged into the first
    let repeated = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![-0.0, 0.0], vec![0.0, 0.0]];
    let d = delaunay(&repeated).unwrap();
    assert_eq!(d.vertex_of, vec![0, 0, 2, 2]);
    assert_eq!((d.dimension, d.simplices.len()), (1, 1));
    // Tiny (even subnormal) and huge coordinates are scaled by a power of two, and those negligible next to the
    // largest flushed to zero, so the triangulators accept them
    for factor in [1e-300, 1e-41, 1e200] {
        let scaled: Vec<Vec<f64>> = grid.iter().map(|p| p.iter().map(|x| x * factor).collect()).collect();
        let d = delaunay(&scaled).unwrap();
        assert_eq!((d.dimension, d.simplices.len()), (3, 4 * 4 * 4 * 6), "{factor}");
    }
    let d = delaunay(&[vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 3.6e-41], vec![1.0, 0.0, 0.0]]).unwrap();
    assert_eq!(d.vertex_of, vec![0, 0, 2], "a point negligibly far from another is merged into it");
    let single = delaunay(&[vec![1.0, 2.0, 3.0]]).unwrap();
    assert_eq!((single.dimension, single.simplices.len()), (0, 1));
    assert!(delaunay(&[vec![1.0]]).is_err());
    assert!(delaunay(&[vec![0.0, f64::NAN]]).is_err());
}

#[test]
fn values_are_a_filtration() {
    for (points, name) in [
        (random_points(300, 2, 1), "random plane"),
        (random_points(300, 3, 2), "random space"),
        (
            (0..4)
                .flat_map(|x| {
                    (0..4).flat_map(move |y| (0..4).map(move |z| vec![x as f64, y as f64, z as f64]))
                })
                .collect(),
            "grid",
        ),
    ] {
        let complex = alpha_complex(&points).unwrap();
        for dim in 1..=complex.dimension() {
            for (i, f) in complex.facets[dim].iter().enumerate() {
                for &face in &f[..dim + 1] {
                    assert!(
                        complex.values[dim - 1][face as usize] <= complex.values[dim][i],
                        "{name}: a face after its coface"
                    );
                }
            }
        }
        // The Delaunay triangulation of points in general position is contractible: Euler characteristic 1
        let euler: i64 = complex
            .simplices
            .iter()
            .enumerate()
            .map(|(d, s)| if d % 2 == 0 { s.len() as i64 } else { -(s.len() as i64) })
            .sum();
        assert_eq!(euler, 1, "{name}");
    }
}

#[test]
fn unit_square() {
    // Edges of length 1 appear at radius 1/2; the loop is filled by the two triangles at the circumradius sqrt(1/2)
    let square = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0]];
    let algorithm = Algorithm::Clearing(Clearing::new().with_columns(PivotColumns));
    let (complex, bars) =
        alpha_barcode(&square, None, false, Method::Cohomology, algorithm, true).unwrap();
    let mut h1: Vec<_> = bars
        .iter()
        .filter(|bar| complex.cell(bar.birth).0 == 1)
        .map(|bar| (bar.birth_value.into_inner(), bar.death.map(|(_, d)| d.into_inner())))
        .collect();
    h1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(h1.len(), 1);
    assert_eq!(h1[0].0, 0.5);
    assert!((h1[0].1.unwrap() - 0.5f64.sqrt()).abs() < 1e-12);
    // Three finite H0 bars dying at 1/2, and one essential
    let h0: Vec<_> = bars.iter().filter(|bar| complex.cell(bar.birth).0 == 0).collect();
    assert_eq!(h0.len(), 4);
    assert_eq!(h0.iter().filter(|bar| bar.death.is_none()).count(), 1);
}

/// The values in the plane agree with alphalpha's (which computes them its own way, from spade's triangulation)
#[test]
fn plane_values_match_alphalpha() {
    use spade::{Point2, Triangulation};
    for seed in 0..5 {
        let points = random_points(200, 2, seed);
        let complex = alpha_complex(&points).unwrap();
        let filtration =
            alphalpha::alpha_filtration(points.iter().map(|p| Point2::new(p[0], p[1])).collect());
        let index = |v: spade::handles::FixedVertexHandle| {
            let p = filtration.vertex(v).position();
            points.iter().position(|q| q[0] == p.x && q[1] == p.y).unwrap() as u32
        };
        let mut theirs: Vec<(Simplex, f64)> = vec![];
        for e in filtration.undirected_edges() {
            let mut vs = e.vertices().map(|v| index(v.fix()));
            vs.sort_unstable();
            theirs.push(([vs[0], vs[1], PAD, PAD], e.data().filtration));
        }
        for f in filtration.inner_faces() {
            let mut vs = f.vertices().map(|v| index(v.fix()));
            vs.sort_unstable();
            theirs.push(([vs[0], vs[1], vs[2], PAD], f.data().filtration));
        }
        theirs.sort_by(|a, b| a.0.cmp(&b.0));
        let ours: Vec<(Simplex, f64)> = (1..=2)
            .flat_map(|d| {
                complex.simplices[d]
                    .iter()
                    .copied()
                    .zip(complex.values[d].iter().copied())
            })
            .collect();
        let mut ours = ours;
        ours.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(ours.len(), theirs.len());
        for ((s, a), (t, b)) in ours.iter().zip(&theirs) {
            assert_eq!(s, t);
            assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{s:?}: {a} vs {b}");
        }
    }
}
