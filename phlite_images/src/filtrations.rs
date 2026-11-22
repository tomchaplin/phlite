//! Filtrations of binary images, as giotto-tda's `gtda.images` filtrations: each turns a binary image (which pixels
//! are *active*) into a greyscale one, whose sublevel sets [`crate::cubical`] then filters by.
//!
//! Images are flat, in row-major (C) order, with their `shape` (any number of axes); coordinates are in the axes'
//! order. Pixels a filtration doesn't reach are `f64::INFINITY`, so that they never enter; [`unreached_last`] gives
//! them one more than the largest value instead, so that they enter last, as giotto-tda does.

/// A metric on pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

impl Metric {
    /// The length of `delta` (absolute coordinate differences).
    fn length(self, delta: impl Iterator<Item = f64>) -> f64 {
        match self {
            Metric::Euclidean => delta.map(|x| x * x).sum::<f64>().sqrt(),
            Metric::Manhattan => delta.sum(),
            Metric::Chebyshev => delta.fold(0.0, f64::max),
        }
    }
}

/// What can be wrong with a filtration's parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum FiltrationError {
    /// The image doesn't have `product(shape)` pixels
    Shape,
    /// A direction or centre doesn't have one component per axis
    Components,
    /// A direction with every component zero
    ZeroDirection,
}

impl std::fmt::Display for FiltrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape => write!(f, "the image doesn't have as many pixels as its shape says"),
            Self::Components => write!(f, "give one component per axis"),
            Self::ZeroDirection => write!(f, "the direction must not be all zero"),
        }
    }
}

impl std::error::Error for FiltrationError {}

fn check(image: &[bool], shape: &[usize]) -> Result<(), FiltrationError> {
    if image.len() == shape.iter().product::<usize>() {
        Ok(())
    } else {
        Err(FiltrationError::Shape)
    }
}

/// The coordinates of pixel `index`.
fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for (axis, &n) in shape.iter().enumerate().rev() {
        out[axis] = index % n;
        index /= n;
    }
    out
}

/// The height filtration: each active pixel at its height along `direction` (its coordinates' dot product with the
/// unit vector, measured from the corner the direction points away from, so heights start at 0), as giotto-tda's
/// `HeightFiltration`; inactive pixels unreached.
///
/// # Errors
/// If the shape or the direction don't fit, or the direction is zero.
pub fn height(image: &[bool], shape: &[usize], direction: &[f64]) -> Result<Vec<f64>, FiltrationError> {
    check(image, shape)?;
    if direction.len() != shape.len() {
        return Err(FiltrationError::Components);
    }
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err(FiltrationError::ZeroDirection);
    }
    let unit: Vec<f64> = direction.iter().map(|x| x / norm).collect();
    Ok((0..image.len())
        .map(|i| {
            if !image[i] {
                return f64::INFINITY;
            }
            coordinates(i, shape)
                .iter()
                .zip(shape)
                .zip(&unit)
                .map(|((&c, &n), &u)| {
                    let c = if u < 0.0 { (n - 1 - c) as f64 } else { c as f64 };
                    c * u.abs()
                })
                .sum()
        })
        .collect())
}

/// The radial filtration: each active pixel within `radius` of `center` at its distance from it in `metric`, as
/// giotto-tda's `RadialFiltration`; other pixels unreached.
///
/// # Errors
/// If the shape or the centre don't fit.
pub fn radial(
    image: &[bool],
    shape: &[usize],
    center: &[f64],
    radius: f64,
    metric: Metric,
) -> Result<Vec<f64>, FiltrationError> {
    check(image, shape)?;
    if center.len() != shape.len() {
        return Err(FiltrationError::Components);
    }
    Ok((0..image.len())
        .map(|i| {
            let d = metric.length(coordinates(i, shape).iter().zip(center).map(|(&c, &x)| (c as f64 - x).abs()));
            if image[i] && d <= radius {
                d
            } else {
                f64::INFINITY
            }
        })
        .collect())
}

/// The Manhattan (L1) distance from every pixel to the nearest source pixel (`f64::INFINITY` if there is none):
/// exactly, as the L1 distance transform separates into one-dimensional ones, a forward and a backward pass along
/// each axis.
///
/// # Panics
/// If `sources` doesn't have `product(shape)` pixels.
pub fn manhattan_distance(sources: &[bool], shape: &[usize]) -> Vec<f64> {
    assert_eq!(sources.len(), shape.iter().product::<usize>(), "a pixel for every position");
    let mut out: Vec<f64> = sources.iter().map(|&s| if s { 0.0 } else { f64::INFINITY }).collect();
    let mut stride = 1;
    for &n in shape.iter().rev() {
        // Lines along this axis: their starts are the indices whose coordinate on the axis is 0
        let block = stride * n;
        for start in (0..out.len()).filter(|i| (i % block) < stride) {
            for k in 1..n {
                let (here, before) = (start + k * stride, start + (k - 1) * stride);
                out[here] = out[here].min(out[before] + 1.0);
            }
            for k in (0..n.saturating_sub(1)).rev() {
                let (here, after) = (start + k * stride, start + (k + 1) * stride);
                out[here] = out[here].min(out[after] + 1.0);
            }
        }
        stride = block;
    }
    out
}

fn cap(values: Vec<f64>, n_iterations: Option<usize>) -> Vec<f64> {
    match n_iterations {
        None => values,
        Some(n) => values.into_iter().map(|v| if v <= n as f64 { v } else { f64::INFINITY }).collect(),
    }
}

/// The dilation filtration: active pixels at 0, every other pixel at its Manhattan distance to the active ones (the
/// number of 4-neighbour dilations that reach it), up to `n_iterations`, as giotto-tda's `DilationFiltration`.
pub fn dilation(image: &[bool], shape: &[usize], n_iterations: Option<usize>) -> Vec<f64> {
    cap(manhattan_distance(image, shape), n_iterations)
}

/// The erosion filtration: inactive pixels at 0, every active pixel at its Manhattan distance to the inactive ones,
/// up to `n_iterations`, as giotto-tda's `ErosionFiltration`.
pub fn erosion(image: &[bool], shape: &[usize], n_iterations: Option<usize>) -> Vec<f64> {
    let inactive: Vec<bool> = image.iter().map(|&a| !a).collect();
    cap(manhattan_distance(&inactive, shape), n_iterations)
}

/// The signed distance filtration, as giotto-tda's `SignedDistanceFiltration`: an active pixel at its Manhattan
/// distance to the inactive ones minus 1 (0 on the boundary of an active region), an inactive pixel at minus its
/// distance to the active ones; distances beyond `n_iterations` unreached.
pub fn signed_distance(image: &[bool], shape: &[usize], n_iterations: Option<usize>) -> Vec<f64> {
    let inactive: Vec<bool> = image.iter().map(|&a| !a).collect();
    let (to_inactive, to_active) = (manhattan_distance(&inactive, shape), manhattan_distance(image, shape));
    (0..image.len())
        .map(|i| {
            let (value, distance) =
                if image[i] { (to_inactive[i] - 1.0, to_inactive[i]) } else { (-to_active[i], to_active[i]) };
            match n_iterations {
                Some(n) if distance > n as f64 => f64::INFINITY,
                _ => value,
            }
        })
        .collect()
}

/// The density filtration: every pixel at the number of active pixels within `radius` of it in `metric` (itself
/// included), as giotto-tda's `DensityFiltration`.
pub fn density(image: &[bool], shape: &[usize], radius: f64, metric: Metric) -> Vec<f64> {
    let reach = radius.floor().max(0.0) as i64;
    // The offsets within the radius
    let mut offsets: Vec<Vec<i64>> = vec![vec![]];
    for _ in shape {
        offsets = offsets
            .into_iter()
            .flat_map(|o| (-reach..=reach).map(move |x| [o.clone(), vec![x]].concat()))
            .collect();
    }
    offsets.retain(|o| metric.length(o.iter().map(|&x| x.unsigned_abs() as f64)) <= radius);
    let strides: Vec<usize> = {
        let mut s = vec![1; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            s[axis] = s[axis + 1] * shape[axis + 1];
        }
        s
    };
    (0..image.len())
        .map(|i| {
            let here = coordinates(i, shape);
            offsets
                .iter()
                .filter(|o| {
                    let mut index = 0;
                    for (axis, (&c, &x)) in here.iter().zip(o.iter()).enumerate() {
                        let c = c as i64 + x;
                        if c < 0 || c >= shape[axis] as i64 {
                            return false;
                        }
                        index += c as usize * strides[axis];
                    }
                    image[index]
                })
                .count() as f64
        })
        .collect()
}

/// Give the unreached (non-finite) pixels one more than the largest finite value (or 1 if there is none), so that they
/// enter last, as giotto-tda's `max_value_`.
pub fn unreached_last(values: &mut [f64]) {
    let last = values.iter().copied().filter(|v| v.is_finite()).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
    let last = last.unwrap_or(0.0) + 1.0;
    for v in values.iter_mut().filter(|v| !v.is_finite()) {
        *v = last;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_and_filtrations() {
        // A 3 × 4 image with one active pixel at (1, 1)
        let shape = [3, 4];
        let mut image = vec![false; 12];
        image[5] = true;
        let d = dilation(&image, &shape, None);
        assert_eq!(d, vec![2.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0]);
        assert_eq!(dilation(&image, &shape, Some(1))[3], f64::INFINITY);
        let s = signed_distance(&image, &shape, None);
        assert_eq!((s[5], s[0]), (0.0, -2.0));
        assert!(erosion(&image, &shape, None).iter().all(|&v| v <= 1.0));
        let h = height(&image, &shape, &[0.0, -1.0]).unwrap();
        assert_eq!(h[5], 2.0); // measured from the right: column 1 of 4 is 2 away
        assert_eq!(density(&image, &shape, 1.0, Metric::Manhattan)[1], 1.0);
        let mut last = d.clone();
        last[3] = f64::INFINITY;
        unreached_last(&mut last);
        assert_eq!(last[3], 4.0);
    }
}
