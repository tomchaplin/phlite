//! What can be wrong with points, distances or weights given to this crate.

use std::fmt;

/// Why points, a distance matrix or weights were rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum CloudError {
    /// No points at all, where some are needed (a triangulation).
    NoPoints,
    /// More points than can be indexed (`u32`).
    TooManyPoints { n_points: usize },
    /// A sparse distance between points `i` and `j`, one of them outside `0..n_points`.
    PointOutOfRange { i: usize, j: usize, n_points: usize },
    /// A negative or `NaN` distance (distances are non-negative, and infinite for a pair never joined).
    BadDistance { i: usize, j: usize, distance: f64 },
    /// The distance between `i` and `j` given twice, differently.
    ConflictingDistances { i: usize, j: usize, first: f64, second: f64 },
    /// A point with a coordinate that is infinite or `NaN`.
    NonFiniteCoordinate { point: usize },
    /// Points with different numbers of coordinates: `point` has `len`, the first has `expected`.
    WrongDimension { point: usize, len: usize, expected: usize },
    /// Points in a dimension the construction doesn't support (alpha and Delaunay–Čech: 2 or 3).
    UnsupportedDimension { dimension: usize },
    /// A weight per point is needed.
    WrongNumberOfWeights { weights: usize, n_points: usize },
    /// A weight that is infinite or `NaN`.
    BadWeight { point: usize, weight: f64 },
    /// The distance to measure's `k`, which must be in `1..=n_points`.
    BadK { k: usize, n_points: usize },
    /// The distance to measure's exponent, which must be positive and finite.
    BadQ { q: f64 },
    /// The triangulation library failed (on input it should accept: a bug, to report).
    Triangulation(String),
}

impl fmt::Display for CloudError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CloudError::NoPoints => write!(f, "There are no points"),
            CloudError::TooManyPoints { n_points } => write!(f, "Too many points ({n_points})"),
            CloudError::PointOutOfRange { i, j, n_points } => {
                write!(f, "Edge ({i}, {j}) is out of range for {n_points} points")
            }
            CloudError::BadDistance { i, j, distance } => write!(
                f,
                "The distance between {i} and {j} is {distance}, but distances must be non-negative (or infinite)"
            ),
            CloudError::ConflictingDistances { i, j, first, second } => {
                write!(f, "The distance between {i} and {j} is given as both {first} and {second}")
            }
            CloudError::NonFiniteCoordinate { point } => {
                write!(f, "Every coordinate must be finite, but point {point} has one that isn't")
            }
            CloudError::WrongDimension { point, len, expected } => {
                write!(f, "Every point must have {expected} coordinates, but point {point} has {len}")
            }
            CloudError::UnsupportedDimension { dimension } => write!(
                f,
                "Alpha complexes need points in the plane or in space (2 or 3 coordinates), not {dimension}"
            ),
            CloudError::WrongNumberOfWeights { weights, n_points } => {
                write!(f, "There are {weights} weights for {n_points} points")
            }
            CloudError::BadWeight { point, weight } => {
                write!(f, "Weights must be finite, but weight {point} is {weight}")
            }
            CloudError::BadK { k, n_points } => {
                write!(f, "k must be between 1 and the number of points ({n_points}), not {k}")
            }
            CloudError::BadQ { q } => write!(f, "q must be positive and finite, not {q}"),
            CloudError::Triangulation(e) => write!(f, "Could not triangulate the points: {e}"),
        }
    }
}

impl std::error::Error for CloudError {}

/// A `ValueError`, with the message above.
#[cfg(feature = "python")]
impl From<CloudError> for pyo3::PyErr {
    fn from(e: CloudError) -> Self {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}

/// Check that `points` all have the same number of coordinates, every one finite; returns that number (0 if there are
/// no points).
pub fn check_points(points: &[Vec<f64>]) -> Result<usize, CloudError> {
    if u32::try_from(points.len()).is_err() {
        return Err(CloudError::TooManyPoints { n_points: points.len() });
    }
    let expected = points.first().map_or(0, Vec::len);
    for (point, p) in points.iter().enumerate() {
        if p.len() != expected {
            return Err(CloudError::WrongDimension {
                point,
                len: p.len(),
                expected,
            });
        }
        if !p.iter().all(|x| x.is_finite()) {
            return Err(CloudError::NonFiniteCoordinate { point });
        }
    }
    Ok(expected)
}
