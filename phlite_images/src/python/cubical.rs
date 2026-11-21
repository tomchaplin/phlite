//! `phlite.images.cubical`: persistence of the cubical complex of an image (or volume), by [`crate::cubical`].

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{
    algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars,
};

use crate::cubical::{filtration_barcode, Construction, CubicalComplex, Filtration, MAX_AXES};

/// The pipeline `'images.cubical'`. Columns are recomputed from the pixels on the fly, as for Vietoris–Rips; by
/// measurement, storing R (rather than rebuilding each pivot column from V) makes reducing the middle dimensions of a
/// volume up to 3x faster, and merged columns are then as fast as any (images have no reduction left in between).
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "images.cubical",
    columns: Columns::Merged,
    cache: false,
    store_r: true,
    row_ranks: false,
    union_find: Some(true),
    collapse: None,
};

/// Compute the persistent homology of the cubical complex of an image (or a volume, or any grid with 1 to 4 axes), in
/// dimensions `0, ..., max_dim`.
///
/// Args:
///     image: The pixel values: a numpy array (or anything numpy can make one of) with 1 to 4 axes. NaN pixels are
///         left out of the complex, as are those that would enter at infinity (`+inf` for sublevel sets, `-inf` for
///         superlevel sets).
///     max_dim: The maximum homology dimension to compute (default: one less than the number of axes, the highest
///         that can be non-zero).
///     construction: `'T'` (the default, as GUDHI's `top_dimensional_cells`): pixels are the top-dimensional cells
///         (squares, cubes), and every other cell enters with the smallest of the top cells around it; or `'V'` (as
///         Cubical Ripser's default, and GUDHI's `vertices`): pixels are the vertices, and every other cell enters
///         with the largest of its vertices.
///     filtration: `'sublevel'` (the default: by increasing value) or `'superlevel'` (by decreasing value; bars are
///         then reported in the image's own values, so run downwards, from a larger birth to a smaller death).
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'images.cubical'`): columns='merged', cache=False, store_r=True, union_find=True.
///         With `union_find` (by cohomology, without representatives), dimension 0 and the top dimension are found by
///         union-find, and the T-construction of a volume (3 axes) by a filtered discrete gradient (as Perseus).
///     representatives: Whether to also compute a representative cycle for every bar.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`. Cells (births, deaths and representatives) are given by their
///     coordinates on the doubled grid, as GUDHI does, in the order of the image's axes: a coordinate is even where
///     the cell sits at a vertex along that axis and odd where it spans an interval, so in the T-construction pixel
///     `p` is the top cell `2 p + 1`, and in the V-construction the vertex `2 p`.
#[gen_stub_pyfunction(module = "phlite.images")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (image, max_dim = None, *, construction = "T", filtration = "sublevel", method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn cubical<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "numpy.typing.ArrayLike", imports = ("numpy.typing")))]
    image: &Bound<'_, PyAny>,
    max_dim: Option<usize>,
    #[gen_stub(override_type(type_repr = "typing.Literal['T', 'V']", imports = ("typing")))]
    construction: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['sublevel', 'superlevel']", imports = ("typing")))]
    filtration: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let construction = match construction {
        "T" | "t" => Construction::T,
        "V" | "v" => Construction::V,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown construction {other:?}: expected 'T' or 'V'"
            )))
        }
    };
    let filtration = match filtration {
        "sublevel" => Filtration::Sublevel,
        "superlevel" => Filtration::Superlevel,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown filtration {other:?}: expected 'sublevel' or 'superlevel'"
            )))
        }
    };
    let (pixels, shape) = read_image(py, image)?;
    let max_dim = max_dim.unwrap_or(shape.len() - 1);
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py.detach(|| {
        algorithm.run(CubicalRun {
            pixels,
            shape,
            construction,
            filtration,
            max_dim,
            method,
            representatives,
            union_find: algorithm.union_find(),
            parallel_setup: algorithm.parallel_setup(),
        })
    })?;
    bars.map_err(PyValueError::new_err)?
        .into_python(py, &algorithm, representatives, None)
}

/// The image's pixels in C order, as `f64`, and its shape.
fn read_image(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("ascontiguousarray", (image, numpy.getattr("float64")?))?;
    let array = array.cast::<PyArrayDyn<f64>>()?.readonly();
    let shape = array.shape().to_vec();
    if !(1..=MAX_AXES).contains(&shape.len()) {
        return Err(PyValueError::new_err(format!(
            "image must have between 1 and {MAX_AXES} axes, but has {}",
            shape.len()
        )));
    }
    Ok((array.as_slice()?.to_vec(), shape))
}

/// Compute the barcode of the cubical complex with whichever algorithm is configured.
struct CubicalRun {
    pixels: Vec<f64>,
    shape: Vec<usize>,
    construction: Construction,
    filtration: Filtration,
    max_dim: usize,
    method: Method,
    representatives: bool,
    union_find: bool,
    parallel_setup: bool,
}

impl CubicalRun {
    fn run<R: ColumnRepresentation, const D: usize>(
        self,
        algorithm: Algorithm<'_, R>,
    ) -> Result<SortedBars, String> {
        let shape: [usize; D] = self.shape.try_into().expect("checked the number of axes");
        let complex = CubicalComplex::new(self.pixels, shape, self.construction, self.filtration)
            .map_err(|e| e.to_string())?;
        // In the sublevel filtration's values (of the negated image, for superlevel sets), so that `SortedBars` knows
        // which cells never appear; negated back afterwards
        let bars = filtration_barcode(
            &complex,
            self.max_dim,
            self.method,
            algorithm,
            self.representatives,
            false,
            self.union_find,
                    self.parallel_setup,
        );
        let bars = SortedBars::new(bars, self.max_dim + 1, |cube| {
            (complex.dimension(*cube), complex.coordinates(*cube).to_vec())
        });
        Ok(match self.filtration {
            Filtration::Sublevel => bars,
            Filtration::Superlevel => bars.negate(),
        })
    }
}

impl AlgorithmVisitor for CubicalRun {
    type Output = Result<SortedBars, String>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        match self.shape.len() {
            1 => self.run::<R, 1>(algorithm),
            2 => self.run::<R, 2>(algorithm),
            3 => self.run::<R, 3>(algorithm),
            4 => self.run::<R, 4>(algorithm),
            _ => unreachable!("checked the number of axes"),
        }
    }
}

/// The cubical complex of an image as an explicit chain complex, with the oriented boundary (coefficients ±1), for
/// your own computations: for instance over the integers,
///
/// ```python
/// c = cubical_complex(image)
/// chain = ChainComplex.from_csc(c["indptr"], c["indices"], c["dimensions"], data=c["coefficients"], modulus=0)
/// filtered = chain.filtration(c["values"])     # then .persistence(), .torsion(), .spectral_sequence()
/// ```
///
/// Args:
///     image: The pixel values, as for `cubical`.
///     construction: `'T'` or `'V'`, as for `cubical`.
///     filtration: `'sublevel'` or `'superlevel'`, as for `cubical`; for superlevel sets the values are those of the
///         negated image (so that they increase).
///
/// Returns:
///     A dict of numpy arrays, one entry per cell that enters the filtration: `dimensions`, the boundary as `indptr`,
///     `indices` (faces as positions among the cells) and `coefficients`, `values` (sublevel), and `coordinates`
///     (each cell's coordinates on the doubled grid, as `cubical`'s cells).
#[gen_stub_pyfunction(module = "phlite.images")]
#[gen_stub(override_return_type(type_repr = "dict[str, numpy.typing.NDArray[typing.Any]]", imports = ("numpy.typing", "typing")))]
#[pyfunction]
#[pyo3(signature = (image, *, construction = "T", filtration = "sublevel"))]
fn cubical_complex<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "numpy.typing.ArrayLike", imports = ("numpy.typing")))]
    image: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Literal['T', 'V']", imports = ("typing")))]
    construction: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['sublevel', 'superlevel']", imports = ("typing")))]
    filtration: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray2};
    let construction = match construction {
        "T" | "t" => Construction::T,
        "V" | "v" => Construction::V,
        other => return Err(PyValueError::new_err(format!("Unknown construction {other:?}: expected 'T' or 'V'"))),
    };
    let filtration = match filtration {
        "sublevel" => Filtration::Sublevel,
        "superlevel" => Filtration::Superlevel,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown filtration {other:?}: expected 'sublevel' or 'superlevel'"
            )))
        }
    };
    let (pixels, shape) = read_image(py, image)?;
    let ndim = shape.len();
    #[allow(clippy::type_complexity)]
    fn export<const D: usize>(
        pixels: Vec<f64>,
        shape: Vec<usize>,
        construction: Construction,
        filtration: Filtration,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<f64>, Vec<u32>), String> {
        let shape: [usize; D] = shape.try_into().expect("checked the number of axes");
        let complex = CubicalComplex::new(pixels, shape, construction, filtration).map_err(|e| e.to_string())?;
        let (d, p, i, c, v, coordinates) = complex.explicit();
        Ok((d, p, i, c, v, coordinates.into_iter().flatten().collect()))
    }
    let (dims, indptr, indices, coefficients, values, coordinates) = py
        .detach(|| match ndim {
            1 => export::<1>(pixels, shape, construction, filtration),
            2 => export::<2>(pixels, shape, construction, filtration),
            3 => export::<3>(pixels, shape, construction, filtration),
            4 => export::<4>(pixels, shape, construction, filtration),
            _ => unreachable!("checked the number of axes"),
        })
        .map_err(PyValueError::new_err)?;
    let n = dims.len();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("dimensions", dims.into_pyarray(py))?;
    out.set_item("indptr", indptr.into_pyarray(py))?;
    out.set_item("indices", indices.into_pyarray(py))?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("values", values.into_pyarray(py))?;
    let coordinates = numpy::ndarray::Array2::from_shape_vec((n, ndim), coordinates).expect("ndim per cell");
    let coordinates: Bound<'_, PyArray2<u32>> = coordinates.into_pyarray(py);
    out.set_item("coordinates", coordinates)?;
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cubical, m)?)?;
    m.add_function(wrap_pyfunction!(cubical_complex, m)?)
}
