//! An [`AlgorithmSpec`] as plain Python data, which is how algorithms cross between phlite's extension modules.

use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::algorithm::{storage_from_name, storage_name, AlgorithmSpec, Columns, Kind};

/// The version of the format of [`spec_to_dict`]. Every package is released in lockstep, which [`check_core`]
/// enforces, so this only catches a spec from anywhere else.
///
/// [`check_core`]: crate::check_core
pub const SPEC_PROTOCOL: u32 = 1;

/// The spec as plain Python data, which is what `phlite.core.Algorithm._spec()` returns:
///
/// ```text
/// {"protocol": 1, "kind": "standard" | "clearing" | "parallel-clearing",
///  "columns": None | "merged" | "heap" | "sorted" | "pivot" | "dynamic", "threshold": None | int,
///  "cache": None | bool, "store_r": None | bool,
///  "skip_zero_apparent_pairs": bool, "dimension_order": None | list[int], "storage": "locking" | "lock-free",
///  "threads": None | int, "union_find": None | bool, "parallel_setup": None | bool}
/// ```
///
/// `None` means the pipeline's default.
pub fn spec_to_dict<'py>(py: Python<'py>, spec: &AlgorithmSpec) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("protocol", SPEC_PROTOCOL)?;
    dict.set_item("kind", spec.kind.name())?;
    dict.set_item("columns", spec.columns.map(Columns::name))?;
    dict.set_item("threshold", spec.threshold)?;
    dict.set_item("cache", spec.cache)?;
    dict.set_item("store_r", spec.store_r)?;
    dict.set_item("skip_zero_apparent_pairs", spec.skip_zero_apparent_pairs)?;
    dict.set_item("dimension_order", spec.dimension_order.clone())?;
    dict.set_item("storage", storage_name(spec.storage))?;
    dict.set_item("threads", spec.threads)?;
    dict.set_item("union_find", spec.union_find)?;
    dict.set_item("parallel_setup", spec.parallel_setup)?;
    Ok(dict)
}

/// Read a dict made by [`spec_to_dict`], and validate it.
pub fn spec_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<AlgorithmSpec> {
    let get = |key: &str| -> PyResult<Bound<'_, PyAny>> {
        dict.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("The algorithm's spec has no {key:?}")))
    };
    let protocol: u32 = get("protocol")?.extract()?;
    if protocol != SPEC_PROTOCOL {
        return Err(PyValueError::new_err(format!(
            "The algorithm's spec has protocol {protocol}, but this package reads protocol {SPEC_PROTOCOL}: install \
             the same version of every phlite package"
        )));
    }
    let columns: Option<String> = get("columns")?.extract()?;
    let spec = AlgorithmSpec {
        kind: Kind::from_name(&get("kind")?.extract::<String>()?)?,
        columns: columns.as_deref().map(Columns::from_name).transpose()?,
        threshold: get("threshold")?.extract()?,
        cache: get("cache")?.extract()?,
        store_r: get("store_r")?.extract()?,
        skip_zero_apparent_pairs: get("skip_zero_apparent_pairs")?.extract()?,
        dimension_order: get("dimension_order")?.extract()?,
        storage: storage_from_name(&get("storage")?.extract::<String>()?)?,
        threads: get("threads")?.extract()?,
        union_find: get("union_find")?.extract()?,
        parallel_setup: get("parallel_setup")?.extract()?,
    };
    spec.validate()?;
    Ok(spec)
}
