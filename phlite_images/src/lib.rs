//! Persistent homology of images, implemented with [`phlite`]: the cubical complex of a 2D or 3D grid of values.
//!
//! Nothing here yet. The plan is a dedicated cubical matrix oracle (cells indexed arithmetically, as
//! `phlite_clouds::rips` indexes simplices), with both the V-construction (values on the top cells) and the
//! T-construction (values on the vertices), in 2D and 3D. With the feature `python`, this crate is also the Python
//! package `phlite-images`, `phlite.images` (its bindings are in `src/python`, one file per pipeline).
//!
//! Until then, `phlite.complexes` computes cubical persistence from an explicit boundary matrix: see the image
//! notebook (`docs/notebooks/phlite_image.py`).

#[cfg(feature = "python")]
mod python;

/// What `phlite_images_stub_gen` writes: `python/phlite/images/__init__.pyi`.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    phlite_bindings::stub_info_for("phlite.images", env!("CARGO_MANIFEST_DIR"))
}
