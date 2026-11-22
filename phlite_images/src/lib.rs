//! Persistent homology of images, implemented with [`phlite`]: the cubical complex of a 2D image or 3D volume (or of a
//! grid with any number of axes up to [`cubical::MAX_AXES`]).
//!
//! See [`cubical`]: a dedicated cubical matrix oracle, with cells numbered arithmetically (as `phlite_clouds::rips`
//! numbers simplices), and both the V-construction (pixels as vertices) and the T-construction (pixels as top cells),
//! for sublevel and superlevel filtrations. With the feature `python`, this crate is also the Python package
//! `phlite-images`, `phlite.images` (its bindings are in `src/python`).

pub mod cubical;
pub mod filtrations;

#[cfg(feature = "python")]
mod python;

/// What `phlite_images_stub_gen` writes: `python/phlite/images/__init__.pyi`.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    phlite_bindings::stub_info_for("phlite.images", env!("CARGO_MANIFEST_DIR"))
}
