//! Persistent homology of images, implemented with [`phlite`]: the cubical complex of a 2D or 3D grid of values.
//!
//! Nothing here yet. The plan is a dedicated cubical matrix oracle (cells indexed arithmetically, as
//! `phlite_cloud::rips` indexes simplices), with both the V-construction (values on the top cells) and the
//! T-construction (values on the vertices), in 2D and 3D. In Python this crate is the `phlite.image` submodule (its
//! bindings are in `phlite_py/src/python/image`).
//!
//! Until then, the `phlite` Python package computes cubical persistence from an explicit boundary matrix: see the
//! image notebook (`docs/notebooks/phlite_image.py`).
