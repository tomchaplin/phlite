//! The main capabilities provided by this crate are
//! 1. A framework for implementing and manipulating lazy oracles into sparse matrices.
//!    This is implemented in the [`matrices`] module.
//! 2. `R=DV` reduction algorithms for sufficiently well-structured oracles of this sort.
//!    These are implemented in the [`reduction`] module.
//!
//! Usage of this crate typically involves the following steps:
//! 1. Implement the [`MatrixOracle`](matrices::MatrixOracle) trait.
//! 2. Implement the [`HasColBasis`](matrices::HasColBasis) and [`HasRowFiltration`](matrices::HasRowFiltration) traits.
//! 3. Pass the resulting matrix to a [`reduction`] algorithm to compute persistent homology.
//!
//! TODO:
//! * Motivate crate
//! * Overview steps to implementing a `phlite` matrix
//! * Talk through the boundary matrix in the rips example? Or maybe reference?

pub mod columns;
pub mod fields;
pub mod matrices;
pub mod reduction;

// TODO: Better system for column algebra
// TODO: Documentation
// TODO: Figure out whether it makes more sense to pass a &col to MatrixOracle::column
// TODO: Add better tests throughout
// TODO: Implement lock-free reduction algorithm

#[cfg(test)]
mod tests {}
