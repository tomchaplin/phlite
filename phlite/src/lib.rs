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
