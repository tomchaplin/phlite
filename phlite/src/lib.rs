//! TODO:
//! * Motivate crate
//! * Overview steps to implementing a `phlite` matrix
//! * Talk through the boundary matrix in the rips example? Or maybe reference?

pub mod columns;
pub mod fields;
pub mod matrices;
pub mod reduction;

// TODO: Consider peeling off filtration and column basis into separate object?
// TODO: Better system for column algebra
// TODO: Implement a more efficient matrix reduction algorithm + oracle
// TODO: Documentation

#[cfg(test)]
mod tests {}
