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

/// The unified error type for all `phlite` traits and functions.
#[derive(Debug)]
pub enum PhliteError {
    /// The requested column is not in the domain of the matrix (column space)
    NotInDomain,
    /// The requested column is not in the codomain of the matrix (row space)
    NotInCodomain,
}

#[cfg(test)]
mod tests {}
