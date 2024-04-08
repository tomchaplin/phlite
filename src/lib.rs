pub mod columns;
pub mod fields;
pub mod filtrations;
pub mod matricies;
pub mod reduction;

// TODO: Consider peeling off filtration and column basis into separate object?
// TODO: Better system for column algebra
// TODO: Implement a more efficient matrix reduction algorithm + oracle
// TODO: Documentation

#[derive(Debug)]
pub enum PhliteError {
    NotInDomain,
    NotInCodomain,
}

#[cfg(test)]
mod tests {}
