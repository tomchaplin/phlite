pub mod columns;
pub mod fields;
pub mod matricies;
pub mod reduction;

#[derive(Debug)]
pub enum PhliteError {
    NotInDomain,
    NotInCodomain,
}

#[cfg(test)]
mod tests {}
