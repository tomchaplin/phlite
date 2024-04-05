#![feature(binary_heap_into_iter_sorted)]

pub mod columns;
pub mod fields;
pub mod matricies;

#[derive(Debug)]
pub enum PhliteError {
    NotInDomain,
    NotInCodomain,
}

#[cfg(test)]
mod tests {}
