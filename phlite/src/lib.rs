//! The main capabilities provided by this crate are
//! 1. A framework for implementing and manipulating lazy oracles into sparse matrices.
//!    This is implemented in the [`matrices`] module.
//! 2. `R=DV` reduction algorithms for sufficiently well-structured oracles of this sort.
//!    These are implemented in the [`reduction`] module.
//!
//! Usage of this crate typically involves the following steps:
//! 1. Decide on the matrix you would like to reduce.
//!    * Typically this is the total boundary or total coboundary matrix for your filtered chain complex.
//!    * Note that to decide on this matrix you must make a _choice_ of basis for both the target and domain of your linear transformation. Often, though not always, there is a fairly obvious choice of standard basis.
//!    * Also note, if you wish to to compute _relative_ cohomology (i.e. reduce the anti-transpose of your boundary matrix) then it usually best to just implement the coboundary matrix directly (in the default filtration order) and then call [`reverse`](matrices::MatrixOracle::reverse).
//! 2. Implement the [`MatrixOracle`](matrices::MatrixOracle) trait for your matrix.
//!    * As part of this, you will have to choose a type for the indices of your columns and rows. Unlike most matrix software packages, these don't need to be integers, they just have to implement [`BasisElement`](matrices::BasisElement).
//!    * Note that row and column indicies get `clone`d __a lot__ so make sure this is relatively cheap to `clone` and store in memory. For example, Ripser encodes the simplices which index its columns and rows as single integers!
//!    * You will have to implement a default ordering on your indices in order to break ties.
//!    * The main method you have to implement is [`column`](matrices::MatrixOracle::column) which provides lazy access into the columns of your matrices. The research suggests that you probably want to recompute your columns on every access, in order to reduce memory usage!
//! 2. Implement the [`HasColBasis`](matrices::HasColBasis) trait.
//!    * Earlier we had to _choose_ a basis for our linear transformation in order to get a well-defined matrix to implement, but note we didn't have to compute the basis.
//!    * In order to implement [`HasColBasis`](matrices::HasColBasis) you will probably now have to compute the basis for your column space and store it in a struct implementing [`ColBasis`](matrices::ColBasis) (which gives random access to the elemnts of your basis).
//!    * You can then attach this basis to your matrix by calling [`with_basis`](matrices::MatrixOracle::with_basis).
//!    * In order to take advantage of the clearing optimsiation you should store basis elements in each dimension in a separate [`ColBasis`](matrices::ColBasis) and then gather them into a global [`ColBasis`](matrices::ColBasis) which implements [`SplitByDimension`](matrices::SplitByDimension)
//!    * Note, we didn't have to compute a row basis (though we did have to choose one) - this can massively speed up computation of cohomology in the maximum dimension!
//! 3. Implement the [`HasRowFiltration`](matrices::HasRowFiltration) trait.
//!    * This trait embues our matrix with a filtration function for our rows.
//!    * The output of the type of the filtraiton function has to implement [`Ord`] so [`NotNan<f64>`](ordered_float::NotNan) is often a good choice.
//!    * You might be able to compute this directly from the information stored in your matrix already. Failing that you can attach a closure using [`with_filtration`](matrices::MatrixOracle::with_filtration).
//!    * If you have no filtration and just want to compute homology, use [`with_trivial_filtration`](matrices::MatrixOracle::with_trivial_filtration).
//! 4. Pass the resulting matrix to a [`reduction`] algorithm to compute persistent homology.
//!
//! As you can see, `phlite` takes a "bring your own `*`" approach where `*` matches matrix representation/basis representation/matrix index types/filtration types.
//! To see an example implementation of Vietoris-Rips persitent homology, with a step-by-step tutorial, have a look at the [`phlite_rips`](https://github.com/tomchaplin/phlite/tree/main/phlite_rips) crate.

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
