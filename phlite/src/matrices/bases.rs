//! Ready-made basis types, in particular a split basis backed by a single flat allocation.

use super::{BasisElement, ColBasis, SplitByDimension};

/// A [`ColBasis`] split into sub-bases by dimension, stored as one flat `Vec` plus a table of offsets.
///
/// This replaces the common pattern of storing a `Vec<SubBasis>` and handing out references to each sub-basis.
/// Here the sub-bases never exist as owned values: [`SplitByDimension`] is implemented on the view `&SplitBasis<T>`
/// and produces a [`BasisSlice`] borrowing from the flat storage.
/// Elements of the whole basis are the concatenation of the sub-bases in dimension order, so random access
/// into the total basis is O(1) (no scan over dimensions).
#[derive(Clone, Debug)]
pub struct SplitBasis<T> {
    elems: Vec<T>,
    /// `offsets[d]..offsets[d + 1]` is the range of dimension `d` within `elems`.
    offsets: Vec<usize>,
}

impl<T: BasisElement> SplitBasis<T> {
    /// Build from an iterator over the (already ordered) elements in each dimension, starting at dimension 0.
    pub fn from_dimensions<I, J>(dimensions: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = T>,
    {
        let mut elems = vec![];
        let mut offsets = vec![0];
        for dim in dimensions {
            elems.extend(dim);
            offsets.push(elems.len());
        }
        Self { elems, offsets }
    }

    /// The number of dimensions in the basis.
    pub fn n_dimensions(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl<T: BasisElement> ColBasis for SplitBasis<T> {
    type ElemT = T;

    fn element(&self, index: usize) -> Self::ElemT {
        self.elems[index].clone()
    }

    fn size(&self) -> usize {
        self.elems.len()
    }
}

impl<'a, T: BasisElement> SplitByDimension for &'a SplitBasis<T> {
    type SubBasisT = BasisSlice<'a, T>;

    fn n_dimensions(&self) -> usize {
        SplitBasis::n_dimensions(self)
    }

    fn in_dimension(self, dimension: usize) -> Self::SubBasisT {
        BasisSlice(&self.elems[self.offsets[dimension]..self.offsets[dimension + 1]])
    }
}

/// A [`ColBasis`] given by a borrowed slice of elements. This is [`Copy`] regardless of `T`.
#[derive(Debug)]
pub struct BasisSlice<'a, T>(pub &'a [T]);

impl<T> Clone for BasisSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for BasisSlice<'_, T> {}

impl<T: BasisElement> ColBasis for BasisSlice<'_, T> {
    type ElemT = T;

    fn element(&self, index: usize) -> Self::ElemT {
        self.0[index].clone()
    }

    fn size(&self) -> usize {
        self.0.len()
    }
}
