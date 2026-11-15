//! Simplices as sequences of vertices, packed into a single `u128`.
//!
//! The directed flag complex and the ordered-tuple complex have *ordered* simplices (the order of the vertices
//! matters, and for ordered tuples a vertex may appear more than once), while a Dowker complex has ordinary simplices,
//! which we store as their vertices in increasing order. Either way a simplex is a short sequence of vertices, and a
//! [`Simplex`] packs it into a `u128`: the dimension in the top four bits, then each vertex in a field of
//! [`Codec::bits`] bits, the first vertex most significant. So simplices are ordered by dimension and then
//! lexicographically, by comparing a single integer, and inserting or removing a vertex is a few shifts and masks.

use phlite::matrices::BasisElement;

/// A simplex (or ordered simplex): a sequence of at most [`MAX_VERTICES`] vertices, packed as described in the
/// [module docs](self). Decode it with a [`Codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Simplex(u128);

impl BasisElement for Simplex {}

/// The most vertices a [`Simplex`] can have (the dimension has four bits); fewer for large graphs, see
/// [`Codec::max_vertices`].
pub const MAX_VERTICES: usize = 16;

/// The bits below the dimension, holding the vertices.
const BODY_BITS: u32 = 124;
const BODY: u128 = (1 << BODY_BITS) - 1;

impl Simplex {
    /// The dimension (one less than the number of vertices).
    #[inline]
    pub fn dimension(self) -> usize {
        (self.0 >> BODY_BITS) as usize
    }

    /// The number of vertices.
    #[inline]
    pub fn n_vertices(self) -> usize {
        self.dimension() + 1
    }
}

/// The vertices of a simplex, in order, on the stack (see [`Codec::vertices`]).
#[derive(Clone, Copy)]
pub struct Vertices {
    buf: [u32; MAX_VERTICES],
    len: usize,
}

impl std::ops::Deref for Vertices {
    type Target = [u32];

    #[inline]
    fn deref(&self) -> &[u32] {
        &self.buf[..self.len]
    }
}

impl std::fmt::Debug for Vertices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Packs and unpacks [`Simplex`]es on the vertices `0, ..., n - 1`, each vertex in [`bits`](Self::bits) bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codec {
    bits: u32,
}

impl Codec {
    /// A codec for simplices on `n_vertices` vertices.
    pub fn new(n_vertices: u32) -> Self {
        // Vertices 0..n need ceil(log2(n)) bits (and at least one)
        let bits = (u32::BITS - n_vertices.saturating_sub(1).leading_zeros()).max(1);
        Self { bits }
    }

    /// The number of bits per vertex.
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// The most vertices a simplex can have: 16, or fewer when the vertices need more than 7 bits (124 / `bits`).
    pub fn max_vertices(self) -> usize {
        ((BODY_BITS / self.bits) as usize).min(MAX_VERTICES)
    }

    /// The position of the lowest bit of the field holding vertex `i`.
    #[inline]
    fn shift(self, i: usize) -> u32 {
        BODY_BITS - (i as u32 + 1) * self.bits
    }

    /// The body bits of the fields of the vertices before position `i`.
    #[inline]
    fn before(self, i: usize) -> u128 {
        let top = BODY_BITS - i as u32 * self.bits;
        BODY & !((1u128 << top) - 1)
    }

    /// The simplex with these vertices, in this order.
    ///
    /// # Panics
    /// If there are no vertices, or more than [`max_vertices`](Self::max_vertices).
    pub fn encode(self, vertices: &[u32]) -> Simplex {
        assert!(
            !vertices.is_empty() && vertices.len() <= self.max_vertices(),
            "a simplex has between 1 and {} vertices",
            self.max_vertices()
        );
        let mut packed = ((vertices.len() - 1) as u128) << BODY_BITS;
        for (i, &v) in vertices.iter().enumerate() {
            debug_assert!(self.bits == 32 || v >> self.bits == 0);
            packed |= (v as u128) << self.shift(i);
        }
        Simplex(packed)
    }

    /// The vertex in position `i` of `simplex`.
    #[inline]
    pub fn vertex(self, simplex: Simplex, i: usize) -> u32 {
        ((simplex.0 >> self.shift(i)) & ((1u128 << self.bits) - 1)) as u32
    }

    /// The vertices of `simplex`, in order.
    #[inline]
    pub fn vertices(self, simplex: Simplex) -> Vertices {
        let mut vertices = Vertices {
            buf: [0; MAX_VERTICES],
            len: simplex.n_vertices(),
        };
        for i in 0..vertices.len {
            vertices.buf[i] = self.vertex(simplex, i);
        }
        vertices
    }

    /// `simplex` with `vertex` inserted at position `position` (so `vertex` becomes its `position`-th vertex, and
    /// those from there on move up one).
    #[inline]
    pub fn insert(self, simplex: Simplex, position: usize, vertex: u32) -> Simplex {
        debug_assert!(simplex.n_vertices() < self.max_vertices());
        let body = simplex.0 & BODY;
        let before = self.before(position);
        let dimension = (simplex.dimension() as u128 + 1) << BODY_BITS;
        Simplex(
            dimension
                | (body & before)
                | ((vertex as u128) << self.shift(position))
                | ((body & !before) >> self.bits),
        )
    }

    /// `simplex` with the vertex in position `position` removed.
    #[inline]
    pub fn remove(self, simplex: Simplex, position: usize) -> Simplex {
        debug_assert!(simplex.dimension() > 0);
        let body = simplex.0 & BODY;
        let before = self.before(position);
        let after = !self.before(position + 1) & BODY;
        let dimension = (simplex.dimension() as u128 - 1) << BODY_BITS;
        Simplex(dimension | (body & before) | ((body & after) << self.bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_is_dimension_then_lexicographic() {
        let codec = Codec::new(10);
        let mut simplices = vec![
            vec![3],
            vec![0, 9],
            vec![1, 0],
            vec![0, 1],
            vec![9, 8, 7],
            vec![0, 1, 2, 3],
            vec![0, 2, 1],
            vec![9],
            vec![0],
        ];
        let mut encoded: Vec<_> = simplices.iter().map(|s| codec.encode(s)).collect();
        simplices.sort_by_key(|s| (s.len(), s.clone()));
        encoded.sort();
        let decoded: Vec<_> = encoded.iter().map(|&s| codec.vertices(s).to_vec()).collect();
        assert_eq!(decoded, simplices);
    }

    #[test]
    fn insert_and_remove_match_encoding() {
        for n in [2, 3, 100, 1 << 16, u32::MAX] {
            let codec = Codec::new(n);
            let top = n - 1;
            let max = codec.max_vertices();
            for len in 1..max {
                let vertices: Vec<u32> = (0..len as u32).map(|i| top.wrapping_sub(i * 7) % n).collect();
                let simplex = codec.encode(&vertices);
                assert_eq!(simplex.dimension(), len - 1);
                assert_eq!(codec.vertices(simplex).to_vec(), vertices);
                for position in 0..=len {
                    let mut inserted = vertices.clone();
                    inserted.insert(position, top);
                    assert_eq!(codec.insert(simplex, position, top), codec.encode(&inserted));
                }
                if len > 1 {
                    for position in 0..len {
                        let mut removed = vertices.clone();
                        removed.remove(position);
                        assert_eq!(codec.remove(simplex, position), codec.encode(&removed));
                    }
                }
            }
        }
    }

    #[test]
    fn capacity() {
        assert_eq!(Codec::new(1).bits(), 1);
        assert_eq!(Codec::new(2).bits(), 1);
        assert_eq!(Codec::new(3).bits(), 2);
        assert_eq!(Codec::new(128).max_vertices(), 16);
        assert_eq!(Codec::new(1024).max_vertices(), 12);
        assert_eq!(Codec::new(65_536).max_vertices(), 7);
        assert_eq!(Codec::new(u32::MAX).max_vertices(), 3);
    }
}
