//! The T-construction in three dimensions by a filtered discrete gradient: Mischaikow and Nanda's filtered Morse
//! reduction, as Perseus 6 does it for cubical grids, then the Morse complex reduced as an explicit complex.
//!
//! Cells are put in *blocks*: a cell enters with its earliest top coface `τ` (by rank, then index), so the cells of
//! `τ`'s block are the faces of `τ` none of whose other top cofaces is earlier. Each block is matched once per shape
//! (and cached by its set of cells) by a greedy collapse inside the block: repeatedly pair a cell with its only coface
//! left in the block (that coface having none left), or, when there is no such pair, leave a cell with no coface left
//! (of the highest dimension) critical. In the reverse of the order cells were removed, each pair is adjacent and faces
//! come before cofaces, and ordering the blocks by their top cells refines the filtration: so the matching is a
//! filtered discrete gradient, and its Morse complex has the barcode of the cubical complex (up to bars of length
//! zero). On the benchmark images the matching is perfect (twice the finite bars plus the essential ones).
//!
//! The Morse complex has the critical cells, in that order, with the boundary over Z/2 given by gradient paths: a face
//! `f` of a critical cell contributes `f` if it is critical, nothing if it is matched with a face of its own, and the
//! flow of its partner's other faces if it is matched with a coface (memoised, since paths share cells). That complex
//! is reduced by `phlite_complexes`, union-find at both ends included.
//!
//! Both passes can run in parallel: every cell is in exactly one block, so blocks are matched independently (pixels in
//! index order, the critical cells then sorted into filtration order), and the flows into cells of each dimension only
//! touch cells of that dimension, so the boundaries of each dimension are computed independently.
//!
//! Only worth it in three dimensions: in one or two, union-find at both ends already leaves nothing to reduce. It keeps
//! one `u32` per cell of the doubled grid, so it is used up to [`MAX_MORSE_CELLS`] cells.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use rayon::prelude::*;

use phlite::barcode::{Algorithm, Bar, Method};
use phlite::columns::ColumnRepresentation;
use phlite::fields::Z2;
use phlite_complexes::complex::{Cell, Chain, RawChain};

use super::{Cube, CubicalBar, CubicalComplex};

/// The most cells of the doubled grid for which the Morse reduction is used (one `u32` each).
pub const MAX_MORSE_CELLS: usize = 1 << 27;

/// Local cells of a block: offsets `o ∈ {-1, 0, 1}^3` from the top cell, numbered in base 3 (digit `o_a + 1` for axis
/// `a`, the last axis least significant); `o_a = 0` means the cell spans axis `a`.
const N_LOCAL: usize = 27;
const NOT_IN_BLOCK: u8 = u8::MAX;

/// Per-cell state, in one `u32`: the partner cell (27 bits) with the partner's [axis mask](Geometry::mask) above it,
/// `CRITICAL | index`, or `FLOW | span` once the flow of a cell matched with a coface is known; `ABSENT` for cells not
/// in the complex (never reached). Carrying masks spares the traversal any division.
const TAG: u32 = 3 << 30;
const ABSENT: u32 = 1 << 30;
const CRITICAL: u32 = 2 << 30;
const FLOW: u32 = 3 << 30;
/// A critical cell above the dimensions wanted (never reached either).
const UNWANTED: u32 = CRITICAL | ((1 << 30) - 1);
const CELL_BITS: u32 = 27;
const CELL: u32 = (1 << CELL_BITS) - 1;

fn offsets_of(mut l: usize) -> [i8; 3] {
    let mut o = [0i8; 3];
    for a in (0..3).rev() {
        o[a] = (l % 3) as i8 - 1;
        l /= 3;
    }
    o
}

fn local_of(o: [i8; 3]) -> usize {
    o.iter().fold(0, |acc, &x| acc * 3 + (x + 1) as usize)
}

/// The geometry of a block, computed once.
struct Tables {
    /// For each local cell, the axes it spans (bit `a` for axis `a`)
    mask: [u8; N_LOCAL],
    /// For each local cell, the neighbours (bits `0..26`: the non-zero offsets in local order) that are top cofaces of it
    cofaces_mask: [u32; N_LOCAL],
    faces: [Vec<u8>; N_LOCAL],
    cofaces: [Vec<u8>; N_LOCAL],
    dimension: [u8; N_LOCAL],
    /// The non-zero offsets, by bit
    neighbours: Vec<[i8; 3]>,
}

impl Tables {
    fn new() -> Self {
        let neighbours: Vec<[i8; 3]> = (0..N_LOCAL)
            .map(offsets_of)
            .filter(|o| *o != [0; 3])
            .collect();
        let cofaces_mask = std::array::from_fn(|l| {
            let o = offsets_of(l);
            neighbours
                .iter()
                .enumerate()
                .filter(|(_, t)| t.iter().zip(&o).all(|(&t, &o)| t == 0 || t == o))
                .fold(0, |m, (bit, _)| m | 1 << bit)
        });
        let faces = std::array::from_fn(|l| {
            let o = offsets_of(l);
            let mut out = vec![];
            for a in 0..3 {
                if o[a] == 0 {
                    for s in [-1, 1] {
                        let mut f = o;
                        f[a] = s;
                        out.push(local_of(f) as u8);
                    }
                }
            }
            out
        });
        let cofaces = std::array::from_fn(|l| {
            let o = offsets_of(l);
            let mut out = vec![];
            for a in 0..3 {
                if o[a] != 0 {
                    let mut c = o;
                    c[a] = 0;
                    out.push(local_of(c) as u8);
                }
            }
            out
        });
        let dimension =
            std::array::from_fn(|l| offsets_of(l).iter().filter(|&&x| x == 0).count() as u8);
        let mask = std::array::from_fn(|l| {
            let o = offsets_of(l);
            (0..3).filter(|&a| o[a] == 0).fold(0, |m, a| m | 1 << a)
        });
        Self {
            mask,
            cofaces_mask,
            faces,
            cofaces,
            dimension,
            neighbours,
        }
    }
}

/// A block's matching: each local cell's partner (itself if critical, [`NOT_IN_BLOCK`] if not in the block), and the
/// critical cells in the block's order.
struct Pattern {
    partner: [u8; N_LOCAL],
    critical: Vec<u8>,
}

/// Match the block of the local cells in `in_block` (a bit per local cell) by a greedy collapse.
fn match_block(tables: &Tables, in_block: u32) -> Pattern {
    let member = |l: usize| in_block >> l & 1 == 1;
    let mut left: [u8; N_LOCAL] = std::array::from_fn(|l| {
        tables.cofaces[l]
            .iter()
            .filter(|&&c| member(c as usize))
            .count() as u8
    });
    let mut alive = in_block;
    let mut partner = [NOT_IN_BLOCK; N_LOCAL];
    let mut sequence: Vec<u8> = Vec::with_capacity(N_LOCAL);
    let mut remove = |l: usize, alive: &mut u32, left: &mut [u8; N_LOCAL]| {
        *alive &= !(1 << l);
        sequence.push(l as u8);
        for &f in &tables.faces[l] {
            if *alive >> f & 1 == 1 {
                left[f as usize] -= 1;
            }
        }
    };
    while alive != 0 {
        let now = alive;
        let alive_cells = move || (0..N_LOCAL).filter(move |&l| now >> l & 1 == 1);
        // A free pair: a cell with one coface left, which has none left
        let free = alive_cells().find_map(|s| {
            if left[s] != 1 {
                return None;
            }
            let t = *tables.cofaces[s].iter().find(|&&c| now >> c & 1 == 1)? as usize;
            (left[t] == 0).then_some((s, t))
        });
        if let Some((s, t)) = free {
            remove(t, &mut alive, &mut left);
            remove(s, &mut alive, &mut left);
            partner[s] = t as u8;
            partner[t] = s as u8;
            continue;
        }
        // Otherwise a cell with no coface left, of the highest dimension: critical
        let c = alive_cells()
            .filter(|&l| left[l] == 0)
            .max_by_key(|&l| (tables.dimension[l], std::cmp::Reverse(l)))
            .expect("a finite block has a maximal cell");
        remove(c, &mut alive, &mut left);
        partner[c] = c as u8;
    }
    let critical = sequence
        .iter()
        .rev()
        .copied()
        .filter(|&l| partner[l as usize] == l)
        .collect();
    Pattern { partner, critical }
}

/// Sort `scratch` and append it to `arena` with equal pairs cancelled (a sum over Z/2); returns the span written.
fn cancel_into(scratch: &mut [u32], arena: &mut Vec<u32>) -> (u32, u32) {
    scratch.sort_unstable();
    let start = arena.len();
    let mut i = 0;
    while i < scratch.len() {
        if i + 1 < scratch.len() && scratch[i] == scratch[i + 1] {
            i += 2;
        } else {
            arena.push(scratch[i]);
            i += 1;
        }
    }
    (start as u32, (arena.len() - start) as u32)
}

/// The bars of `complex` (T-construction, three axes, at most [`MAX_MORSE_CELLS`] cells) in dimensions
/// `0..=max_dim`, in its internal values, by the Morse complex of the filtered gradient, built on the current rayon
/// thread pool if `parallel`. Zero-length bars are left out.
///
/// # Panics
/// If `D` isn't 3 or the grid has too many cells.
pub(crate) fn morse_barcode<R: ColumnRepresentation, const D: usize>(
    complex: &CubicalComplex<D>,
    pixel_ranks: &[u32],
    rank_values: &[f64],
    infinite: u32,
    max_dim: usize,
    algorithm: Algorithm<'_, R>,
    parallel: bool,
) -> Vec<CubicalBar> {
    assert_eq!(D, 3);
    let shape: [usize; 3] = std::array::from_fn(|a| complex.shape[a]);
    let geometry = Geometry::new(shape);
    assert!(geometry.n_cells <= MAX_MORSE_CELLS);
    let tables = Tables::new();
    let top = (max_dim + 1).min(3);
    let (state, critical) = gradient(&geometry, &tables, pixel_ranks, infinite, top, parallel);
    let boundaries = morse_boundaries(&geometry, &state, &critical, top, parallel);
    drop(state);

    let mut raw = RawChain {
        dimensions: vec![],
        indptr: vec![0],
        indices: vec![],
        coefficients: None,
    };
    let mut next = [0usize; 4];
    for &(c, _) in &critical {
        let k = geometry.dimension(c as usize);
        raw.dimensions.push(k as i64);
        if k > 0 {
            let (indptr, indices) = &boundaries[k - 1];
            let j = next[k];
            next[k] += 1;
            raw.indices.extend(
                indices[indptr[j]..indptr[j + 1]]
                    .iter()
                    .map(|&b| i64::from(b)),
            );
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    drop(boundaries);
    let values = critical
        .iter()
        .map(|&(_, rank)| rank_values[rank as usize])
        .collect();
    let morse = Chain::<Z2>::from_raw(raw)
        .and_then(|chain| chain.filter(Some(values)))
        .expect("the Morse complex of a filtered gradient is a filtered complex");
    let bars = morse.barcode(Some(max_dim.min(3)), Method::Cohomology, algorithm, false);
    let grid = complex.grid_ref();
    let cube = |i: Cell| -> Cube {
        let c = critical[i.0 as usize].0 as usize;
        grid.cube_at(std::array::from_fn(|a| geometry.coordinate(c, a) as u32))
            .expect("a cell of the grid")
    };
    bars.into_iter()
        .filter(|bar| morse.dimension(bar.birth) <= max_dim)
        .map(|bar| Bar {
            birth: cube(bar.birth),
            birth_value: bar.birth_value,
            death: bar.death.map(|(d, v)| (cube(d), v)),
            representative: None,
        })
        .collect()
}

/// The doubled grid of a three-dimensional image: `2 n + 1` cells along each axis of `n` pixels, pixel `i` at `2 i + 1`,
/// in C order.
struct Geometry {
    shape: [usize; 3],
    dims: [usize; 3],
    strides: [usize; 3],
    n_cells: usize,
}

impl Geometry {
    fn new(shape: [usize; 3]) -> Self {
        let dims = shape.map(|n| 2 * n + 1);
        let strides = [dims[1] * dims[2], dims[2], 1];
        Self {
            shape,
            dims,
            strides,
            n_cells: dims.iter().product(),
        }
    }

    #[inline]
    fn coordinate(&self, c: usize, a: usize) -> usize {
        c / self.strides[a] % self.dims[a]
    }

    /// The axes `c` spans (its odd coordinates), a bit per axis.
    #[inline]
    fn mask(&self, c: usize) -> u32 {
        (0..3)
            .filter(|&a| self.coordinate(c, a) % 2 == 1)
            .fold(0, |m, a| m | 1 << a)
    }

    #[inline]
    fn dimension(&self, c: usize) -> usize {
        self.mask(c).count_ones() as usize
    }

    /// The faces of `c`, which spans the axes in `mask`, into `out` (each with its mask), returning how many.
    #[inline]
    fn faces(&self, c: u32, mask: u32, out: &mut [(u32, u32); 6]) -> usize {
        let mut n = 0;
        for (a, &stride) in self.strides.iter().enumerate() {
            if mask >> a & 1 == 1 {
                let face_mask = mask & !(1 << a);
                out[n] = (c - stride as u32, face_mask);
                out[n + 1] = (c + stride as u32, face_mask);
                n += 2;
            }
        }
        n
    }
}

/// The filtered gradient: each cell's state (partner, or `CRITICAL | index` numbering the critical cells of dimension
/// at most `top` in order, [`UNWANTED`] above, [`ABSENT`] outside the complex), and the critical cells in order with
/// their value's rank.
fn gradient(
    geometry: &Geometry,
    tables: &Tables,
    pixel_ranks: &[u32],
    infinite: u32,
    top: usize,
    parallel: bool,
) -> (Vec<AtomicU32>, Vec<(u32, u32)>) {
    let shape = geometry.shape;
    let pixel_strides = [shape[1] * shape[2], shape[2], 1];
    let n_pixels = pixel_ranks.len();
    // Each pixel's position in filtration order (rank, then index); the absent ones last
    let mut order: Vec<u64> = (0..n_pixels as u32)
        .filter(|&p| pixel_ranks[p as usize] < infinite)
        .map(|p| (u64::from(pixel_ranks[p as usize]) << 32) | u64::from(p))
        .collect();
    if parallel {
        order.par_sort_unstable();
    } else {
        order.sort_unstable();
    }
    let mut position = vec![u32::MAX; n_pixels];
    for (i, &key) in order.iter().enumerate() {
        position[key as u32 as usize] = i as u32;
    }
    drop(order);
    let neighbour_shift: Vec<isize> = tables
        .neighbours
        .iter()
        .map(|t| {
            (0..3)
                .map(|a| t[a] as isize * pixel_strides[a] as isize)
                .sum()
        })
        .collect();
    // Each plane of pixels (first coordinate `z`) in C order, with their coordinates, without dividing
    let plane = move |z: usize| {
        (0..shape[1]).flat_map(move |y| {
            (0..shape[2]).map(move |x| ((z * shape[1] + y) * shape[2] + x, [z, y, x]))
        })
    };
    // The cells of each present pixel's block, as a bit per local cell (0 if absent)
    let block = |p: usize, pc: [usize; 3]| -> u32 {
        let me = position[p];
        if me == u32::MAX {
            return 0;
        }
        let interior = (0..3).all(|a| pc[a] >= 1 && pc[a] + 1 < shape[a]);
        let mut earlier = 0u32;
        for (bit, t) in tables.neighbours.iter().enumerate() {
            let inside = interior
                || (0..3).all(|a| {
                    let c = pc[a] as isize + t[a] as isize;
                    c >= 0 && (c as usize) < shape[a]
                });
            if inside && position[(p as isize + neighbour_shift[bit]) as usize] < me {
                earlier |= 1 << bit;
            }
        }
        (0..N_LOCAL)
            .filter(|&l| earlier & tables.cofaces_mask[l] == 0)
            .fold(0, |m, l| m | 1 << l)
    };
    let mut blocks: Vec<u32> = if parallel {
        (0..shape[0])
            .into_par_iter()
            .flat_map_iter(|z| plane(z).map(|(p, pc)| block(p, pc)))
            .collect()
    } else {
        (0..shape[0])
            .flat_map(|z| plane(z).map(|(p, pc)| block(p, pc)))
            .collect()
    };
    // Match each shape of block once, then replace each block by its shape's number (`u32::MAX` if absent)
    let mut shapes: Vec<u32> = blocks.iter().copied().filter(|&b| b != 0).collect();
    shapes.sort_unstable();
    shapes.dedup();
    let patterns: Vec<Pattern> = if parallel {
        shapes.par_iter().map(|&b| match_block(tables, b)).collect()
    } else {
        shapes.iter().map(|&b| match_block(tables, b)).collect()
    };
    let number = |b: &mut u32| {
        *b = if *b == 0 {
            u32::MAX
        } else {
            shapes.binary_search(b).expect("a shape") as u32
        };
    };
    if parallel {
        blocks.par_iter_mut().for_each(number);
    } else {
        blocks.iter_mut().for_each(number);
    }
    let local_shift: [isize; N_LOCAL] = std::array::from_fn(|l| {
        let o = offsets_of(l);
        (0..3)
            .map(|a| o[a] as isize * geometry.strides[a] as isize)
            .sum()
    });
    // Write the matching, listing the critical cells as (pixel's rank and index, index in the block's order, cell)
    let state: Vec<AtomicU32> = (0..geometry.n_cells)
        .map(|_| AtomicU32::new(ABSENT))
        .collect();
    let fill = |out: &mut Vec<(u64, u8, u32)>, (p, pc): (usize, [usize; 3])| {
        if blocks[p] == u32::MAX {
            return;
        }
        let pattern = &patterns[blocks[p] as usize];
        let centre = (0..3)
            .map(|a| (2 * pc[a] + 1) * geometry.strides[a])
            .sum::<usize>() as isize;
        let cell_of = |l: u8| (centre + local_shift[l as usize]) as u32;
        for l in 0..N_LOCAL as u8 {
            let pl = pattern.partner[l as usize];
            if pl != NOT_IN_BLOCK {
                let s = if pl == l {
                    CRITICAL
                } else {
                    cell_of(pl) | u32::from(tables.mask[pl as usize]) << CELL_BITS
                };
                state[cell_of(l) as usize].store(s, Relaxed);
            }
        }
        let key = (u64::from(pixel_ranks[p]) << 32) | p as u64;
        out.extend(
            pattern
                .critical
                .iter()
                .enumerate()
                .map(|(i, &l)| (key, i as u8, cell_of(l))),
        );
    };
    let mut critical: Vec<(u64, u8, u32)> = if parallel {
        (0..shape[0])
            .into_par_iter()
            .fold(Vec::new, |mut out, z| {
                plane(z).for_each(|pixel| fill(&mut out, pixel));
                out
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            })
    } else {
        let mut out = vec![];
        (0..shape[0]).for_each(|z| plane(z).for_each(|pixel| fill(&mut out, pixel)));
        out
    };
    if parallel {
        critical.par_sort_unstable();
    } else {
        critical.sort_unstable();
    }
    // Number those wanted
    let mut wanted = Vec::with_capacity(critical.len());
    for (key, _, c) in critical {
        if geometry.dimension(c as usize) <= top {
            state[c as usize].store(CRITICAL | wanted.len() as u32, Relaxed);
            wanted.push((c, (key >> 32) as u32));
        } else {
            state[c as usize].store(UNWANTED, Relaxed);
        }
    }
    (state, wanted)
}

/// The Morse boundaries of the critical cells of each dimension `1..=top` (in their order), as compressed columns of
/// critical indices.
fn morse_boundaries(
    geometry: &Geometry,
    state: &[AtomicU32],
    critical: &[(u32, u32)],
    top: usize,
    parallel: bool,
) -> Vec<(Vec<usize>, Vec<u32>)> {
    let of_dimension = |k: usize| -> (Vec<usize>, Vec<u32>) {
        let mut flows = Flows {
            spans: vec![(0, 0)],
            arena: vec![],
            scratch: vec![],
            stack: vec![],
        };
        let mut indptr = vec![0];
        let mut indices = vec![];
        let mut boundary = vec![];
        let mut buffer = [(0, 0); 6];
        for &(c, _) in critical {
            let mask = geometry.mask(c as usize);
            if mask.count_ones() as usize != k {
                continue;
            }
            boundary.clear();
            let n = geometry.faces(c, mask, &mut buffer);
            for &(f, f_mask) in &buffer[..n] {
                flows.flow_of(geometry, state, f, f_mask, &mut boundary);
            }
            cancel_into(&mut boundary, &mut indices);
            indptr.push(indices.len());
        }
        (indptr, indices)
    };
    if parallel {
        (1..=top).into_par_iter().map(of_dimension).collect()
    } else {
        (1..=top).map(of_dimension).collect()
    }
}

/// The flows of cells of one dimension, memoised as spans of `arena` (span 0 empty, for cells matched with a face).
struct Flows {
    spans: Vec<(u32, u32)>,
    arena: Vec<u32>,
    scratch: Vec<u32>,
    stack: Vec<(u32, u32, bool)>,
}

impl Flows {
    /// Append the flow of a known state to `out`, or return false if it isn't known.
    #[inline]
    fn read(&self, s: u32, out: &mut Vec<u32>) -> bool {
        match s & TAG {
            FLOW => {
                let (start, len) = self.spans[(s & !TAG) as usize];
                out.extend_from_slice(&self.arena[start as usize..(start + len) as usize]);
                true
            }
            CRITICAL => {
                debug_assert_ne!(s, UNWANTED);
                out.push(s & !TAG);
                true
            }
            _ => {
                debug_assert_ne!(s, ABSENT);
                false
            }
        }
    }

    /// Append the flow of `f` (a face of a cell of the dimension above, spanning the axes in `f_mask`) to `out`,
    /// computing it if need be.
    fn flow_of(
        &mut self,
        geometry: &Geometry,
        state: &[AtomicU32],
        f: u32,
        f_mask: u32,
        out: &mut Vec<u32>,
    ) {
        let known = |s: u32| s & TAG == FLOW || s & TAG == CRITICAL;
        let get = |c: u32| state[c as usize].load(Relaxed);
        if self.read(get(f), out) {
            return;
        }
        let mut buffer = [(0, 0); 6];
        self.stack.clear();
        self.stack.push((f, f_mask, false));
        while let Some((g, g_mask, expanded)) = self.stack.pop() {
            let s = get(g);
            if known(s) {
                continue;
            }
            let (partner, partner_mask) = (s & CELL, s >> CELL_BITS);
            // Matched with a face: no flow
            if partner_mask.count_ones() < g_mask.count_ones() {
                state[g as usize].store(FLOW, Relaxed);
                continue;
            }
            let n = geometry.faces(partner, partner_mask, &mut buffer);
            if expanded {
                let mut scratch = std::mem::take(&mut self.scratch);
                scratch.clear();
                for &(h, _) in &buffer[..n] {
                    if h != g {
                        let found = self.read(get(h), &mut scratch);
                        debug_assert!(found);
                    }
                }
                let span = cancel_into(&mut scratch, &mut self.arena);
                self.scratch = scratch;
                self.spans.push(span);
                state[g as usize].store(FLOW | (self.spans.len() - 1) as u32, Relaxed);
            } else {
                self.stack.push((g, g_mask, true));
                for &(h, h_mask) in &buffer[..n] {
                    if h != g && !known(get(h)) {
                        self.stack.push((h, h_mask, false));
                    }
                }
            }
        }
        let found = self.read(get(f), out);
        debug_assert!(found);
    }
}
