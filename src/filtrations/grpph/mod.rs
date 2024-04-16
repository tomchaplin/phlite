use std::{cmp::Reverse, marker::PhantomData};

use ordered_float::NotNan;
use rustc_hash::{FxHashMap, FxHashSet};

// TODO: Restructure this code and give everything better names

use crate::{
    columns::ColumnEntry,
    fields::NonZeroCoefficient,
    matrices::{
        BasisElement, ColBasis, HasColBasis, HasRowFiltration, MatrixOracle, SplitByDimension,
    },
    PhliteError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathHomCell {
    Node(u16),
    Edge(u16, u16),
    TwoCell(PathHom2Cell),
}

impl BasisElement for PathHomCell {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathHom2Cell {
    DoubleEdge(u16, u16),            // (a, b), -> aba
    DirectedTriangle(u16, u16, u16), // (a, b, c), -> abc
    LongSquare(u16, u16, u16, u16),  // (a, b, c, d), -> acd - abd
}

type PrimaryBridgeMap = FxHashMap<(u16, u16), u16>;
type DigraphFiltration = FxHashMap<(u16, u16), NotNan<f64>>;
type DigraphEdgeSet = FxHashSet<(u16, u16)>;

fn two_path_time(filtration: &DigraphFiltration, a: &u16, b: &u16, c: &u16) -> Option<NotNan<f64>> {
    let ab_time = filtration.get(&(*a, *b))?;
    let bc_time = filtration.get(&(*b, *c))?;
    Some(*ab_time.max(bc_time))
}

// Returns two values
// 1. The two cell induced by the path abc (if it exists) along with its filtration time
// 2. Whether abc is a primary bridge - thus we need to enumerate all of those cells too
fn two_path_to_two_cell_in_filtration(
    bridge_map: &PrimaryBridgeMap,
    filtration: &DigraphFiltration,
    a: u16,
    b: u16,
    c: u16,
) -> (Option<(PathHom2Cell, NotNan<f64>)>, bool) {
    // Path never appears if this gives None
    let Some(abc_time) = two_path_time(&filtration, &a, &b, &c) else {
        return (None, false);
    };

    if a == c {
        return (Some((PathHom2Cell::DoubleEdge(a, b), abc_time)), false);
    }

    if let Some(ac_time) = filtration.get(&(a, c)) {
        if abc_time >= *ac_time {
            (
                Some((PathHom2Cell::DirectedTriangle(a, b, c), abc_time)),
                false,
            )
        } else {
            // The path abc enters strictly before ac so there must be a primary bridge
            let primary_bridge = bridge_map.get(&(a, c)).unwrap();

            if b == *primary_bridge {
                // This is the collapsing directed triangle
                (
                    Some((PathHom2Cell::DirectedTriangle(a, b, c), *ac_time)),
                    true,
                )
                // TODO: Also yield all the long squares!
            } else {
                (
                    Some((PathHom2Cell::LongSquare(a, *primary_bridge, b, c), abc_time)),
                    false,
                )
            }
        }
    } else {
        // The path abc enters strictly before ac so there must be a primary bridge
        let primary_bridge = bridge_map.get(&(a, c)).unwrap();

        if b == *primary_bridge {
            // ac_time is infinite so collapse never appears
            (None, true)
            // TODO: Also yield all the long squares!
        } else {
            (
                Some((PathHom2Cell::LongSquare(a, *primary_bridge, b, c), abc_time)),
                false,
            )
        }
    }
}

// Produces the triangle 2-cells of the form sbt
fn produce_triangle_base_coboundary<'a, CF: NonZeroCoefficient>(
    bridge_map: &'a PrimaryBridgeMap,
    filtration: &'a DigraphFiltration,
    n_vertices: u16,
    s: u16,
    t: u16,
) -> Option<impl Iterator<Item = (CF, PathHom2Cell, NotNan<f64>)> + 'a> {
    // First off, we only get columns of the time if (s, t) appears in finite time
    let st_time = filtration.get(&(s, t))?;

    //If there are bridges of the form s->b->t then we must had add a column of the form sb_{st}t at st_time
    let primary_bridge = bridge_map.get(&(s, t));
    let bridge_collapse_col = primary_bridge.into_iter().map(move |pb| {
        (
            CF::one().additive_inverse(),
            PathHom2Cell::DirectedTriangle(s, *pb, t),
            *st_time,
        )
    });

    // Then we add triangles that appear after st_time
    let conventional_triangles = (0..n_vertices).filter_map(move |b| {
        let sbt_time = two_path_time(filtration, &s, &b, &t)?;
        if sbt_time < *st_time {
            // This would result in a bridge and maybe a long square
            None
        } else {
            // We get a directed triangle
            Some((
                CF::one().additive_inverse(),
                PathHom2Cell::DirectedTriangle(s, b, t),
                sbt_time,
            ))
        }
    });

    Some(bridge_collapse_col.chain(conventional_triangles))
}

// It is given that abc consitutes a primary bridge
// Produce all of the long squares based at abc
// TODO: Is it better to store a list of all of the bridges?
fn produce_long_squares_with_primary_bridge<'a>(
    filtration: &'a DigraphFiltration,
    n_vertices: u16,
    a: u16,
    b: u16,
    c: u16,
) -> impl Iterator<Item = (PathHom2Cell, NotNan<f64>)> + 'a {
    // WARNING: This fails if anything has filtration time f64::MAX
    let ac_time = filtration
        .get(&(a, c))
        .copied()
        .unwrap_or(NotNan::new(f64::MAX).unwrap());
    (0..n_vertices)
        .filter(move |&i| i != a)
        .filter(move |&i| i != b)
        .filter(move |&i| i != c)
        .filter_map(move |i| {
            // Look for a finite time two paths a -> i -> c
            let aic_time = two_path_time(filtration, &a, &i, &c)?;
            // Only return a long square if they appear before shortcut a -> c
            if aic_time >= ac_time {
                None
            } else {
                Some((PathHom2Cell::LongSquare(a, b, i, c), aic_time))
            }
        })
}

pub fn produce_edge_total_coboundary<'a, CF: NonZeroCoefficient>(
    bridge_map: &'a PrimaryBridgeMap,
    filtration: &'a DigraphFiltration,
    n_vertices: u16,
    s: u16,
    t: u16,
) -> impl Iterator<Item = (CF, PathHom2Cell, NotNan<f64>)> + 'a {
    let part_1 = (0..n_vertices).filter(move |&i| i != s).flat_map(move |i| {
        let (cell, is_pb) = two_path_to_two_cell_in_filtration(bridge_map, filtration, i, s, t);
        let cell = cell
            .into_iter()
            .map(|(cell, f_time)| (CF::one(), cell, f_time));
        // Chain on generators correspondiong to primary bridge
        let lswpb_cols = is_pb
            .then(|| produce_long_squares_with_primary_bridge(filtration, n_vertices, i, s, t))
            .into_iter()
            .flatten()
            .map(|(cell, f_time)| (CF::one().additive_inverse(), cell, f_time));
        cell.chain(lswpb_cols)
    });

    let part_2 = produce_triangle_base_coboundary(bridge_map, filtration, n_vertices, s, t)
        .into_iter()
        .flatten();

    let part_3 = (0..n_vertices).filter(move |&i| i != t).flat_map(move |i| {
        let (cell, is_pb) = two_path_to_two_cell_in_filtration(bridge_map, filtration, s, t, i);
        let cell = cell
            .into_iter()
            .map(|(cell, f_time)| (CF::one(), cell, f_time));
        // Chain on generators correspondiong to primary bridge
        let lswpb_cols = is_pb
            .then(|| produce_long_squares_with_primary_bridge(filtration, n_vertices, s, t, i))
            .into_iter()
            .flatten()
            .map(|(cell, f_time)| (CF::one().additive_inverse(), cell, f_time));
        cell.chain(lswpb_cols)
    });

    // TODO: There is probably a smarter way to do this so that we don't keep looking up primary bridges

    part_1.chain(part_2).chain(part_3)
}

pub fn produce_node_total_coboundary<'a, CF: NonZeroCoefficient>(
    filtration: &'a DigraphFiltration,
    edge_set: &'a DigraphEdgeSet,
    n_vertices: u16,
    s: u16,
) -> impl Iterator<Item = (CF, PathHomCell, NotNan<f64>)> + 'a {
    let outgoing = (0..n_vertices).filter_map(move |j| {
        if edge_set.contains(&(s, j)) {
            // This is the grounding - edges in graph are born at 0
            Some((
                CF::one().additive_inverse(),
                PathHomCell::Edge(s, j),
                unsafe { NotNan::new_unchecked(0.0) },
            ))
        } else if let Some(time) = filtration.get(&(s, j)) {
            Some((CF::one().additive_inverse(), PathHomCell::Edge(s, j), *time))
        } else {
            None
        }
    });

    let incoming = (0..n_vertices).filter_map(move |j| {
        if edge_set.contains(&(j, s)) {
            // This is the grounding - edges in graph are born at 0
            Some((CF::one(), PathHomCell::Edge(j, s), unsafe {
                NotNan::new_unchecked(0.0)
            }))
        } else if let Some(time) = filtration.get(&(j, s)) {
            Some((CF::one(), PathHomCell::Edge(j, s), *time))
        } else {
            None
        }
    });
    outgoing.chain(incoming)
}

// TODO: Compute this in parallel
pub fn build_primary_bridge_map(
    filtration: &DigraphFiltration,
    n_vertices: u16,
) -> PrimaryBridgeMap {
    let mut map = FxHashMap::default();
    for s in 0..n_vertices {
        for t in 0..n_vertices {
            // We have a unique choice of primary bridge since we look for min by (time, bridge_vertex_index)
            let primary_bridge = (0..n_vertices)
                .filter_map(|i| Some((two_path_time(filtration, &s, &i, &t)?, i)))
                .min();

            if let Some((time, bridge_vertex)) = primary_bridge {
                // This only forms a bridge if its arrival time is before the arrival of the edge s->t
                if let Some(st_time) = filtration.get(&(s, t)) {
                    if time >= *st_time {
                        continue;
                    }
                }
                map.insert((s, t), bridge_vertex);
            }
        }
    }
    map
}

// TODO: Make this generic over (edge_set, n_vertices, filtration) which is a digraph with filtration
pub struct GrPPHCoboundary<CF> {
    filtration: DigraphFiltration,
    bridge_map: PrimaryBridgeMap,
    edge_set: DigraphEdgeSet,
    n_vertices: u16,
    phantom: PhantomData<CF>,
    basis: Vec<Vec<(Reverse<NotNan<f64>>, PathHomCell)>>,
}

fn build_basis(
    filtration: &DigraphFiltration,
    edge_set: &DigraphEdgeSet,
    n_vertices: u16,
) -> Vec<Vec<(Reverse<NotNan<f64>>, PathHomCell)>> {
    let degree_0: Vec<_> = (0..n_vertices)
        .map(|i| {
            (
                Reverse(unsafe { NotNan::new_unchecked(0.0) }),
                PathHomCell::Node(i),
            )
        })
        .collect();
    let mut degree_1: Vec<_> = edge_set
        .iter()
        .map(|(i, j)| {
            (
                Reverse(unsafe { NotNan::new_unchecked(0.0) }),
                PathHomCell::Edge(*i, *j),
            )
        })
        .collect();
    for (edge, time) in filtration.iter() {
        if edge_set.contains(edge) {
            continue;
        }
        degree_1.push((Reverse(*time), PathHomCell::Edge(edge.0, edge.1)))
    }
    degree_1.sort_unstable();
    vec![degree_0, degree_1]
}

impl ColBasis for Vec<(Reverse<NotNan<f64>>, PathHomCell)> {
    type ElemT = PathHomCell;

    fn element(&self, index: usize) -> Self::ElemT {
        self[index].1
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl ColBasis for Vec<Vec<(Reverse<NotNan<f64>>, PathHomCell)>> {
    type ElemT = PathHomCell;

    fn element(&self, index: usize) -> Self::ElemT {
        let mut working = index;
        let mut dim = 0;
        while working >= self[dim].size() {
            working -= self[dim].size();
            dim += 1;
        }
        self[dim].element(working)
    }

    fn size(&self) -> usize {
        self.iter().map(|basis| basis.size()).sum()
    }
}

impl SplitByDimension for Vec<Vec<(Reverse<NotNan<f64>>, PathHomCell)>> {
    type SubBasisT = Vec<(Reverse<NotNan<f64>>, PathHomCell)>;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        &self[dimension]
    }
}

impl<CF: NonZeroCoefficient> GrPPHCoboundary<CF> {
    pub fn build(filtration: DigraphFiltration, edge_set: DigraphEdgeSet, n_vertices: u16) -> Self {
        let bridge_map = build_primary_bridge_map(&filtration, n_vertices);
        let basis = build_basis(&filtration, &edge_set, n_vertices);
        Self {
            filtration,
            bridge_map,
            edge_set,
            n_vertices,
            basis,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> MatrixOracle for GrPPHCoboundary<CF> {
    type CoefficientField = CF;

    type ColT = PathHomCell;

    type RowT = PathHomCell;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, crate::PhliteError>
    {
        let boundary: Box<dyn Iterator<Item = (Self::CoefficientField, Self::RowT)>> = match col {
            PathHomCell::Node(s) => Box::new(
                produce_node_total_coboundary(&self.filtration, &self.edge_set, self.n_vertices, s)
                    .map(|(coeff, cell, _time)| (coeff, cell)),
            ),
            PathHomCell::Edge(s, t) => Box::new(
                produce_edge_total_coboundary(
                    &self.bridge_map,
                    &self.filtration,
                    self.n_vertices,
                    s,
                    t,
                )
                .map(|(coeff, cell, _time)| (coeff, PathHomCell::TwoCell(cell))),
            ),
            PathHomCell::TwoCell(_) => return Err(PhliteError::NotInDomain),
        };
        Ok(boundary)
    }
}

impl<CF: NonZeroCoefficient> HasColBasis for GrPPHCoboundary<CF> {
    type BasisT = Vec<Vec<(Reverse<NotNan<f64>>, PathHomCell)>>;

    fn basis(&self) -> &Self::BasisT {
        &self.basis
    }
}

impl<CF: NonZeroCoefficient> HasRowFiltration for GrPPHCoboundary<CF> {
    type FiltrationT = Reverse<NotNan<f64>>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        match row {
            PathHomCell::Node(_s) => Ok(Reverse(unsafe { NotNan::new_unchecked(0.0) })),
            PathHomCell::Edge(s, t) => {
                // This is the grounding - edges in graph are born at 0
                if self.edge_set.contains(&(s, t)) {
                    Ok(Reverse(unsafe { NotNan::new_unchecked(0.0) }))
                } else if let Some(time) = self.filtration.get(&(s, t)) {
                    Ok(Reverse(*time))
                } else {
                    Err(PhliteError::NotInCodomain)
                }
            }
            PathHomCell::TwoCell(cell) => match cell {
                PathHom2Cell::DoubleEdge(a, b) => Ok(Reverse(
                    two_path_time(&self.filtration, &a, &b, &a).unwrap(),
                )),
                PathHom2Cell::DirectedTriangle(a, b, c) => {
                    let abc_time = two_path_time(&self.filtration, &a, &b, &c).unwrap();
                    let ac_time = self.filtration.get(&(a, c)).unwrap();
                    let arrival_time = abc_time.max(*ac_time);
                    Ok(Reverse(arrival_time))
                }
                PathHom2Cell::LongSquare(a, b, c, d) => {
                    let abd_time = two_path_time(&self.filtration, &a, &b, &d).unwrap();
                    let acd_time = two_path_time(&self.filtration, &a, &c, &d).unwrap();
                    let arrival_time = abd_time.max(acd_time);
                    Ok(Reverse(arrival_time))
                }
            },
        }
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = Result<ColumnEntry<Self>, PhliteError>>, PhliteError> {
        let boundary: Box<dyn Iterator<Item = Result<ColumnEntry<Self>, PhliteError>>> = match col {
            PathHomCell::Node(s) => Box::new(
                produce_node_total_coboundary(&self.filtration, &self.edge_set, self.n_vertices, s)
                    .map(|(coeff, cell, time)| {
                        Ok(ColumnEntry {
                            filtration_value: Reverse(time),
                            row_index: cell,
                            coeff,
                        })
                    }),
            ),
            PathHomCell::Edge(s, t) => Box::new(
                produce_edge_total_coboundary(
                    &self.bridge_map,
                    &self.filtration,
                    self.n_vertices,
                    s,
                    t,
                )
                .map(|(coeff, cell, time)| {
                    Ok(ColumnEntry {
                        filtration_value: Reverse(time),
                        row_index: PathHomCell::TwoCell(cell),
                        coeff,
                    })
                }),
            ),
            PathHomCell::TwoCell(_) => return Err(PhliteError::NotInDomain),
        };
        Ok(boundary)
    }
}

#[cfg(test)]
mod tests {

    use std::mem;

    use ordered_float::NotNan;
    use rustc_hash::{FxHashMap, FxHashSet};

    use crate::{
        fields::{Z2, Z3},
        filtrations::grpph::{produce_edge_total_coboundary, PathHomCell},
        matrices::HasRowFiltration,
        reduction::ClearedReductionMatrix,
    };

    use super::{build_primary_bridge_map, GrPPHCoboundary};

    #[test]
    fn test_coboundary() {
        let mut filtration = FxHashMap::default();
        for i in 0..5 {
            filtration.insert((0, i + 1), NotNan::new(1.0).unwrap());
            filtration.insert((i + 1, 11), NotNan::new(1.0).unwrap());
        }
        filtration.insert((0, 11), NotNan::new(2.0).unwrap());
        for i in 0..5 {
            filtration.insert((0, i + 6), NotNan::new(3.0).unwrap());
            filtration.insert((i + 6, 11), NotNan::new(3.0).unwrap());
        }
        filtration.insert((11, 0), NotNan::new(2.0).unwrap());

        let bm = build_primary_bridge_map(&filtration, 12);

        println!("{bm:?}");

        let cob1 = produce_edge_total_coboundary::<Z3>(&bm, &filtration, 12, 0, 11);
        for tup in cob1 {
            println!("{tup:?}");
        }

        println!("{}", mem::size_of::<PathHomCell>());
    }

    #[test]
    fn test_grpph_cycle() {
        let n = 30;
        let mut filtration = FxHashMap::default();
        let mut edge_set = FxHashSet::default();
        for i in 0..n {
            edge_set.insert((i, (i + 1) % n));
            for j in 1..n {
                filtration.insert((i, (i + j) % n), NotNan::new(j as f64).unwrap());
            }
        }

        let d = GrPPHCoboundary::<Z2>::build(filtration, edge_set, n);

        let (_v, diagram) = ClearedReductionMatrix::build_with_diagram(&d, 0..=1);

        let mut count = 0;

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = d.filtration_value(*idx).unwrap().0;
            println!(" birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let idx_tup = (tup.1, tup.0);
            let birth_f = d.filtration_value(tup.1).unwrap().0;
            let death_f = d.filtration_value(tup.0).unwrap().0;
            if death_f == birth_f {
                continue;
            }
            count += 1;
            println!(" pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }
        println!("{count}");

        // Initial graph is a a single cycle so only one element in barcode
        assert_eq!(count, 1);
        // Only essential cycle is the initial component
        assert_eq!(diagram.essential.len(), 1);
    }

    #[test]
    fn test_grpph_complete() {
        let n = 30;
        let mut filtration = FxHashMap::default();
        let mut edge_set = FxHashSet::default();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                edge_set.insert((i, j));
                filtration.insert((i, j), NotNan::new(1.0).unwrap());
            }
        }

        let d = GrPPHCoboundary::<Z2>::build(filtration, edge_set, n);

        let (_v, diagram) = ClearedReductionMatrix::build_with_diagram(&d, 0..=1);

        let mut count = 0;

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = d.filtration_value(*idx).unwrap().0;
            println!(" birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let idx_tup = (tup.1, tup.0);
            let birth_f = d.filtration_value(tup.1).unwrap().0;
            let death_f = d.filtration_value(tup.0).unwrap().0;
            if death_f == birth_f {
                continue;
            }
            count += 1;
            println!(" pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }
        println!("{count}");

        // Initial graph is complete and single component so Euler charcteristic tells us the circuit rank
        assert_eq!(count, n * (n - 1) - n + 1);
        // Only essential cycle is the initial component
        assert_eq!(diagram.essential.len(), 1);
    }
}
