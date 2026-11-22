//! Six-packs of persistence barcodes for maps of filtrations: for a filtration-preserving chain map `f: L -> K`
//! between filtered complexes, the barcodes of `L` (the *domain*), `K` (the *codomain*), the *image*, *kernel* and
//! *cokernel* of `f_*: H(L_t) -> H(K_t)`, and the *relative* homology of the map (of its mapping cone, which is that
//! of `(K, L)` for an inclusion), each bar with a representative.
//!
//! A map is a [`FilteredChainMap`]: two boundary oracles (any of phlite's, lazy or explicit), their cells, and the
//! image of each cell of `L`, a chain of cells of `K` no later than it. Its [`six_pack`](FilteredChainMap::six_pack)
//! reduces the inclusion of `L` in the map's (algebraic) mapping cylinder, a [`MappingCylinder`] oracle, by
//! [`inclusion_six_pack`]: the cylinder has
//!
//! * a cell `Domain(σ)` for each cell `σ` of `L`, with boundary `Domain(∂σ)`,
//! * a cell `Codomain(τ)` for each cell `τ` of `K`, with boundary `Codomain(∂τ)`,
//! * a cell `Cylinder(σ)` (`σ × [0, 1]`, one dimension up) for each `σ` of `L`, with boundary
//!   `Codomain(f σ) - Domain(σ) - Cylinder(∂σ)`,
//!
//! `Domain(σ)` and `Cylinder(σ)` at `σ`'s value, `Codomain(τ)` at `τ`'s. At every value `t` it is the mapping
//! cylinder of `f_t: L_t -> K_t`, which deformation retracts onto `K_t` (by `Domain(σ) -> f σ`, `Cylinder(σ) -> 0`),
//! so `H(L_t) -> H(Cyl_t)` is `f_*`, and `H(Cyl_t, L_t)` is the homology of the mapping cone. Representatives of the
//! codomain and cokernel are carried back to `K` by the retraction; those of the relative barcode are chains of the
//! cone, on [`ConeCell`]s.
//!
//! An inclusion of a subcomplex needs no cylinder: [`Inclusion`] overrides `six_pack` to reduce the pair directly.

use std::fmt::Debug;

use crate::columns::{ColumnCoefficient, ColumnFiltration, ColumnRepresentation, ColumnRow};
use crate::fields::{Invertible, NonZeroCoefficient};
use crate::image_kernel::{consolidated, inclusion_six_pack, CellBar, CellSixPack, Restricted};
use crate::matrices::{BasisElement, HasRowFiltration, MatrixOracle};

/// A filtration-preserving chain map `f: L -> K` between filtered complexes, given by boundary oracles.
///
/// `apply` must commute with the boundaries (`∂ f = f ∂`) and never make a cell earlier: every cell of `f σ` has a
/// value no later than `σ`'s. Neither is checked here.
pub trait FilteredChainMap: Sync {
    /// The cells of `L`
    type DomainCell: ColumnRow + Debug + 'static;
    /// The cells of `K`
    type CodomainCell: ColumnRow + Debug + 'static;
    /// The coefficients
    type Field: Invertible + ColumnCoefficient + 'static;
    /// The filtration values
    type Value: ColumnFiltration;
    /// The boundary of `L`
    type Domain: MatrixOracle<
            CoefficientField = Self::Field,
            ColT = Self::DomainCell,
            RowT = Self::DomainCell,
        > + HasRowFiltration<FiltrationT = Self::Value>
        + Clone
        + Send
        + Sync;
    /// The boundary of `K`
    type Codomain: MatrixOracle<
            CoefficientField = Self::Field,
            ColT = Self::CodomainCell,
            RowT = Self::CodomainCell,
        > + HasRowFiltration<FiltrationT = Self::Value>
        + Clone
        + Send
        + Sync;

    fn domain(&self) -> Self::Domain;
    fn codomain(&self) -> Self::Codomain;
    /// The cells of `L`, by dimension
    fn domain_cells(&self) -> Vec<Vec<DomainCell<Self>>>;
    /// The cells of `K`, by dimension
    fn codomain_cells(&self) -> Vec<Vec<CodomainCell<Self>>>;
    /// `f σ`, a chain of cells of `K` of `σ`'s dimension (zero coefficients left out)
    fn apply(&self, cell: DomainCell<Self>) -> Vec<(Coefficient<Self>, CodomainCell<Self>)>;

    /// The six-pack of the map (see the [module docs](self)) in dimensions up to `max_dim` (all if `None`; only cells
    /// up to dimension `max_dim + 1` are used), with representatives if `representatives`, reducing with the column
    /// representation `columns`.
    fn six_pack<R: ColumnRepresentation>(
        &self,
        columns: R,
        representatives: bool,
        max_dim: Option<usize>,
    ) -> MapSixPack<Self>
    where
        Self: Sized,
    {
        six_pack_by_cylinder(self, columns, representatives, max_dim)
    }
}

type DomainCell<Map> = <Map as FilteredChainMap>::DomainCell;
type CodomainCell<Map> = <Map as FilteredChainMap>::CodomainCell;
type Coefficient<Map> = <Map as FilteredChainMap>::Field;
type Value<Map> = <Map as FilteredChainMap>::Value;
/// A map's six-pack.
pub type MapSixPack<Map> =
    SixPack<DomainCell<Map>, CodomainCell<Map>, Value<Map>, Coefficient<Map>>;

/// A cell of a mapping cylinder.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CylinderCell<L, K> {
    Domain(L),
    /// `σ × [0, 1]`
    Cylinder(L),
    Codomain(K),
}

impl<L: Ord + Clone, K: Ord + Clone> BasisElement for CylinderCell<L, K> {}

/// A cell of a mapping cone (what the relative representatives are chains of): a cell of `K`, or the cone `Cone(σ)` on
/// a cell `σ` of `L`, one dimension up. For an inclusion, only cells of `K` outside `L` appear.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConeCell<L, K> {
    Codomain(K),
    Cone(L),
}

/// A bar of a six-pack: its dimension, birth and death values (`None` if essential) and, if asked for, a
/// representative (`(coefficient, cell)` pairs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SixPackBar<F, C, CF> {
    pub dimension: usize,
    pub birth: F,
    pub death: Option<F>,
    pub representative: Option<Vec<(CF, C)>>,
}

/// The six barcodes of a map `f: L -> K` (bars of length zero left out), with representatives:
/// chains of `L` for the domain, image and kernel; of `K` for the codomain and cokernel; of the mapping cone for the
/// relative barcode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SixPack<L, K, F, CF> {
    pub domain: Vec<SixPackBar<F, L, CF>>,
    pub codomain: Vec<SixPackBar<F, K, CF>>,
    /// Represented by cycles of `L` (whose images represent the image classes in `K`)
    pub image: Vec<SixPackBar<F, L, CF>>,
    /// Represented by cycles of `L` that map to boundaries; a bar's dimension is its classes'
    pub kernel: Vec<SixPackBar<F, L, CF>>,
    /// Represented by cycles of `K`
    pub cokernel: Vec<SixPackBar<F, K, CF>>,
    pub relative: Vec<SixPackBar<F, ConeCell<L, K>, CF>>,
}

impl<F, C, CF> SixPackBar<F, C, CF> {
    /// The same bar with values and cells mapped, or `None` if that makes it of length zero.
    pub fn map<G: PartialEq, D>(
        self,
        value: impl Fn(F) -> G,
        cell: impl Fn(C) -> D,
    ) -> Option<SixPackBar<G, D, CF>> {
        let (birth, death) = (value(self.birth), self.death.map(&value));
        if death.as_ref().is_some_and(|d| *d == birth) {
            return None;
        }
        Some(SixPackBar {
            dimension: self.dimension,
            birth,
            death,
            representative: self
                .representative
                .map(|rep| rep.into_iter().map(|(x, c)| (x, cell(c))).collect()),
        })
    }
}

impl<L, K, F, CF> SixPack<L, K, F, CF> {
    /// The same six-pack with values and cells mapped (by `value`, `domain` for cells of `L`, `codomain` for those of
    /// `K`), leaving out bars this makes of length zero: for example from cells numbered by their position in the
    /// filtration (unique) to their filtration values (which may tie).
    pub fn map<G: PartialEq, L2, K2>(
        self,
        value: impl Fn(F) -> G,
        domain: impl Fn(L) -> L2,
        codomain: impl Fn(K) -> K2,
    ) -> SixPack<L2, K2, G, CF> {
        let cone = |c: ConeCell<L, K>| match c {
            ConeCell::Codomain(k) => ConeCell::Codomain(codomain(k)),
            ConeCell::Cone(l) => ConeCell::Cone(domain(l)),
        };
        SixPack {
            domain: self
                .domain
                .into_iter()
                .filter_map(|b| b.map(&value, &domain))
                .collect(),
            codomain: self
                .codomain
                .into_iter()
                .filter_map(|b| b.map(&value, &codomain))
                .collect(),
            image: self
                .image
                .into_iter()
                .filter_map(|b| b.map(&value, &domain))
                .collect(),
            kernel: self
                .kernel
                .into_iter()
                .filter_map(|b| b.map(&value, &domain))
                .collect(),
            cokernel: self
                .cokernel
                .into_iter()
                .filter_map(|b| b.map(&value, &codomain))
                .collect(),
            relative: self
                .relative
                .into_iter()
                .filter_map(|b| b.map(&value, cone))
                .collect(),
        }
    }
}

/// Bars of one barcode with zero-length ones and those above `max_dim` left out, and representatives carried by `rep`.
fn carry<C, F: Eq, CF, D>(
    bars: Vec<CellBar<C, F, CF>>,
    max_dim: Option<usize>,
    rep: impl Fn(Vec<(CF, C)>) -> Vec<(CF, D)>,
) -> Vec<SixPackBar<F, D, CF>> {
    bars.into_iter()
        .filter(|bar| bar.death.as_ref().is_none_or(|(_, d)| *d != bar.birth.1))
        .filter(|bar| max_dim.is_none_or(|m| bar.dimension <= m))
        .map(|bar| SixPackBar {
            dimension: bar.dimension,
            birth: bar.birth.1,
            death: bar.death.map(|(_, d)| d),
            representative: bar.representative.map(&rep),
        })
        .collect()
}

/// The mapping cylinder of a [`FilteredChainMap`], as a boundary oracle (see the [module docs](self)).
pub struct MappingCylinder<'a, Map: FilteredChainMap> {
    map: &'a Map,
    domain: Map::Domain,
    codomain: Map::Codomain,
}

impl<'a, Map: FilteredChainMap> MappingCylinder<'a, Map> {
    pub fn new(map: &'a Map) -> Self {
        Self {
            map,
            domain: map.domain(),
            codomain: map.codomain(),
        }
    }

    /// Its cells by dimension, each in filtration order.
    pub fn cells(&self) -> Vec<Vec<CylinderCell<DomainCell<Map>, CodomainCell<Map>>>> {
        self.cells_up_to(None)
    }

    /// Its cells of dimension at most `top` (all if `None`) by dimension, each in filtration order.
    pub fn cells_up_to(
        &self,
        top: Option<usize>,
    ) -> Vec<Vec<CylinderCell<DomainCell<Map>, CodomainCell<Map>>>> {
        let (domain, codomain) = (self.map.domain_cells(), self.map.codomain_cells());
        let mut cells = vec![vec![]; (domain.len() + 1).max(codomain.len())];
        for (d, dim) in domain.into_iter().enumerate() {
            cells[d + 1].extend(dim.iter().cloned().map(CylinderCell::Cylinder));
            cells[d].extend(dim.into_iter().map(CylinderCell::Domain));
        }
        for (d, dim) in codomain.into_iter().enumerate() {
            cells[d].extend(dim.into_iter().map(CylinderCell::Codomain));
        }
        if let Some(top) = top {
            cells.truncate(top + 1);
        }
        for dim in &mut cells {
            dim.sort_by_cached_key(|c| (self.filtration_value(c.clone()), c.clone()));
        }
        cells
    }
}

impl<Map: FilteredChainMap> Clone for MappingCylinder<'_, Map> {
    fn clone(&self) -> Self {
        Self {
            map: self.map,
            domain: self.domain.clone(),
            codomain: self.codomain.clone(),
        }
    }
}

impl<Map: FilteredChainMap> MatrixOracle for MappingCylinder<'_, Map> {
    type CoefficientField = Coefficient<Map>;
    type ColT = CylinderCell<DomainCell<Map>, CodomainCell<Map>>;
    type RowT = Self::ColT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        let column: Vec<_> = match col {
            CylinderCell::Domain(l) => self
                .domain
                .column(l)
                .map(|(x, f)| (x, CylinderCell::Domain(f)))
                .collect(),
            CylinderCell::Codomain(k) => self
                .codomain
                .column(k)
                .map(|(x, f)| (x, CylinderCell::Codomain(f)))
                .collect(),
            CylinderCell::Cylinder(l) => {
                let minus_one = Coefficient::<Map>::one().additive_inverse();
                let mut column: Vec<_> = self
                    .map
                    .apply(l.clone())
                    .into_iter()
                    .map(|(x, k)| (x, CylinderCell::Codomain(k)))
                    .collect();
                column.push((minus_one, CylinderCell::Domain(l.clone())));
                column.extend(
                    self.domain
                        .column(l)
                        .map(|(x, f)| (x.additive_inverse(), CylinderCell::Cylinder(f))),
                );
                column
            }
        };
        column.into_iter()
    }
}

impl<Map: FilteredChainMap> HasRowFiltration for MappingCylinder<'_, Map> {
    type FiltrationT = Value<Map>;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        match row {
            CylinderCell::Domain(l) | CylinderCell::Cylinder(l) => self.domain.filtration_value(l),
            CylinderCell::Codomain(k) => self.codomain.filtration_value(k),
        }
    }
}

/// The six-pack of `map` by its mapping cylinder (the default [`FilteredChainMap::six_pack`]).
pub fn six_pack_by_cylinder<Map: FilteredChainMap, R: ColumnRepresentation>(
    map: &Map,
    columns: R,
    representatives: bool,
    max_dim: Option<usize>,
) -> MapSixPack<Map> {
    type Cell<Map> = CylinderCell<DomainCell<Map>, CodomainCell<Map>>;
    let cylinder = MappingCylinder::new(map);
    let cells = cylinder.cells_up_to(max_dim.map(|m| m + 1));
    let pack: CellSixPack<Cell<Map>, _, _> = inclusion_six_pack(
        cylinder,
        cells,
        |c: &Cell<Map>| matches!(c, CylinderCell::Domain(_)),
        columns,
        representatives,
    );
    let in_domain =
        |chain: Vec<(Coefficient<Map>, Cell<Map>)>| -> Vec<(Coefficient<Map>, DomainCell<Map>)> {
            chain
                .into_iter()
                .map(|(x, c)| match c {
                    CylinderCell::Domain(l) => (x, l),
                    other => unreachable!("a representative in L has a cell {other:?} outside it"),
                })
                .collect()
        };
    // The retraction onto K
    let retract = |chain: Vec<(Coefficient<Map>, Cell<Map>)>| {
        consolidated(chain.into_iter().flat_map(
            |(x, c)| -> Vec<(Coefficient<Map>, CodomainCell<Map>)> {
                match c {
                    CylinderCell::Domain(l) => {
                        map.apply(l).into_iter().map(|(y, k)| (x * y, k)).collect()
                    }
                    CylinderCell::Cylinder(_) => vec![],
                    CylinderCell::Codomain(k) => vec![(x, k)],
                }
            },
        ))
    };
    let in_cone = |chain: Vec<(Coefficient<Map>, Cell<Map>)>| {
        chain
            .into_iter()
            .map(|(x, c)| match c {
                CylinderCell::Cylinder(l) => (x, ConeCell::Cone(l)),
                CylinderCell::Codomain(k) => (x, ConeCell::Codomain(k)),
                CylinderCell::Domain(l) => {
                    unreachable!("a relative representative has a cell {l:?} of L")
                }
            })
            .collect()
    };
    SixPack {
        domain: carry(pack.domain, max_dim, in_domain),
        codomain: carry(pack.codomain, max_dim, retract),
        image: carry(pack.image, max_dim, in_domain),
        kernel: carry(pack.kernel, max_dim, in_domain),
        cokernel: carry(pack.cokernel, max_dim, retract),
        relative: carry(pack.relative, max_dim, in_cone),
    }
}

/// The inclusion of a subcomplex `L` (the cells for which `in_subcomplex` holds, closed under faces) in a filtered
/// complex `K`, given by its boundary oracle and its cells by dimension (each in filtration order). Its six-pack needs
/// no cylinder.
pub struct Inclusion<M: MatrixOracle, P> {
    pub boundary: M,
    pub cells: Vec<Vec<M::ColT>>,
    pub in_subcomplex: P,
}

impl<M, P> FilteredChainMap for Inclusion<M, P>
where
    M: MatrixOracle<RowT = <M as MatrixOracle>::ColT> + HasRowFiltration + Clone + Send + Sync,
    M::ColT: ColumnRow + Debug + 'static,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    P: Fn(&M::ColT) -> bool + Sync,
{
    type DomainCell = M::ColT;
    type CodomainCell = M::ColT;
    type Field = M::CoefficientField;
    type Value = M::FiltrationT;
    type Domain = Restricted<M>;
    type Codomain = M;

    fn domain(&self) -> Restricted<M> {
        Restricted(self.boundary.clone())
    }

    fn codomain(&self) -> M {
        self.boundary.clone()
    }

    fn domain_cells(&self) -> Vec<Vec<M::ColT>> {
        self.cells
            .iter()
            .map(|dim| {
                dim.iter()
                    .filter(|c| (self.in_subcomplex)(c))
                    .cloned()
                    .collect()
            })
            .collect()
    }

    fn codomain_cells(&self) -> Vec<Vec<M::ColT>> {
        self.cells.clone()
    }

    fn apply(&self, cell: M::ColT) -> Vec<(M::CoefficientField, M::ColT)> {
        vec![(M::CoefficientField::one(), cell)]
    }

    fn six_pack<R: ColumnRepresentation>(
        &self,
        columns: R,
        representatives: bool,
        max_dim: Option<usize>,
    ) -> MapSixPack<Self> {
        let mut cells = self.cells.clone();
        if let Some(m) = max_dim {
            cells.truncate(m + 2);
        }
        let pack = inclusion_six_pack(
            self.boundary.clone(),
            cells,
            &self.in_subcomplex,
            columns,
            representatives,
        );
        let same = |chain: Vec<(M::CoefficientField, M::ColT)>| chain;
        let in_cone = |chain: Vec<(M::CoefficientField, M::ColT)>| {
            chain
                .into_iter()
                .map(|(x, c)| (x, ConeCell::Codomain(c)))
                .collect()
        };
        SixPack {
            domain: carry(pack.domain, max_dim, same),
            codomain: carry(pack.codomain, max_dim, same),
            image: carry(pack.image, max_dim, same),
            kernel: carry(pack.kernel, max_dim, same),
            cokernel: carry(pack.cokernel, max_dim, same),
            relative: carry(pack.relative, max_dim, in_cone),
        }
    }
}
