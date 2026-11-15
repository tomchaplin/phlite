//! Choosing a reduction algorithm, independently of Python and of the pipeline it runs in.
//!
//! The Python `phlite.core.Algorithm` class is a builder for an [`AlgorithmSpec`]: the algorithm, and whichever options the
//! user set. Some options' best values depend on the matrix (see the benchmarks), so each pipeline (an explicit
//! complex, Vietoris–Rips, GrPPH, ...) has its own [`AlgorithmDefaults`], and options left unset take them when the
//! pipeline [resolves](AlgorithmSpec::resolve) the spec. The result, a [`ResolvedAlgorithm`], runs an
//! [`AlgorithmVisitor`] with the configured [`phlite::barcode::Algorithm`].
//!
//! Resolution validates everything, since a spec may come from another extension module (see
//! [`algorithm_from_py`](crate::algorithm_from_py)); the Python builders check what they can early, for clearer errors.

use phlite::{
    barcode::Algorithm,
    columns::{ColumnRepresentation, HeapColumns, MergedColumns, PivotColumns, SortedColumns},
    reduction::{Clearing, Standard, Storage},
};

/// The reduction algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// [`Standard`]
    Standard,
    /// [`Clearing`]
    Clearing,
    /// [`Clearing::parallel`]
    ParallelClearing,
}

impl Kind {
    /// `'standard'`, `'clearing'` or `'parallel-clearing'`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Clearing => "clearing",
            Self::ParallelClearing => "parallel-clearing",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, AlgorithmError> {
        match name {
            "standard" => Ok(Self::Standard),
            "clearing" => Ok(Self::Clearing),
            "parallel-clearing" => Ok(Self::ParallelClearing),
            other => Err(AlgorithmError(format!(
                "Unknown algorithm {other:?}, expected 'standard', 'clearing' or 'parallel-clearing'"
            ))),
        }
    }
}

/// The column representations, see [`phlite::columns::ColumnRepresentation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Columns {
    /// [`MergedColumns`]
    Merged,
    /// [`HeapColumns`]
    Heap,
    /// [`SortedColumns`]
    Sorted,
    /// [`PivotColumns`]
    Pivot,
}

impl Columns {
    /// `'merged'`, `'heap'`, `'sorted'` or `'pivot'`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Merged => "merged",
            Self::Heap => "heap",
            Self::Sorted => "sorted",
            Self::Pivot => "pivot",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, AlgorithmError> {
        match name {
            "merged" => Ok(Self::Merged),
            "heap" => Ok(Self::Heap),
            "sorted" => Ok(Self::Sorted),
            "pivot" => Ok(Self::Pivot),
            other => Err(AlgorithmError(format!(
                "Unknown columns {other:?}, expected 'merged', 'heap', 'sorted' or 'pivot'"
            ))),
        }
    }
}

/// `'locking'` or `'lock-free'`.
pub fn storage_name(storage: Storage) -> &'static str {
    match storage {
        Storage::LockFree => "lock-free",
        _ => "locking",
    }
}

pub fn storage_from_name(name: &str) -> Result<Storage, AlgorithmError> {
    match name {
        "locking" => Ok(Storage::Locking),
        "lock-free" => Ok(Storage::LockFree),
        other => Err(AlgorithmError(format!(
            "Unknown storage {other:?}, expected 'locking' or 'lock-free'"
        ))),
    }
}

/// An invalid algorithm, or one that can't run in a pipeline. A `ValueError` in Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgorithmError(pub String);

impl std::fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AlgorithmError {}

/// What a pipeline uses for the options an [`AlgorithmSpec`] leaves unset.
///
/// The best choice depends on the matrix (see the benchmarks): columns that are expensive to compute favour caching,
/// cheap ones favour recomputing them and saving the memory, and storing R pays off when columns need many additions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlgorithmDefaults {
    /// The pipeline, for error messages, as users call it (e.g. `"phlite_clouds.rips"`).
    pub pipeline: &'static str,
    /// Serial clearing's columns. Only serial clearing uses [`Columns::Pivot`], so the other algorithms merge instead.
    pub columns: Columns,
    pub cache: bool,
    /// Serial clearing's choice (the other algorithms don't support storing R).
    pub store_r: bool,
    /// Whether the pipeline's matrices know the rank of every row
    /// ([`row_ranks`](phlite::matrices::HasRowFiltration::row_ranks)), which [`Columns::Pivot`] needs: without
    /// them, asking for pivot columns is refused rather than silently merging.
    pub row_ranks: bool,
    /// Whether the pipeline finds dimension 0 (and, for images, the top dimension) by union-find rather than by
    /// reducing it, when computing by cohomology; `None` if it can't (asking for it is then refused).
    pub union_find: Option<bool>,
}

/// A reduction algorithm, with the options the user set: `None` means the pipeline's default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgorithmSpec {
    pub kind: Kind,
    pub columns: Option<Columns>,
    pub cache: Option<bool>,
    pub store_r: Option<bool>,
    /// The default (skip, except for the standard algorithm) depends only on the algorithm, so is always set.
    pub skip_zero_apparent_pairs: bool,
    /// `None` detects the order from the matrix.
    pub dimension_order: Option<Vec<usize>>,
    pub storage: Storage,
    /// `None` for all cores.
    pub threads: Option<usize>,
    /// `None` means the pipeline's default (see [`AlgorithmDefaults::union_find`]).
    pub union_find: Option<bool>,
}

impl AlgorithmSpec {
    /// `kind` with every option at its default.
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            columns: None,
            cache: None,
            store_r: None,
            // The barcode omits zero-length bars anyway, so there's no point recording them
            skip_zero_apparent_pairs: kind != Kind::Standard,
            dimension_order: None,
            storage: Storage::Locking,
            threads: None,
            union_find: None,
        }
    }

    /// An error unless the algorithm is one of `kinds`.
    pub fn require(&self, kinds: &[Kind], option: &str) -> Result<(), AlgorithmError> {
        if kinds.contains(&self.kind) {
            Ok(())
        } else {
            Err(AlgorithmError(format!(
                "{option} can't be set for the {} algorithm",
                self.kind.name()
            )))
        }
    }

    /// Check the options that don't depend on the pipeline.
    pub fn validate(&self) -> Result<(), AlgorithmError> {
        if self.columns == Some(Columns::Pivot) && self.kind != Kind::Clearing {
            return Err(AlgorithmError(format!(
                "'pivot' columns are only used by serial clearing, not the {} algorithm (it would merge them)",
                self.kind.name()
            )));
        }
        if self.store_r == Some(true) {
            self.require(&[Kind::Clearing], "store_r")?;
        }
        if self.skip_zero_apparent_pairs {
            self.require(
                &[Kind::Clearing, Kind::ParallelClearing],
                "skip_zero_apparent_pairs",
            )?;
        }
        if self.dimension_order.is_some() {
            self.require(&[Kind::Clearing, Kind::ParallelClearing], "dimension_order")?;
        }
        if self.storage != Storage::Locking {
            self.require(&[Kind::ParallelClearing], "storage")?;
        }
        if let Some(threads) = self.threads {
            self.require(&[Kind::ParallelClearing], "threads")?;
            if threads == 0 {
                return Err(AlgorithmError("threads must be positive".into()));
            }
        }
        Ok(())
    }

    /// Fill in the unset options from `defaults`, and check the algorithm can run in that pipeline.
    pub fn resolve(
        &self,
        defaults: &AlgorithmDefaults,
    ) -> Result<ResolvedAlgorithm, AlgorithmError> {
        self.validate()?;
        let columns = match self.columns {
            Some(Columns::Pivot) if !defaults.row_ranks => {
                return Err(AlgorithmError(format!(
                    "'pivot' columns need the rank of every row in filtration order, which the matrices of the \
                     pipeline '{}' don't give (they would merge instead)",
                    defaults.pipeline
                )))
            }
            Some(columns) => columns,
            // Only serial clearing uses pivot columns (the others make a fresh column per column, which merges)
            None if defaults.columns == Columns::Pivot && self.kind != Kind::Clearing => Columns::Merged,
            None => defaults.columns,
        };
        let union_find = match (self.union_find, defaults.union_find) {
            (Some(true), None) => {
                return Err(AlgorithmError(format!(
                    "The pipeline '{}' can't find dimension 0 by union-find",
                    defaults.pipeline
                )))
            }
            (_, None) => None,
            (set, Some(default)) => Some(set.unwrap_or(default)),
        };
        Ok(ResolvedAlgorithm {
            kind: self.kind,
            union_find,
            columns,
            cache: self.cache.unwrap_or(defaults.cache),
            store_r: self
                .store_r
                .unwrap_or(defaults.store_r && self.kind == Kind::Clearing),
            skip_zero_apparent_pairs: self.skip_zero_apparent_pairs,
            dimension_order: self.dimension_order.clone(),
            storage: self.storage,
            threads: self.threads,
        })
    }
}

/// `Algorithm.serial()`: clearing, with every option at its default.
impl Default for AlgorithmSpec {
    fn default() -> Self {
        Self::new(Kind::Clearing)
    }
}

/// What to do with an [`Algorithm`]: implement this to call your (generic) barcode function,
/// then pass it to [`ResolvedAlgorithm::run`].
///
/// This is a trait rather than a closure since the column representation is a type parameter.
pub trait AlgorithmVisitor: Send {
    type Output: Send;
    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output;
}

/// An [`AlgorithmSpec`] with every option decided, ready to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAlgorithm {
    pub kind: Kind,
    pub columns: Columns,
    pub cache: bool,
    pub store_r: bool,
    pub skip_zero_apparent_pairs: bool,
    pub dimension_order: Option<Vec<usize>>,
    pub storage: Storage,
    pub threads: Option<usize>,
    /// `None` if the pipeline can't use union-find.
    pub union_find: Option<bool>,
}

impl ResolvedAlgorithm {
    /// The spec with every option set as resolved (resolving it again, for any pipeline that can take it, gives this).
    pub fn to_spec(&self) -> AlgorithmSpec {
        AlgorithmSpec {
            kind: self.kind,
            columns: Some(self.columns),
            cache: Some(self.cache),
            store_r: Some(self.store_r),
            skip_zero_apparent_pairs: self.skip_zero_apparent_pairs,
            dimension_order: self.dimension_order.clone(),
            storage: self.storage,
            threads: self.threads,
            union_find: self.union_find,
        }
    }

    /// Whether to use union-find (for a pipeline that can).
    pub fn union_find(&self) -> bool {
        self.union_find.unwrap_or(false)
    }

    /// Run `visitor` with the configured [`Algorithm`], on a pool of `threads` threads if set.
    /// With threads set, the visitor itself runs on the pool, so any other rayon work it does (such as building a basis)
    /// uses it too. Fails only if the thread pool can't be built.
    pub fn run<V: AlgorithmVisitor>(&self, visitor: V) -> Result<V::Output, AlgorithmError> {
        match self.columns {
            Columns::Merged => self.run_with(MergedColumns, visitor),
            Columns::Heap => self.run_with(HeapColumns, visitor),
            Columns::Sorted => self.run_with(SortedColumns, visitor),
            Columns::Pivot => self.run_with(PivotColumns, visitor),
        }
    }

    fn build_clearing<R: ColumnRepresentation>(&self, columns: R) -> Clearing<R> {
        let mut clearing = Clearing::new()
            .with_columns(columns)
            .cache_columns(self.cache)
            .store_r(self.store_r);
        if let Some(order) = &self.dimension_order {
            clearing = clearing.with_dimension_order(order.iter().copied());
        }
        clearing.skip_zero_apparent_pairs(self.skip_zero_apparent_pairs)
    }

    fn run_with<R: ColumnRepresentation, V: AlgorithmVisitor>(
        &self,
        columns: R,
        visitor: V,
    ) -> Result<V::Output, AlgorithmError> {
        match self.kind {
            Kind::Standard => Ok(visitor.visit(Algorithm::Standard(
                Standard::new()
                    .with_columns(columns)
                    .cache_columns(self.cache),
            ))),
            Kind::Clearing => Ok(visitor.visit(Algorithm::Clearing(self.build_clearing(columns)))),
            Kind::ParallelClearing => {
                let algorithm = self
                    .build_clearing(columns)
                    .parallel()
                    .storage(self.storage);
                match self.threads {
                    None => Ok(visitor.visit(Algorithm::ParallelClearing(algorithm))),
                    Some(n) => {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(n)
                            .build()
                            .map_err(|e| {
                                AlgorithmError(format!("Could not build thread pool: {e}"))
                            })?;
                        let algorithm = algorithm.in_pool(&pool);
                        Ok(pool.install(|| visitor.visit(Algorithm::ParallelClearing(algorithm))))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RIPS: AlgorithmDefaults = AlgorithmDefaults {
        pipeline: "rips",
        columns: Columns::Heap,
        cache: false,
        store_r: false,
        row_ranks: false,
        union_find: None,
    };
    const COMPLEX: AlgorithmDefaults = AlgorithmDefaults {
        pipeline: "complex",
        columns: Columns::Pivot,
        cache: false,
        store_r: true,
        row_ranks: true,
        union_find: None,
    };

    #[test]
    fn unset_options_take_the_pipelines_defaults() {
        let serial = AlgorithmSpec::default();
        let rips = serial.resolve(&RIPS).unwrap();
        assert_eq!(
            (rips.columns, rips.cache, rips.store_r),
            (Columns::Heap, false, false)
        );
        let complex = serial.resolve(&COMPLEX).unwrap();
        assert_eq!((complex.columns, complex.store_r), (Columns::Pivot, true));
        // Only serial clearing uses pivot columns or stores R
        let parallel = AlgorithmSpec::new(Kind::ParallelClearing)
            .resolve(&COMPLEX)
            .unwrap();
        assert_eq!(
            (parallel.columns, parallel.store_r),
            (Columns::Merged, false)
        );
    }

    #[test]
    fn set_options_win() {
        let spec = AlgorithmSpec {
            columns: Some(Columns::Sorted),
            cache: Some(true),
            store_r: Some(false),
            ..AlgorithmSpec::default()
        };
        let resolved = spec.resolve(&COMPLEX).unwrap();
        assert_eq!(
            (resolved.columns, resolved.cache, resolved.store_r),
            (Columns::Sorted, true, false)
        );
    }

    #[test]
    fn union_find_only_where_supported() {
        let with = |union_find| AlgorithmSpec { union_find, ..AlgorithmSpec::default() };
        let supported = AlgorithmDefaults { union_find: Some(true), ..RIPS };
        assert_eq!(with(None).resolve(&supported).unwrap().union_find, Some(true));
        assert_eq!(with(Some(false)).resolve(&supported).unwrap().union_find, Some(false));
        assert_eq!(with(None).resolve(&RIPS).unwrap().union_find, None);
        assert_eq!(with(Some(false)).resolve(&RIPS).unwrap().union_find, None);
        assert!(with(Some(true)).resolve(&RIPS).unwrap_err().0.contains("union-find"));
    }

    #[test]
    fn pivot_columns_need_row_ranks() {
        let spec = AlgorithmSpec {
            columns: Some(Columns::Pivot),
            ..AlgorithmSpec::default()
        };
        assert!(spec.resolve(&COMPLEX).is_ok());
        assert!(spec.resolve(&RIPS).unwrap_err().0.contains("rips"));
    }

    #[test]
    fn options_must_suit_the_algorithm() {
        let standard = AlgorithmSpec::new(Kind::Standard);
        assert!(standard.validate().is_ok());
        for bad in [
            AlgorithmSpec {
                store_r: Some(true),
                ..standard.clone()
            },
            AlgorithmSpec {
                skip_zero_apparent_pairs: true,
                ..standard.clone()
            },
            AlgorithmSpec {
                threads: Some(2),
                ..AlgorithmSpec::default()
            },
            AlgorithmSpec {
                threads: Some(0),
                ..AlgorithmSpec::new(Kind::ParallelClearing)
            },
            AlgorithmSpec {
                columns: Some(Columns::Pivot),
                ..AlgorithmSpec::new(Kind::ParallelClearing)
            },
        ] {
            assert!(bad.resolve(&COMPLEX).is_err(), "{bad:?}");
        }
    }
}
