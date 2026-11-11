//! Maps and sets keyed by basis elements, for the reduction algorithms' per-column state.
//!
//! An [`ElementMap`] is a vector indexed by [`dense_index`](BasisElement::dense_index) when the key type has
//! [`DENSE_INDEX`](BasisElement::DENSE_INDEX) set, and a hash map otherwise. The choice is made from a constant, so the
//! branch not taken compiles away.
//!
//! Vectors are much faster than hash maps (no hashing, and neighbouring cells sit together in memory), but cost 4 bytes
//! for every index up to the largest one used (plus the entries themselves), so need the numbering to be dense.

use std::hash::Hash;
use std::marker::PhantomData;

use rustc_hash::FxHashMap;

use crate::matrices::BasisElement;

/// A map from basis elements to `V`, see the [module docs](self).
#[derive(Debug, Clone)]
pub struct ElementMap<K, V> {
    storage: Storage<V, FxHashMap<K, V>>,
    _key: PhantomData<K>,
}

/// Marks an empty slot of [`Storage::Dense`].
const EMPTY: u32 = u32::MAX;

#[derive(Debug, Clone)]
enum Storage<V, H> {
    /// `slots[i]` is the position in `entries` of the key numbered `i`, or [`EMPTY`].
    /// So a map holding few of many keys costs 4 bytes per key, plus its entries.
    Dense {
        slots: Vec<u32>,
        entries: Vec<(u32, V)>,
    },
    Hashed(H),
}

/// The slot of `key`: its dense index, which must fit in a `u32` (less one, for [`EMPTY`]).
#[inline]
fn slot<K: BasisElement>(key: &K) -> usize {
    let index = key.dense_index();
    debug_assert!(
        index < EMPTY as usize,
        "dense indices must be less than u32::MAX"
    );
    index
}

impl<K: BasisElement + Hash, V> Default for ElementMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: BasisElement + Hash, V> ElementMap<K, V> {
    /// An empty map.
    pub fn new() -> Self {
        let storage = if K::DENSE_INDEX {
            Storage::Dense {
                slots: Vec::new(),
                entries: Vec::new(),
            }
        } else {
            Storage::Hashed(FxHashMap::default())
        };
        Self {
            storage,
            _key: PhantomData,
        }
    }

    /// The value for `key`, if any.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        match &self.storage {
            Storage::Dense { slots, entries } => match *slots.get(slot(key))? {
                EMPTY => None,
                position => Some(&entries[position as usize].1),
            },
            Storage::Hashed(map) => map.get(key),
        }
    }

    /// Whether `key` has a value.
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Set the value for `key`, returning the old one.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match &mut self.storage {
            Storage::Dense { slots, entries } => {
                let i = slot(&key);
                if i >= slots.len() {
                    // `resize` reserves with amortised growth, so this is cheap on average
                    slots.resize(i + 1, EMPTY);
                }
                match slots[i] {
                    EMPTY => {
                        slots[i] = entries.len() as u32;
                        entries.push((i as u32, value));
                        None
                    }
                    position => Some(std::mem::replace(&mut entries[position as usize].1, value)),
                }
            }
            Storage::Hashed(map) => map.insert(key, value),
        }
    }

    /// Remove and return the value for `key`.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match &mut self.storage {
            Storage::Dense { slots, entries } => {
                let i = slot(key);
                let position = match slots.get(i) {
                    None | Some(&EMPTY) => return None,
                    Some(&position) => position as usize,
                };
                slots[i] = EMPTY;
                let (_, value) = entries.swap_remove(position);
                // The last entry moved into the gap
                if let Some((moved, _)) = entries.get(position) {
                    slots[*moved as usize] = position as u32;
                }
                Some(value)
            }
            Storage::Hashed(map) => map.remove(key),
        }
    }

    /// The number of keys with a value.
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Dense { entries, .. } => entries.len(),
            Storage::Hashed(map) => map.len(),
        }
    }

    /// Whether no key has a value.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every key and a reference to its value, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> {
        let (dense, hashed) = match &self.storage {
            Storage::Dense { entries, .. } => (Some(entries), None),
            Storage::Hashed(map) => (None, Some(map)),
        };
        let dense = dense.into_iter().flat_map(|entries| {
            entries
                .iter()
                .map(|(i, value)| (K::from_dense_index(*i as usize), value))
        });
        dense.chain(
            hashed
                .into_iter()
                .flat_map(|map| map.iter().map(|(k, v)| (k.clone(), v))),
        )
    }

    /// Every key and value, in no particular order.
    pub fn into_entries(self) -> impl Iterator<Item = (K, V)> {
        let (dense, hashed) = match self.storage {
            Storage::Dense { entries, .. } => (Some(entries), None),
            Storage::Hashed(map) => (None, Some(map)),
        };
        let dense = dense.into_iter().flat_map(|entries| {
            entries
                .into_iter()
                .map(|(i, value)| (K::from_dense_index(i as usize), value))
        });
        dense.chain(hashed.into_iter().flatten())
    }
}

/// A set of basis elements, see the [module docs](self).
#[derive(Debug, Clone)]
pub struct ElementSet<K>(ElementMap<K, ()>);

impl<K: BasisElement + Hash> Default for ElementSet<K> {
    fn default() -> Self {
        Self(ElementMap::new())
    }
}

impl<K: BasisElement + Hash> ElementSet<K> {
    /// Add `key`, returning whether it was new.
    #[inline]
    pub fn insert(&mut self, key: K) -> bool {
        self.0.insert(key, ()).is_none()
    }

    /// Whether `key` is in the set.
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.0.contains_key(key)
    }

    /// Remove `key`, returning whether it was there.
    #[inline]
    pub fn remove(&mut self, key: &K) -> bool {
        self.0.remove(key).is_some()
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Every element, in no particular order.
    pub fn into_elements(self) -> impl Iterator<Item = K> {
        self.0.into_entries().map(|(key, ())| key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Dense(u32);

    impl BasisElement for Dense {
        const DENSE_INDEX: bool = true;
        fn dense_index(&self) -> usize {
            self.0 as usize
        }
        fn from_dense_index(index: usize) -> Self {
            Dense(index as u32)
        }
    }

    /// Dense and hashed maps behave the same
    fn exercise<K: BasisElement + Hash + std::fmt::Debug>(key: impl Fn(u32) -> K) {
        let mut map = ElementMap::new();
        assert!(map.is_empty());
        assert_eq!(map.insert(key(5), "a"), None);
        assert_eq!(map.insert(key(2), "b"), None);
        assert_eq!(map.insert(key(5), "c"), Some("a"));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key(5)), Some(&"c"));
        assert_eq!(map.get(&key(3)), None);
        assert_eq!(map.get(&key(100)), None);
        assert_eq!(map.remove(&key(2)), Some("b"));
        assert_eq!(map.remove(&key(2)), None);
        assert_eq!(map.remove(&key(100)), None);
        assert_eq!(map.len(), 1);
        map.insert(key(0), "d");
        let mut borrowed: Vec<_> = map.iter().map(|(k, v)| (k, *v)).collect();
        borrowed.sort();
        let mut all: Vec<_> = map.into_entries().collect();
        all.sort();
        let mut expected = vec![(key(0), "d"), (key(5), "c")];
        expected.sort();
        assert_eq!(all, expected);
        assert_eq!(borrowed, all);

        // Removing an early entry moves a later one into its place; lookups must still find it
        let mut map = ElementMap::new();
        for i in [3, 9, 1, 6] {
            map.insert(key(i), i);
        }
        assert_eq!(map.remove(&key(3)), Some(3));
        assert_eq!(map.remove(&key(9)), Some(9));
        for i in [1, 6] {
            assert_eq!(map.get(&key(i)), Some(&i));
        }
        assert_eq!(map.get(&key(3)), None);
        map.insert(key(3), 30);
        assert_eq!(
            (map.len(), map.get(&key(3)), map.get(&key(6))),
            (3, Some(&30), Some(&6))
        );

        let mut set = ElementSet::default();
        assert!(set.insert(key(7)));
        assert!(!set.insert(key(7)));
        assert!(set.contains(&key(7)) && !set.contains(&key(1)));
    }

    #[test]
    fn dense_and_hashed_agree() {
        exercise(Dense);
        exercise(|i| i as usize);
        exercise(|i| std::cmp::Reverse(Dense(i)));
    }
}
