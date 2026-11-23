//! Converting between Python integers and `phlite`'s coefficient fields.

use phlite::{
    barcode::Field,
    fields::{Z2, ZP},
};
use std::fmt::Debug;

/// A coefficient field that can be read from (and written back to) integers.
pub trait IntField: Field + Debug {
    /// The characteristic of the field.
    const MODULUS: u64;

    /// The residue of `value` modulo [`MODULUS`](Self::MODULUS), or `None` if it is zero.
    fn from_int(value: i64) -> Option<Self>;

    /// The representative of `self` in `1..MODULUS`.
    fn to_int(self) -> u64;
}

impl IntField for Z2 {
    const MODULUS: u64 = 2;

    fn from_int(value: i64) -> Option<Self> {
        (value.rem_euclid(2) == 1).then_some(Z2)
    }

    fn to_int(self) -> u64 {
        1
    }
}

impl<const P: u8> IntField for ZP<P>
where
    ZP<P>: Field,
{
    const MODULUS: u64 = P as u64;

    fn from_int(value: i64) -> Option<Self> {
        Self::new(value.rem_euclid(P as i64) as u8)
    }

    fn to_int(self) -> u64 {
        self.value() as u64
    }
}

/// What a chain complex's coefficients can be: any [`IntField`] (whose complexes `phlite` reduces), or the integers
/// ([`Integer`], whose complexes have spectral sequences and persistent groups computed exactly, by
/// [`phlite::integral`]).
pub trait ChainCoefficient: Copy + Debug + Send + Sync + 'static {
    /// The coefficient of an entry whose input coefficients sum to `total`, `None` if it is zero, or `Err` if it
    /// doesn't fit.
    fn from_sum(total: i128) -> Result<Option<Self>, i128>;

    /// The coefficient as an integer (over `Z/p`, its representative in `1..p`).
    fn to_i64(self) -> i64;
}

impl<CF: IntField> ChainCoefficient for CF {
    fn from_sum(total: i128) -> Result<Option<Self>, i128> {
        Ok(CF::from_int(total.rem_euclid(CF::MODULUS as i128) as i64))
    }

    fn to_i64(self) -> i64 {
        self.to_int() as i64
    }
}

/// Integer coefficients (non-zero), for chain complexes over Z: `phlite`'s.
pub use phlite::fields::Integer;

impl ChainCoefficient for Integer {
    fn from_sum(total: i128) -> Result<Option<Self>, i128> {
        match i64::try_from(total) {
            Ok(x) => Ok(Integer::new(x)),
            Err(_) => Err(total),
        }
    }

    fn to_i64(self) -> i64 {
        self.get()
    }
}

#[cfg(test)]
mod tests {
    use super::IntField;
    use phlite::fields::{Z13, Z2, Z3, Z5};

    fn round_trips<F: IntField>() {
        for k in -30i64..30 {
            let residue = k.rem_euclid(F::MODULUS as i64) as u64;
            match F::from_int(k) {
                None => assert_eq!(residue, 0u64),
                Some(x) => assert_eq!(x.to_int(), residue),
            }
        }
    }

    #[test]
    fn all_round_trip() {
        round_trips::<Z2>();
        round_trips::<Z3>();
        round_trips::<Z5>();
        round_trips::<Z13>();
    }

    #[test]
    fn arithmetic_agrees() {
        let two = Z5::from_int(2).unwrap();
        let three = Z5::from_int(3).unwrap();
        assert_eq!((two * three).to_int(), 1);
        assert_eq!(two + three, None);
    }
}
