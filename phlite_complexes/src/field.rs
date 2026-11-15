//! Converting between Python integers and `phlite`'s coefficient fields.

use phlite::{
    barcode::Field,
    fields::{NonZeroCoefficient, Z2, ZP},
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

// `ZP`'s representation is private, so we build elements by adding up ones.
// The moduli we support are tiny, so this is cheap, and it only happens on input and output.
impl<const P: u8> IntField for ZP<P>
where
    ZP<P>: Field,
{
    const MODULUS: u64 = P as u64;

    fn from_int(value: i64) -> Option<Self> {
        let residue = value.rem_euclid(P as i64);
        if residue == 0 {
            return None;
        }
        let one = Self::one();
        let mut x = one;
        for _ in 1..residue {
            x = (x + one).expect("a sum of fewer than P ones is non-zero");
        }
        Some(x)
    }

    fn to_int(self) -> u64 {
        (1..P as i64)
            .find(|&k| Self::from_int(k) == Some(self))
            .expect("every non-zero element is a sum of ones") as u64
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
