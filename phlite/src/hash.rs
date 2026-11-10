//! Hashers shared by the concurrent maps in the crate.

use std::hash::{BuildHasherDefault, Hasher};

use rustc_hash::FxHasher;

/// `FxHasher` followed by a finaliser: Fx puts its entropy in the high bits, but the concurrent maps also use the low ones.
#[derive(Default)]
pub struct MixedFx(FxHasher);

impl Hasher for MixedFx {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
    fn write_u8(&mut self, i: u8) {
        self.0.write_u8(i)
    }
    fn write_u16(&mut self, i: u16) {
        self.0.write_u16(i)
    }
    fn write_u32(&mut self, i: u32) {
        self.0.write_u32(i)
    }
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i)
    }
    fn write_usize(&mut self, i: usize) {
        self.0.write_usize(i)
    }
    fn finish(&self) -> u64 {
        let h = self.0.finish();
        let h = (h ^ (h >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^ (h >> 32)
    }
}

pub(crate) type FxBuild = BuildHasherDefault<MixedFx>;
