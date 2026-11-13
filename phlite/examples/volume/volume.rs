//! A 3D greyscale volume and the geometry of its cubical complex, shared by the tutorial's examples
//! (`volume_basic.rs` and `volume_lazy.rs`). Cells are numbered by `u64`s; each example wraps them in its own label.

#![allow(dead_code)]

// ANCHOR: volume
/// A 3D greyscale volume: `size³` voxels, indexed `(x, y, z)`.
pub struct Volume {
    pub size: usize,
    pub voxels: Vec<f32>,
}
// ANCHOR_END: volume

// ANCHOR: geometry
/// Voxels are vertices. The cells sit at the points of a grid of side `2 size - 1`, at coordinates `(x, y, z)`:
/// a coordinate that is odd spans two voxels, so a cell's dimension is its number of odd coordinates (a vertex, an
/// edge, a square or a cube). Cells are numbered `(x · side + y) · side + z`.
impl Volume {
    pub fn side(&self) -> u64 {
        2 * self.size as u64 - 1
    }

    pub fn n_cells(&self) -> u64 {
        self.side().pow(3)
    }

    fn coordinates(&self, cell: u64) -> [u64; 3] {
        let side = self.side();
        [cell / (side * side), (cell / side) % side, cell % side]
    }

    fn number(&self, [x, y, z]: [u64; 3]) -> u64 {
        let side = self.side();
        (x * side + y) * side + z
    }

    pub fn dimension(&self, cell: u64) -> usize {
        self.coordinates(cell).iter().filter(|&&c| c % 2 == 1).count()
    }

    /// The faces of a cell: one step either way along each odd coordinate (at most six).
    pub fn faces(&self, cell: u64) -> impl Iterator<Item = u64> {
        let coords = self.coordinates(cell);
        let mut faces = [None; 6];
        for axis in 0..3 {
            if coords[axis] % 2 == 1 {
                for (k, step) in [-1i64, 1].into_iter().enumerate() {
                    let mut face = coords;
                    face[axis] = (face[axis] as i64 + step) as u64;
                    faces[2 * axis + k] = Some(self.number(face));
                }
            }
        }
        faces.into_iter().flatten()
    }

    /// The cofaces of a cell: one step either way along each even coordinate, inside the grid (at most six), in
    /// increasing order of their numbers: the steps down (along x, then y, then z), then the steps up (z, y, x).
    pub fn cofaces(&self, cell: u64) -> impl Iterator<Item = u64> {
        let coords = self.coordinates(cell);
        let side = self.side();
        let mut cofaces = [None; 6];
        for axis in 0..3 {
            if coords[axis] % 2 == 0 {
                if coords[axis] > 0 {
                    let mut down = coords;
                    down[axis] -= 1;
                    cofaces[axis] = Some(self.number(down));
                }
                if coords[axis] + 1 < side {
                    let mut up = coords;
                    up[axis] += 1;
                    cofaces[5 - axis] = Some(self.number(up));
                }
            }
        }
        cofaces.into_iter().flatten()
    }

    /// The lower-star filtration: a cell appears with its brightest voxel (of the up to eight it spans).
    pub fn value(&self, cell: u64) -> f32 {
        let [x, y, z] = self.coordinates(cell);
        let n = self.size as u64;
        let mut value = f32::NEG_INFINITY;
        for i in [x / 2, x.div_ceil(2)] {
            for j in [y / 2, y.div_ceil(2)] {
                for k in [z / 2, z.div_ceil(2)] {
                    value = value.max(self.voxels[((i * n + j) * n + k) as usize]);
                }
            }
        }
        value
    }
}
// ANCHOR_END: geometry

/// A smooth random volume: noise blurred along each axis in turn, so that it has blobs, tunnels and cavities at
/// several scales. Values are rounded to three decimal places, so there are ties.
pub fn smooth_noise(size: usize, seed: u64) -> Volume {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u32 << 24) as f32
    };
    let mut voxels: Vec<f32> = (0..size * size * size).map(|_| next()).collect();
    let radius = 2usize;
    let strides = [size * size, size, 1];
    for stride in strides {
        let source = voxels.clone();
        for (index, voxel) in voxels.iter_mut().enumerate() {
            let position = (index / stride) % size;
            let low = position.saturating_sub(radius);
            let high = (position + radius).min(size - 1);
            let mut sum = 0.0;
            for p in low..=high {
                sum += source[index - position * stride + p * stride];
            }
            *voxel = sum / (high - low + 1) as f32;
        }
    }
    for voxel in &mut voxels {
        *voxel = (*voxel * 1000.0).round() / 1000.0;
    }
    Volume { size, voxels }
}

/// Start measuring peak memory afresh (Linux only; elsewhere this does nothing).
pub fn reset_peak_memory() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// The peak memory of this process since the last [`reset_peak_memory`], in MB (Linux only; `None` elsewhere).
pub fn peak_memory_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}
