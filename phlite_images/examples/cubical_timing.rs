//! Time the cubical barcode of smoothed noise, the stages separately: `cargo run --release --example cubical_timing`.

use std::time::Instant;

use phlite::{barcode::Algorithm, columns::MergedColumns, reduction::Clearing};
use phlite_images::cubical::{cubical_barcode, Construction, CubicalComplex, Filtration, Method};

/// Uniform noise averaged over a 5-pixel window along each axis (wrapping around), as the Python benchmarks' `smooth`.
fn smooth_noise<const D: usize>(shape: [usize; D], seed: u64) -> Vec<f64> {
    let mut s = seed;
    let n: usize = shape.iter().product();
    let mut image: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();
    let mut stride = 1;
    for axis in (0..D).rev() {
        let len = shape[axis];
        let mut out = vec![0.0; n];
        for (i, value) in out.iter_mut().enumerate() {
            let x = (i / stride) % len;
            let base = i - x * stride;
            *value = (-2i64..=2)
                .map(|d| image[base + ((x as i64 + d).rem_euclid(len as i64) as usize) * stride])
                .sum::<f64>()
                / 5.0;
        }
        image = out;
        stride *= len;
    }
    image.into_iter().map(|v| v as f32 as f64).collect()
}

fn time<const D: usize>(shape: [usize; D]) {
    let pixels = smooth_noise(shape, 1);
    let pixels = if std::env::var("NOISE").is_ok() { (0..pixels.len()).map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 11) as f64 / (1u64 << 53) as f64).collect() } else { pixels };
    for construction in [Construction::T, Construction::V] {
        let start = Instant::now();
        let complex = CubicalComplex::new(pixels.clone(), shape, construction, Filtration::Sublevel).unwrap();
        let built = start.elapsed();
        let start = Instant::now();
        let bars = cubical_barcode(
            &complex,
            D - 1,
            Method::Cohomology,
            Algorithm::Clearing(Clearing::new().with_columns(MergedColumns).store_r(true)),
            false,
        );
        let serial = start.elapsed();
        if std::env::args().len() > 1 {
            println!("{shape:?} {construction:?}: barcode {serial:?} ({} bars)", bars.len());
            continue;
        }
        let start = Instant::now();
        let reps = cubical_barcode(
            &complex,
            D - 1,
            Method::Cohomology,
            Algorithm::Clearing(Clearing::new().with_columns(MergedColumns).store_r(true)),
            true,
        );
        let with_reps = start.elapsed();
        println!(
            "{shape:?} {construction:?}: build {built:?}, barcode {serial:?} ({} bars), with representatives {with_reps:?} ({} bars)",
            bars.len(),
            reps.len()
        );
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("3d") {
        for _ in 0..5 {
            time([64, 64, 64]);
        }
        return;
    }
    time([1024, 1024]);
    if std::env::args().len() == 1 {
        time([64, 64, 64]);
    }
}
