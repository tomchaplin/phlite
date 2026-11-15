//! Compute PPH barcodes of networks read from stdin, in the same text format as the `pph-matlab` test harness.
//!
//! Input: blocks of `n dim t_end` followed by the `n x n` weight matrix (row `a`, column `b` is the weight of `a -> b`,
//! the diagonal is ignored). As in `pph-matlab`, weights are capped at `t_end`; pass `--no-cap` to instead treat weights
//! above `t_end` (e.g. `Inf`) as missing edges.
//!
//! Output: for each network a line `network k` and then one line `dim birth death` per bar.
//!
//! Options: `--method homology|cohomology`, `--algorithm standard|clearing|parallel`, `--field 2|3|5|0` (0 for the rationals), `--no-cap`,
//! `--representatives` (computed but not printed), `--threshold t` (truncate the filtration at `t`).

use std::io::{self, Read, Write};

use phlite::{
    barcode::{Algorithm, Method},
    columns::MergedColumns,
    fields::{Invertible, NonZeroCoefficient, Q, Z2, Z3, Z5},
    reduction::{Clearing, Standard},
};
use phlite_networks::pph::{pph_barcode, Network, PPHBar};

struct Options {
    method: Method,
    algorithm: String,
    field: u8,
    cap: bool,
    representatives: bool,
    threshold: Option<f64>,
}

fn algorithm(name: &str) -> Algorithm<'static, MergedColumns> {
    match name {
        "standard" => Algorithm::Standard(Standard::new()),
        "clearing" => Algorithm::Clearing(Clearing::new()),
        "parallel" => Algorithm::ParallelClearing(Clearing::new().parallel()),
        other => panic!("unknown algorithm {other}"),
    }
}

fn run<CF>(network: &Network, dim: usize, opts: &Options) -> Vec<(usize, f64, f64)>
where
    CF: NonZeroCoefficient + Invertible + Send + Sync + 'static,
{
    let bars: Vec<PPHBar<CF>> = pph_barcode(
        network,
        dim,
        opts.method,
        algorithm(&opts.algorithm),
        opts.representatives,
    );
    bars.into_iter()
        .map(|bar| {
            (
                bar.birth.dimension(),
                bar.birth_value.into_inner(),
                bar.death.map_or(f64::INFINITY, |(_, d)| d.into_inner()),
            )
        })
        .collect()
}

fn main() {
    let mut opts = Options {
        method: Method::Cohomology,
        algorithm: "clearing".into(),
        field: 2,
        cap: true,
        representatives: false,
        threshold: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--method" => {
                i += 1;
                opts.method = match args[i].as_str() {
                    "homology" => Method::Homology,
                    "cohomology" => Method::Cohomology,
                    other => panic!("unknown method {other}"),
                }
            }
            "--algorithm" => {
                i += 1;
                opts.algorithm = args[i].clone();
            }
            "--field" => {
                i += 1;
                opts.field = args[i].parse().unwrap();
            }
            "--no-cap" => opts.cap = false,
            "--representatives" => opts.representatives = true,
            "--threshold" => {
                i += 1;
                opts.threshold = Some(args[i].parse().unwrap());
            }
            other => panic!("unknown option {other}"),
        }
        i += 1;
    }

    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut tokens = input.split_whitespace().map(|t| {
        if t.eq_ignore_ascii_case("inf") {
            f64::INFINITY
        } else {
            t.parse::<f64>().unwrap()
        }
    });
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut k = 0;
    while let Some(n) = tokens.next() {
        let n = n as usize;
        let dim = tokens.next().unwrap() as usize;
        let t_end = tokens.next().unwrap();
        let weights: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        let w = tokens.next().unwrap();
                        if opts.cap {
                            w.min(t_end)
                        } else if w > t_end {
                            f64::INFINITY
                        } else {
                            w
                        }
                    })
                    .collect()
            })
            .collect();
        let mut network = Network::from_dense(&weights);
        if let Some(t_max) = opts.threshold {
            network = network.truncate(t_max);
        }
        let bars = match opts.field {
            2 => run::<Z2>(&network, dim, &opts),
            3 => run::<Z3>(&network, dim, &opts),
            5 => run::<Z5>(&network, dim, &opts),
            0 => run::<Q>(&network, dim, &opts),
            other => panic!("unsupported field {other}"),
        };
        writeln!(out, "network {k}").unwrap();
        for (d, b, e) in bars {
            writeln!(out, "{d} {b:?} {e:?}").unwrap();
        }
        k += 1;
    }
}
