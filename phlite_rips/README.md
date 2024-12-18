# phlite_rips

This crate is a implementation of the Vietoris-Rips filtration in `phlite`.
This is mainly meant as a proof-of-concept for the main library and as a familiar example for those new to `phlite`.

The main output of the crate is a CLI which accepts a distance matrix on STDIN in CSV format and then computes VR persistent homology.
The typical usage looks like
```
cat my_distance_matrix.csv | cargo run --release 3
```
if you want to compute PH for the distance matrix in `my_distance_matrix.csv` up to maximum dimension 3.
If you do not specify max dimension, it is assumed to be 1.

If you want to quickly test this out for say 100 points on a circle (using intrinsic distance) you can try
```
python circle_distance_matrix.py | cargo run --release
```

Note this implementation does not have many of the optimisations of [Ripser](https://github.com/Ripser/ripser), including
* apparent and emergent pairs
* filtration truncation

## `phlite` tutorial

This crate also acts as a tutorial for the `phlite` crate.
There are two entry points for this tutorial:
* `src/lib.rs` - To see how to start implementing the core `phlite` traits.
* `src/main.rs` - To see how to use such an implementation to actually compute persistent homology.
