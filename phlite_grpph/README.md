# phlite_grpph

This library uses [`phlite`](https://github.com/tomchaplin/phlite) to compute grounded persistent path homology (GrPPH) [[1]](#1).
We expose Python bindings to two python functions: `grpph` and `grpph_with_involution`.

Both functions accept two arguments `nodes` and `edges`.
The first, `nodes`, should be an integer: the number of nodes in your weighted digraph.
The second, `edges`, should be a list of tuples `(i, j, w)` representing an edge from node `i` to node `j` with weight `w`.

The first function returns two values: `essential` and `pairings`.
The second function additionally returns `reps`.
The first two return values are the GrPPH barcode.
The `reps` return value is a list of cycle representatives: one for each finite feature in degree `1`.

For example usage, see `reps.py` and `example.py` or get in touch.

## References

<a id="1">[1]</a>
Chaplin, T., Harrington, H.A. and Tillmann, U., 2022.
Grounded persistent path homology: a stable, topological descriptor for weighted digraphs.
arXiv preprint [arXiv:2210.11274](https://arxiv.org/abs/2210.11274).
