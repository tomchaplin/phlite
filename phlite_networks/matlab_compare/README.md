# Checking `phlite_networks::pph` against pph-matlab

This directory compares `phlite_networks::pph` (through the `phlite_networks_pph` CLI) against [`pph-matlab`](https://github.com/samirchowdhury/pph-matlab) under Octave:

```sh
cargo build --release -p phlite_networks
cd phlite_networks/matlab_compare
PPH_MATLAB_DIR=/path/to/pph-matlab python3 compare.py 0 150 higher_dim_networks.txt
python3 sparse_check.py 0 40   # networks with missing edges, against brute force (pph-matlab caps weights instead)
```

`pph_dim.m` is `pph-matlab`'s `computePPH` with the dimension as an argument (it is hard-coded to 1).
`higher_dim_networks.txt` holds 80 networks with bars in dimension 2, found by `search_h2.py`.

## References

<a id="1">[1]</a> Chowdhury, S. and Mémoli, F., 2018. Persistent path homology of directed networks. In *Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1152–1169). [arXiv:1701.00565](https://arxiv.org/abs/1701.00565).
