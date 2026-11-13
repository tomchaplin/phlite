# 7. Representatives and threads

Two last things come almost for free.

```rust,ignore
{{#include ../../../../phlite/examples/volume_lazy.rs:step7}}
```

## Representatives

`barcode::cohomology_with_involution` computes the pairs by cohomology and then cycle representatives by a small
homology reduction of just the death and essential birth columns ([the involution](../algorithms/involution.md)). It
needs the boundary, attached as the transpose. Each `Bar`'s `representative` is a list of `(coefficient, cell)`: for a
bar in \\(H_2\\), a closed surface of squares enclosing a dark cavity in the volume.

```text
... with representatives                              4121.2 ms
    the longest finite H2 bar, [0.536, 0.632), is a closed surface of 1268 squares
```

## Threads

`Clearing::new().parallel()` (feature `parallel`) reduces each dimension on all cores
([parallel clearing](../algorithms/parallel.md)):

```text
cohomology, parallel clearing                         1575.0 ms
```

(against 2020 ms on one thread, on a laptop with four cores). The reduction itself is a small part of the work here,
since most columns finish at their pivot.

## What not to turn on

`phlite` also has PHAT's tricks: [dense indices](../internals/bookkeeping.md) for vector bookkeeping,
[pivot columns](../columns/pivot.md) and [stored R](../algorithms/store-r.md). They need an array over every cell,
which is exactly what this oracle avoids, and they pay when many columns need genuine reduction, which isn't the
case here. If your complex is like that, build it explicitly with the [`phlite` Python package](../python/phlite.md),
which applies them all.

## Next

The same ideas, applied to Vietoris–Rips with much more care over enumerating cofacets, are `phlite_rips`; its source
(`phlite_rips/src/lib.rs` onwards, comments marked `TUTORIAL`) is a good next read.
