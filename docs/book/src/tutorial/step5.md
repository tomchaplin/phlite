# 5. The optional hooks: emergent pairs

`HasRowFiltration` has optional methods (*hooks*) with correct but generic defaults. Overriding them tells the
reduction things about your matrix that it would otherwise work out the slow way. `volume_lazy.rs` starts from the
filtration order on cells, and the two questions apparent pairs ask:

```rust,ignore
{{#include ../../../../phlite/examples/volume_lazy.rs:order}}
```

and overrides three hooks on the coboundary:

```rust,ignore
{{#include ../../../../phlite/examples/volume_lazy.rs:hooks}}
```

- **`column_min`: emergent pairs.** Cohomology reduces the reversed coboundary, whose pivot is a column's *smallest*
  coface. In a lower-star filtration no coface can come before the cell itself, and our cofaces come in increasing
  order of their numbers, so the first coface with the cell's own value *is* the smallest, and the search stops there.
  Most columns need nothing but their pivot (nobody else owns it), so most columns are finished after looking at one
  or two cofaces. Ripser does the same for Vietoris–Rips; Bauer calls these
  [emergent pairs](../algorithms/apparent-pairs.md#emergent-pairs).
- **`first_column_in_row` / `last_column_in_row`: apparent pairs without the transpose.** The columns with an entry
  in row \\(r\\) of the coboundary are exactly the faces of \\(r\\), so the first is its oldest face. No transpose,
  and no adding up of coefficients.
- **`column_pivot`** isn't overridden: homology would use it, cohomology doesn't.

Every hook must be *right*. A pivot that isn't the true smallest entry, or a "first column" that isn't, gives a wrong
barcode silently. The example checks its barcode against the other methods; the [property tests](../appendix/testing.md)
show how to test hooks against their defaults on random inputs.

```rust,ignore
{{#include ../../../../phlite/examples/volume_lazy.rs:step5}}
```

On their own the hooks don't buy much *here*: on the 40³ volume this runs in 167 ms, about the same as plain clearing
in step 3. With at most six cofaces per column, stopping the search early saves little. (For Vietoris–Rips, where a
simplex can have thousands of cofacets, emergent pairs are what make cohomology fast.) What the hooks do buy is cheap,
exact apparent pairs, which the next step builds on.

Heap columns, uncached, as in `phlite_rips`: our columns are short and cheap to recompute, so there is nothing to
gain from sorting or keeping them (see [choosing a representation](../columns/choosing.md)).
