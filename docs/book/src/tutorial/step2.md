# 2. A filtration and a basis

For *persistent* homology the rows need filtration values. Implement `HasRowFiltration` on the oracle:

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:filtration}}
```

The filtration type must be `Ord`; `NotNan` from `ordered-float` is the usual choice. (For a quick experiment you can
attach a closure instead, with `Boundary(&volume).with_filtration(|cell| ...)`, but implementing the trait on your own
type leaves room for the optional hooks in [step 5](step5.md).)

Rows are now ordered by `(value, label)`, so the basis must be too:

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:basis}}
```

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:step2}}
```

The result is a list of `Bar`s: birth cell and value, death cell and value (`None` if essential), and optionally a
representative.

**You can now** compute persistent homology. But the standard algorithm reduces every column, cubes included, and
builds every column in full. On a 40³ volume (493,039 cells) that takes 5.4 s.
