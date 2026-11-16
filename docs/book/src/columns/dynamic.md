# Dynamic heap columns

`DynamicHeapColumns` (`with_columns("dynamic")` in Python) is a [heap column](heap.md) that can switch to keeping
**each row in the heap at most once**, as flagser's working column does: a hash map then holds the sum of the terms
of each row in the heap, and a term whose row is already there is added to that sum rather than pushed. It is
**opt-in**: no pipeline uses it by default.

## When it switches

Hashing every term costs more than pushing it onto a heap, because a heap rarely touches the terms below the pivot
while a hash map touches every one. So deduplicating only pays when a column is **mostly repeats**: the same rows
added over and over, so that the heap would grow far past the number of distinct rows. A column switches when both

- its heap holds at least `threshold` terms (4096 by default), and
- the rows popped from it so far averaged at least `repeats` terms each (2 by default; the column counts this for
  free, as popping a pivot sums its row's terms anyway).

Otherwise it doubles its threshold and stays a plain heap. Switching on size alone, as flagser does, was 1.3–1.7×
slower than a plain heap on Vietoris–Rips, whose long columns have few repeats. With the test for repeats, columns that
don't need deduplicating rarely switch, and cost 1–7% more than a plain heap for the counting.

## When to use it

For **very large directed flag complexes** and other matrices where columns pile up repeated rows. On the directed flag
complex of a random digraph \\(G(500, 0.05)\\) up to dimension 2 it takes 42 s against 55 s for the default (merged,
cached) and 64 s for heap columns.

Elsewhere the defaults win. Compared with each pipeline's own default, in the benchmarks:

| Pipeline | Default | `'dynamic'` relative to it |
|---|---|---|
| Vietoris–Rips (dense, sparse, weighted), flag by shortest paths | heap | 1.00–1.07× |
| directed flag, small and medium | merged, cached | 1.1–1.2× |
| directed flag, \\(G(500, 0.05)\\) | merged, cached | **0.77×** |
| GrPPH | merged, cached | 1.0–2.4× |
| PPH, Dowker, ordered tuples | merged, cached | 1.0–1.3× |
| cubical images | merged, store R | 1.0–1.3× |
| explicit complexes (e.g. `networks.flag` by weight) | pivot, store R | 2.3–3.1× |

```python
from phlite.core import Algorithm
from phlite.networks import directed_flag

barcode = directed_flag(n, edges, 2, algorithm=Algorithm.serial().with_columns("dynamic"))
# or with a different threshold
algorithm = Algorithm.serial().with_columns("dynamic", threshold=256)
```

```rust,ignore
Clearing::new().with_columns(DynamicHeapColumns { threshold: 4096, repeats: 2.0 })
```
