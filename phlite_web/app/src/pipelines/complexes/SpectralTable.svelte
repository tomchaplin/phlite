<script lang="ts">
  // The spectral sequence of the filtration, read off its barcode (over a field the barcode determines every page):
  // a bar of degree k from step b to step d is a generator of E^r at (b, k) and at (d, k + 1) for every r <= d - b,
  // cancelled by d^{d-b}; an essential bar is a generator of every page at (b, k). As phlite's `spectral_sequence()`.
  import type { Bar } from '../../lib/types';
  import { fmt } from '../../lib/format';

  let { bars, values, topDim }: { bars: Bar[]; values: number[]; topDim: number } = $props();

  /** The distinct filtration values: step p is the p-th of them. */
  const levels = $derived([...new Set(values)].sort((a, b) => a - b));
  const step = (v: number) => levels.indexOf(v);
  let r = $state(1);
  /** The last page on which anything changes (after it, E^r = E^∞). */
  const last = $derived(Math.max(1, ...bars.filter((b) => b.death !== null).map((b) => step(b.death!) - step(b.birth))));
  const page = $derived.by(() => {
    const table = levels.map(() => new Array(topDim + 1).fill(0));
    for (const b of bars) {
      if (b.kind) continue;
      const birth = step(b.birth);
      if (b.death === null) {
        table[birth][b.dim] += 1;
        continue;
      }
      const length = step(b.death) - birth;
      if (length >= r) {
        table[birth][b.dim] += 1;
        if (b.dim + 1 <= topDim) table[step(b.death)][b.dim + 1] += 1;
      }
    }
    return table;
  });
</script>

<div class="spectral">
  <div class="head">
    <strong>Spectral sequence</strong>
    <label>page E<sup>{r > last ? '∞' : r}</sup>
      <input type="range" min="1" max={last + 1} bind:value={r} /></label>
  </div>
  <div class="scroll">
    <table>
      <thead><tr><th>step (value) \ degree</th>{#each Array(topDim + 1) as _, k}<th>{k}</th>{/each}</tr></thead>
      <tbody>
        {#each page as row, p}
          <tr><th>{p} ({fmt(levels[p], 3)})</th>{#each row as n}<td class:zero={n === 0}>{n}</td>{/each}</tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p class="note">
    The dimension of each entry E<sup>r</sup><sub>p</sub> of the filtration's spectral sequence, from the barcode: page
    r sees differentials across r steps, and from page {last + 1} on nothing changes (E<sup>∞</sup>, whose rows add up
    to the homology). As <code>FilteredComplex.spectral_sequence()</code>, which also gives groups, generators and the
    differentials, over Z too.
  </p>
</div>

<style>
  .spectral { display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.9rem; }
  .head { display: flex; gap: 1rem; align-items: center; }
  .scroll { overflow-x: auto; max-height: 20rem; overflow-y: auto; }
  table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { padding: 0.15rem 0.5rem; text-align: right; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; }
  td.zero { opacity: 0.4; }
</style>
