<script lang="ts">
  import { dimColour, type Bar } from '../../lib/types';
  import { fmt } from '../../lib/format';
  import type { Complex } from './data';

  let { c, values, bar, modulus }: { c: Complex; values: number[]; bar: Bar | null; modulus: number } = $props();

  let width = $state(500);
  // Cells in filtration order: by value, ties by index
  const order = $derived(c.map((_, i) => i).sort((a, b) => values[a] - values[b] || a - b));
  const rank = $derived(new Map(order.map((cell, r) => [cell, r])));
  const n = $derived(c.length);
  const margin = 46;
  const s = $derived(Math.max(4, Math.min(20, (width - margin - 8) / Math.max(n, 1))));
  const labels = $derived(s >= 12);
  const repCells = $derived(new Map((bar?.rep ?? []).map(([coeff, cell]) => [cell[0], coeff])));
  const reduce = (k: number) => ((k % modulus) + modulus) % modulus;
</script>

<div class="matrix" bind:clientWidth={width}>
  <svg width={margin + n * s + 4} height={margin + n * s + 4}>
    <!-- Column and row labels, in filtration order -->
    {#each order as cell, r}
      {@const highlight = repCells.has(cell) ? dimColour(bar!.dim) : null}
      {#if highlight}
        <rect x={margin + r * s} y={margin} width={s} height={n * s} fill={highlight} opacity="0.15" />
        <rect x={margin} y={margin + r * s} width={n * s} height={s} fill={highlight} opacity="0.15" />
      {/if}
      {#if labels}
        <text class="lab" x={margin + r * s + s / 2} y={margin - 4} transform="rotate(-60 {margin + r * s + s / 2} {margin - 4})">{c[cell].label}</text>
        <text class="lab row" x={margin - 4} y={margin + r * s + s * 0.7}>{c[cell].label}</text>
      {/if}
    {/each}
    <rect x={margin} y={margin} width={n * s} height={n * s} class="frame" />
    {#each order as cell, col}
      {#each c[cell].faces as f}
        {@const row = rank.get(f.cell) ?? 0}
        {@const k = reduce(f.coeff)}
        {#if k !== 0}
          {#if modulus === 2 || !labels}
            <rect x={margin + col * s + s * 0.2} y={margin + row * s + s * 0.2} width={s * 0.6} height={s * 0.6} class="entry" rx="1" />
          {:else}
            <text class="num" x={margin + col * s + s / 2} y={margin + row * s + s * 0.72}>{k}</text>
          {/if}
        {/if}
      {/each}
    {/each}
    {#if bar}
      {@const b = rank.get(bar.birth_cell[0])}
      {#if b !== undefined}<rect x={margin} y={margin + b * s} width={n * s} height={s} class="birth" />{/if}
      {#if bar.death_cell}
        {@const d = rank.get(bar.death_cell[0])}
        {#if d !== undefined}<rect x={margin + d * s} y={margin} width={s} height={n * s} class="death" />{/if}
      {/if}
    {/if}
  </svg>
  <p class="note">
    The boundary matrix (over Z/{modulus}), columns and rows in filtration order{labels ? '' : ' (too many cells for labels)'}.
    {#if bar}Shaded: the representative's cells; outlined: the birth cell's row and the death cell's column.{/if}
    Values run from {fmt(values[order[0]])} to {fmt(values[order[n - 1]])}.
  </p>
</div>

<style>
  .matrix { width: 100%; overflow-x: auto; }
  svg { display: block; }
  .frame { fill: none; stroke: var(--line); }
  .entry { fill: var(--fg); }
  .num { font-size: 10px; text-anchor: middle; fill: var(--fg); }
  .lab { font-size: 10px; fill: var(--muted); }
  .lab.row { text-anchor: end; }
  .birth { fill: none; stroke: var(--accent); stroke-width: 1.5; }
  .death { fill: none; stroke: #e45756; stroke-width: 1.5; stroke-dasharray: 3 2; }
</style>
