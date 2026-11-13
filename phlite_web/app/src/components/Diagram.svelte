<script lang="ts">
  import { dimColour, type Bar } from '../lib/types';
  import { ticks, H } from '../lib/scale';
  import { fmt } from '../lib/format';

  let {
    bars,
    domain,
    selected = null,
    t = null,
    reversed = false,
    hidden = new Set<number>(),
    onselect,
  }: {
    bars: Bar[];
    domain: [number, number];
    selected?: number | null;
    t?: number | null;
    reversed?: boolean;
    hidden?: Set<number>;
    onselect?: (index: number | null) => void;
  } = $props();

  let size = $state(300);
  const pad = 42;
  const top = 18; // room for the line of essential classes (at y = infY)
  const infY = 8;
  const plot = $derived(size - pad - 8);
  const s = (v: number) => {
    const f = (v - domain[0]) / (domain[1] - domain[0] || 1);
    return (reversed ? 1 - f : f) * plot;
  };
  const px = (v: number) => pad + s(v);
  const py = (v: number | null) => (v === null ? infY : top + plot - s(v));
  // Draw the selected bar last, so it's on top
  const order = $derived.by(() => {
    const idx = bars.map((_, i) => i).filter((i) => !hidden.has(bars[i].dim));
    return selected === null ? idx : [...idx.filter((i) => i !== selected), selected];
  });
</script>

<div class="diagram" bind:clientWidth={size}>
  <svg width={size} height={top + plot + pad} role="img" aria-label="Persistence diagram">
    <rect x="0" y="0" width={size} height={top + plot + pad} fill="transparent" role="presentation" onclick={() => onselect?.(null)} />
    <line class="diag" x1={px(domain[0])} y1={py(domain[0])} x2={px(domain[1])} y2={py(domain[1])} />
    <line class="inf" x1={pad} x2={size - 8} y1={infY} y2={infY} />
    <text class="tick" x={pad - 4} y={infY + 3} text-anchor="end">∞</text>
    {#if t !== null}
      <line class="tline" x1={px(t)} x2={px(t)} y1={infY} y2={py(t)} />
      <line class="tline" x1={pad} x2={px(t)} y1={py(t)} y2={py(t)} />
    {/if}
    {#each ticks(domain[0], domain[1], 4) as v}
      <text class="tick" x={px(v)} y={top + plot + 13} text-anchor="middle">{fmt(v)}</text>
      <text class="tick" x={pad - 4} y={py(v) + 3} text-anchor="end">{fmt(v)}</text>
    {/each}
    <text class="label" x={pad + plot / 2} y={top + plot + 26} text-anchor="middle">birth</text>
    <text class="label" x="0" y="0" text-anchor="middle" transform="translate(9,{top + plot / 2}) rotate(-90)">death</text>
    {#each order as i (i)}
      {@const b = bars[i]}
      <circle
        cx={px(b.birth)}
        cy={py(b.death)}
        r={selected === i ? 6 : 3.5}
        fill={dimColour(b.dim)}
        class:sel={selected === i}
        role="button"
        tabindex="-1"
        onclick={(e) => { e.stopPropagation(); onselect?.(selected === i ? null : i); }}
        onkeydown={() => {}}
      >
        <title>{H(b.dim)}: [{fmt(b.birth)}, {fmt(b.death)})</title>
      </circle>
    {/each}
  </svg>
</div>

<style>
  .diagram { width: 100%; max-width: 440px; }
  svg { display: block; overflow: visible; }
  .diag { stroke: var(--muted); }
  .inf { stroke: var(--line); stroke-dasharray: 4 3; }
  .tick { font-size: 10px; fill: var(--muted); }
  .label { font-size: 11px; fill: var(--muted); }
  circle { cursor: pointer; fill-opacity: 0.8; }
  circle.sel { stroke: var(--fg); stroke-width: 2; fill-opacity: 1; }
  .tline { stroke: var(--fg); stroke-dasharray: 3 2; opacity: 0.6; pointer-events: none; }
</style>
