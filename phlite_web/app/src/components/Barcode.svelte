<script lang="ts">
  import { dimColour, type Bar } from '../lib/types';
  import { ticks, alive, persistence, H } from '../lib/scale';
  import { fmt } from '../lib/format';

  let {
    bars,
    domain,
    selected = null,
    t = null,
    reversed = false,
    hidden = new Set<number>(),
    limit = 80,
    onselect,
    onscale,
  }: {
    bars: Bar[];
    domain: [number, number];
    selected?: number | null;
    t?: number | null;
    reversed?: boolean;
    hidden?: Set<number>;
    limit?: number;
    onselect?: (index: number | null) => void;
    onscale?: (t: number) => void;
  } = $props();

  let width = $state(400);
  const left = 8;
  const right = 18;
  const header = 18;
  const axisHeight = 24;

  // Per dimension, the `limit` longest bars, shown in order of birth
  const groups = $derived.by(() => {
    const byDim = new Map<number, number[]>();
    bars.forEach((b, i) => {
      if (hidden.has(b.dim)) return;
      if (!byDim.has(b.dim)) byDim.set(b.dim, []);
      byDim.get(b.dim)!.push(i);
    });
    return [...byDim.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([dim, idx]) => {
        const kept = [...idx].sort((a, b) => persistence(bars[b]) - persistence(bars[a])).slice(0, limit);
        if (selected !== null && bars[selected]?.dim === dim && !kept.includes(selected)) kept.push(selected);
        kept.sort((a, b) => (reversed ? -1 : 1) * (bars[a].birth - bars[b].birth) || persistence(bars[b]) - persistence(bars[a]));
        return { dim, shown: kept, total: idx.length };
      });
  });
  const totalShown = $derived(groups.reduce((s, g) => s + g.shown.length, 0));
  const step = $derived(Math.max(3, Math.min(10, 420 / Math.max(totalShown, 1))));
  const layout = $derived.by(() => {
    let y = 0;
    return groups.map((g) => {
      const top = y;
      y += header + g.shown.length * step + 6;
      return { ...g, top };
    });
  });
  const height = $derived((layout.length ? layout[layout.length - 1].top + header + layout[layout.length - 1].shown.length * step + 6 : 40) + axisHeight);

  const x = (v: number) => {
    const f = (v - domain[0]) / (domain[1] - domain[0] || 1);
    return left + (reversed ? 1 - f : f) * (width - left - right);
  };
  const invert = (px: number) => {
    let f = (px - left) / (width - left - right);
    if (reversed) f = 1 - f;
    return domain[0] + Math.min(1, Math.max(0, f)) * (domain[1] - domain[0]);
  };
  const end = (b: Bar) => (b.death === null ? (reversed ? left : width - right + 10) : x(b.death));

  let dragging = false;
  function scaleAt(e: PointerEvent) {
    const svg = (e.currentTarget as SVGElement).closest('svg')!;
    const px = e.clientX - svg.getBoundingClientRect().left;
    onscale?.(invert(px));
  }
</script>

<div class="barcode" bind:clientWidth={width}>
  <svg {width} {height} role="img" aria-label="Barcode">
    <rect
      class="bg"
      x="0" y="0" {width} {height}
      role="presentation"
      onpointerdown={(e) => { dragging = true; (e.currentTarget as Element).setPointerCapture(e.pointerId); scaleAt(e); onselect?.(null); }}
      onpointermove={(e) => dragging && scaleAt(e)}
      onpointerup={() => (dragging = false)}
    />
    {#each layout as g (g.dim)}
      <text class="head" x={left} y={g.top + 12} fill={dimColour(g.dim)}>
        {H(g.dim)} · {g.total} bar{g.total === 1 ? '' : 's'}{g.shown.length < g.total ? ` (the ${limit} longest shown)` : ''}
      </text>
      {#each g.shown as i, row (i)}
        {@const b = bars[i]}
        {@const y = g.top + header + row * step}
        {@const x0 = x(b.birth)}
        {@const x1 = end(b)}
        <g
          class="bar"
          class:faded={t !== null && !alive(b, t, reversed)}
          class:sel={selected === i}
          role="button"
          tabindex="-1"
          onclick={() => onselect?.(selected === i ? null : i)}
          onkeydown={() => {}}
        >
          <rect x={Math.min(x0, x1)} y={y - 1} width={Math.max(Math.abs(x1 - x0), 1.5)} height={Math.max(step - 1, 2) + 2} fill="transparent" />
          <rect x={Math.min(x0, x1)} y={y + 0.5} width={Math.max(Math.abs(x1 - x0), 1.5)} height={Math.max(step - 1.5, 1.5)} fill={dimColour(b.dim)} rx="1" />
          {#if b.death === null}
            <path d="M{x1},{y + step / 2} l{reversed ? 6 : -6},{-step / 2 - 1} v{step + 2} z" fill={dimColour(b.dim)} />
          {/if}
          <title>{H(b.dim)}: [{fmt(b.birth)}, {fmt(b.death)}){b.rep ? ` · representative of ${b.rep.length} cells` : ''}</title>
        </g>
      {/each}
    {/each}
    <g transform="translate(0,{height - axisHeight})">
      <line x1={left} x2={width - right} y1="4" y2="4" class="axis" />
      {#each ticks(domain[0], domain[1], Math.max(2, Math.floor(width / 80))) as v}
        <line x1={x(v)} x2={x(v)} y1="4" y2="8" class="axis" />
        <text x={x(v)} y="19" class="tick">{fmt(v)}</text>
      {/each}
    </g>
    {#if t !== null}
      <line class="tline" x1={x(t)} x2={x(t)} y1="0" y2={height - axisHeight + 4} />
    {/if}
  </svg>
</div>

<style>
  .barcode { width: 100%; user-select: none; }
  svg { display: block; overflow: visible; }
  .bg { fill: transparent; cursor: col-resize; }
  .head { font-size: 11px; font-weight: 600; }
  .bar { cursor: pointer; }
  .bar.faded { opacity: 0.25; }
  .bar.sel rect:last-of-type, .bar.sel path { stroke: var(--fg); stroke-width: 1.2; }
  .axis { stroke: var(--muted); }
  .tick { font-size: 10px; text-anchor: middle; fill: var(--muted); }
  .tline { stroke: var(--fg); stroke-dasharray: 3 2; pointer-events: none; }
</style>
