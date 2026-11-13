<script lang="ts">
  import Barcode from './Barcode.svelte';
  import Diagram from './Diagram.svelte';
  import { dimColour, type Bar, type RunResult } from '../lib/types';
  import { H, persistence } from '../lib/scale';
  import { fmt, duration, download } from '../lib/format';
  import { engine } from '../lib/engine.svelte';

  let {
    result,
    error,
    running,
    domain,
    selected = $bindable(null),
    t = $bindable(null),
    reversed = false,
    describe = (cell: number[]) => cell.join(','),
    name = 'barcode',
  }: {
    result: RunResult | null;
    error: string | null;
    running: boolean;
    domain: [number, number];
    selected?: number | null;
    t?: number | null;
    reversed?: boolean;
    /** A cell of dimension `dim`, in words. */
    describe?: (cell: number[], dim: number) => string;
    name?: string;
  } = $props();

  let hidden = $state(new Set<number>());
  let plot = $state<'barcode' | 'diagram'>('barcode');
  const bars = $derived(result?.barcode.bars ?? []);
  const dims = $derived([...new Set(bars.map((b) => b.dim))].sort((a, b) => a - b));
  const bar = $derived(selected === null ? null : bars[selected] ?? null);

  function toggle(d: number) {
    const next = new Set(hidden);
    if (next.has(d)) next.delete(d);
    else next.add(d);
    hidden = next;
  }

  function save() {
    download(`${name}.json`, JSON.stringify({ summary: result?.barcode.summary, bars }, null, 1));
  }

  const repLimit = 40;
  const repText = (b: Bar) =>
    (b.rep ?? [])
      .slice(0, repLimit)
      .map(([c, cell]) => `${c === 1 ? '' : c + '·'}${describe(cell, b.dim)}`)
      .join(' + ') + ((b.rep?.length ?? 0) > repLimit ? ` + … (${b.rep!.length - repLimit} more)` : '');
</script>

<div class="results">
  <div class="status">
    {#if running}
      <span class="spinner"></span> Computing…
      <button onclick={() => engine.cancel()}>Cancel</button>
    {:else if error}
      <span class="error">{error}</span>
    {:else if result}
      <span>{result.barcode.summary}</span>
      <span class="time">{duration(result.ms)}</span>
    {:else}
      <span class="muted">No barcode yet.</span>
    {/if}
  </div>

  {#if result && !error}
    <div class="toolbar">
      <div class="chips">
        {#each dims as d}
          <button class:on={!hidden.has(d)} style="color: {dimColour(d)}" onclick={() => toggle(d)}>
            {H(d)} · {bars.filter((b) => b.dim === d).length}
          </button>
        {/each}
      </div>
      <div class="switch" role="tablist" aria-label="Plot">
        <button role="tab" aria-selected={plot === 'barcode'} class:on={plot === 'barcode'} onclick={() => (plot = 'barcode')}>Barcode</button>
        <button role="tab" aria-selected={plot === 'diagram'} class:on={plot === 'diagram'} onclick={() => (plot = 'diagram')}>Diagram</button>
      </div>
      <button class="link" onclick={save}>Download JSON</button>
    </div>
    {#if bars.length === 0}
      <p class="muted">Every bar has length zero.</p>
    {:else}
      <div class="plots" class:stale={running}>
        {#if plot === 'barcode'}
          <div class="bc">
            <Barcode {bars} {domain} {selected} {t} {reversed} {hidden} onselect={(i) => (selected = i)} onscale={(v) => (t = v)} />
          </div>
        {:else}
          <div class="pd">
            <Diagram {bars} {domain} {selected} {t} {reversed} {hidden} onselect={(i) => (selected = i)} />
          </div>
        {/if}
      </div>
      <div class="detail">
        {#if bar}
          <div>
            <strong style="color: {dimColour(bar.dim)}">{H(bar.dim)}</strong>
            [{fmt(bar.birth, 4)}, {fmt(bar.death, 4)}) · persistence {fmt(persistence(bar), 4)}
            <button class="link" onclick={() => (t = bar.birth)}>go to birth</button>
            {#if bar.death !== null}<button class="link" onclick={() => (t = bar.death)}>go to death</button>{/if}
          </div>
          <div class="muted">
            born with {describe(bar.birth_cell, bar.dim)}{bar.death_cell ? `, killed by ${describe(bar.death_cell, bar.dim + 1)}` : ', never dies'}
          </div>
          {#if bar.rep}
            <div class="rep"><span class="muted">Representative ({bar.rep.length} cell{bar.rep.length === 1 ? '' : 's'}):</span> {repText(bar)}</div>
          {:else}
            <div class="muted">Turn on representatives to see a cycle for this bar.</div>
          {/if}
        {:else}
          <span class="muted">Click a bar (or a point of the diagram) to see its representative; drag across the barcode to move through the filtration.</span>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<style>
  .results { display: flex; flex-direction: column; gap: 0.6rem; }
  .status { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; min-height: 1.8rem; font-size: 0.9rem; }
  .time { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--muted); }
  .toolbar { display: flex; align-items: center; gap: 0.5rem; }
  .toolbar .link { font-size: 0.85rem; }
  .switch { margin-left: auto; display: flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .switch button { border: none; border-radius: 0; padding: 0.15rem 0.6rem; font-size: 0.85rem; background: var(--panel); }
  .switch button.on { background: color-mix(in srgb, var(--accent) 18%, var(--panel)); font-weight: 600; }
  .chips button { font-weight: 600; }
  .chips button:not(.on) { opacity: 0.45; }
  .plots.stale { opacity: 0.5; }
  .pd { display: flex; justify-content: center; }
  .bc { max-height: 520px; overflow-y: auto; overflow-x: hidden; }
  .detail { font-size: 0.88rem; border-top: 1px solid var(--line); padding-top: 0.5rem; display: flex; flex-direction: column; gap: 0.2rem; }
  .detail .link { font-size: 0.82rem; margin-left: 0.4rem; }
  .rep { max-height: 6rem; overflow-y: auto; word-break: break-word; }
  .spinner { width: 0.9rem; height: 0.9rem; border: 2px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
