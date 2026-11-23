<script lang="ts">
  // The magnitude-path spectral sequence, a page at a time: each entry E^r_{k,l} as a group (its isomorphism class),
  // the page's non-zero differentials, and for a chosen entry a generator of each cyclic summand, as chains of tuples
  // of nodes (click a tuple to see it on the graph).
  import BarsSpinner from '../../components/BarsSpinner.svelte';
  import { duration } from '../../lib/format';
  import { engine } from '../../lib/engine.svelte';
  import type { MpssGroup, RunResult } from '../../lib/types';

  let {
    result,
    error,
    running,
    onhighlight,
  }: {
    result: RunResult | null;
    error: string | null;
    running: boolean;
    /** Show a tuple of nodes on the graph (or nothing). */
    onhighlight: (tuple: number[] | null) => void;
  } = $props();

  const mpss = $derived(result?.barcode.mpss ?? null);
  let pageIndex = $state(0);
  let chosen = $state<[number, number] | null>(null);
  let highlighted = $state<string | null>(null);
  $effect(() => {
    // A new result: back to the first page, nothing chosen
    result;
    pageIndex = 0;
    chosen = null;
    highlighted = null;
    onhighlight(null);
  });
  const page = $derived(mpss ? mpss.pages[Math.min(pageIndex, mpss.pages.length - 1)] : null);
  const width = $derived(page ? Math.max(...page.table.map((row) => row.length)) : 0);
  const SUP = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
  const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
  const sup = (n: number) => String(n).split('').map((c) => SUP[+c]).join('');
  const sub = (n: number) => String(n).split('').map((c) => SUB[+c]).join('');
  /** A group's isomorphism class: ℤ² ⊕ ℤ/2, or 𝔽₂³ over a field, or 0. */
  function name(g: MpssGroup | null): string {
    if (g === null) return '?';
    const field = mpss!.modulus !== 0;
    const parts: string[] = [];
    if (g.rank > 0) {
      const base = field ? `𝔽${sub(mpss!.modulus)}` : 'ℤ';
      parts.push(g.rank === 1 ? base : base + sup(g.rank));
    }
    // Torsion as invariant factors, repeated ones gathered
    const counts = new Map<string, number>();
    for (const t of g.torsion) counts.set(t, (counts.get(t) ?? 0) + 1);
    for (const [t, m] of counts) parts.push(m === 1 ? `ℤ/${t}` : `(ℤ/${t})${sup(m)}`);
    return parts.join(' ⊕ ') || '0';
  }
  const pageName = (i: number) => {
    const r = mpss!.pages[i].r;
    const last = i === mpss!.pages.length - 1 && mpss!.degenerates_at !== null && r >= mpss!.degenerates_at;
    return last ? `E${sup(r)} = E∞` : `E${sup(r)}`;
  };
  const entry = $derived(chosen && page ? (page.nonzero.find((e) => e.l === chosen![0] && e.k === chosen![1]) ?? null) : null);
  const outgoing = $derived(new Set(page?.differentials.map((d) => `${d.from[0]},${d.from[1]}`) ?? []));
  const incoming = $derived(new Set(page?.differentials.map((d) => `${d.to[0]},${d.to[1]}`) ?? []));
  const tuple = (t: number[]) => `(${t.join(',')})`;
  function coefficient(c: string, first: boolean): string {
    const n = Number(c);
    if (n === 1) return first ? '' : ' + ';
    if (n === -1) return first ? '−' : ' − ';
    if (n < 0) return `${first ? '−' : ' − '}${-n}·`;
    return `${first ? '' : ' + '}${c}·`;
  }
  function highlight(t: number[]) {
    const key = t.join(',');
    highlighted = highlighted === key ? null : key;
    onhighlight(highlighted === null ? null : t);
  }
</script>

<div class="mpss">
  <div class="status">
    {#if running}
      <BarsSpinner /> Computing…
      <button onclick={() => engine.cancel()}>Cancel</button>
    {:else if error}
      <span class="error">{error}</span>
    {:else if result}
      <span>{result.barcode.summary}</span>
      <span class="time">{duration(result.ms)}</span>
    {:else}
      <span class="muted">Nothing computed yet.</span>
    {/if}
  </div>

  {#if mpss && page && !error}
    <div class="pages" role="tablist" aria-label="Page">
      {#each mpss.pages as _, i}
        <button role="tab" aria-selected={i === pageIndex} class:on={i === pageIndex} onclick={() => (pageIndex = i)}>{pageName(i)}</button>
      {/each}
    </div>
    <div class="scroll">
      <table>
        <thead><tr><th>l \ k</th>{#each Array(width) as _, k}<th>{k}</th>{/each}</tr></thead>
        <tbody>
          {#each page.table as row, l}
            <tr>
              <th>{l}</th>
              {#each Array(width) as _, k}
                {@const g = k < row.length ? row[k] : undefined}
                {#if g === undefined}
                  <td></td>
                {:else}
                  {@const zero = g !== null && g.rank === 0 && g.torsion.length === 0}
                  <td
                    class:zero
                    class:unknown={g === null}
                    class:torsion={g !== null && g.torsion.length > 0}
                    class:chosen={chosen?.[0] === l && chosen?.[1] === k}
                    title={g === null ? `not determined by tuples up to length ${mpss.l_max}` : `E${sup(page.r)} in degree ${k}, length ${l}`}
                  >
                    {#if g !== null && !zero}
                      <button class="cell" onclick={() => ((chosen = [l, k]), (highlighted = null), onhighlight(null))}>{name(g)}</button>
                    {:else}{name(g)}{/if}
                    {#if outgoing.has(`${l},${k}`)}<span class="arrow" title="a non-zero differential leaves this entry">↙</span>{/if}
                    {#if incoming.has(`${l},${k}`)}<span class="arrow in" title="a non-zero differential arrives here">•</span>{/if}
                  </td>
                {/if}
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if page.differentials.length}
      <div class="diffs">
        <span class="muted">Differentials d{sup(page.r)}:</span>
        {#each page.differentials as d}
          <span>(k={d.from[1]}, l={d.from[0]}) → (k={d.to[1]}, l={d.to[0]}), image {name(d.image)}</span>
        {/each}
      </div>
    {:else}
      <div class="diffs muted">Every differential d{sup(page.r)} is zero: E{sup(page.r + 1)} = E{sup(page.r)}.</div>
    {/if}

    <div class="detail">
      {#if entry}
        <div>
          <strong>E{sup(page.r)} in degree {entry.k}, length {entry.l}: {name(entry.group)}</strong>
        </div>
        {#if entry.generators}
          <div class="muted">
            A generator for each cyclic summand (torsion first): a chain x of tuples of length {entry.l} with dx of length at most
            {entry.l - page.r}, determined modulo the classes that die by page {page.r}. Click a tuple to show it on the graph, each step
            along a shortest path.
          </div>
          <ol>
            {#each entry.generators as gen}
              <li>
                <span class="order">{gen.order === null ? (mpss.modulus === 0 ? 'free' : 'order ' + mpss.modulus) : `order ${gen.order}`}:</span>
                {#each gen.chain.slice(0, 60) as [c, t], i}
                  {coefficient(c, i === 0)}<button class="tuple" class:on={highlighted === t.join(',')} onclick={() => highlight(t)}>{tuple(t)}</button>
                {/each}
                {#if gen.chain.length > 60}<span class="muted"> + … ({gen.chain.length - 60} more terms)</span>{/if}
              </li>
            {/each}
          </ol>
        {:else}
          <div class="muted">Generators aren't computed for complexes of more than a few thousand tuples ({mpss.n_tuples} here): lower the length.</div>
        {/if}
      {:else}
        <span class="muted">Click a non-zero entry to see its generators.</span>
      {/if}
    </div>
    <p class="note">
      The magnitude-path spectral sequence (Asao; Hepworth and Roff) of the digraph, every edge of length 1: E¹ is magnitude
      homology, E² bigraded path homology (its diagonal k = l is path homology), and the pages converge to reachability
      homology. Entries with l + r &gt; {mpss.l_max + 1} need longer tuples and show as ?. As
      <code>phlite.networks.magnitude_path_spectral_sequence</code>.
    </p>
  {/if}
</div>

<style>
  .mpss { display: flex; flex-direction: column; gap: 0.6rem; font-size: 0.9rem; }
  .status { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; min-height: 1.8rem; }
  .time { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--muted); }
  .pages { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .pages button.on { background: color-mix(in srgb, var(--accent) 18%, var(--panel)); font-weight: 600; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { padding: 0.2rem 0.5rem; text-align: center; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; }
  td.zero { opacity: 0.35; }
  td.unknown { color: var(--muted); opacity: 0.6; }
  td.torsion .cell { color: var(--accent); font-weight: 700; }
  td.chosen { background: color-mix(in srgb, var(--accent) 15%, transparent); }
  .cell { border: none; background: none; padding: 0; font: inherit; cursor: pointer; text-decoration: underline dotted; }
  .arrow { font-size: 0.75rem; color: var(--accent); margin-left: 0.15rem; }
  .diffs { display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; }
  .detail { border-top: 1px solid var(--line); padding-top: 0.5rem; display: flex; flex-direction: column; gap: 0.3rem; }
  ol { margin: 0; padding-left: 1.4rem; max-height: 14rem; overflow-y: auto; }
  li { margin: 0.15rem 0; line-height: 1.6; word-break: break-word; }
  .order { color: var(--muted); margin-right: 0.3rem; }
  .tuple { border: 1px solid var(--line); border-radius: 4px; padding: 0 0.25rem; font: inherit; font-size: 0.85rem; background: var(--panel); }
  .tuple.on { background: color-mix(in srgb, var(--accent) 25%, var(--panel)); }
</style>
