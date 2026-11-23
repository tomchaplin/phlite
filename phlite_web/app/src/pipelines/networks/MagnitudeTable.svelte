<script lang="ts">
  import BarsSpinner from '../../components/BarsSpinner.svelte';
  import { duration } from '../../lib/format';
  import { engine } from '../../lib/engine.svelte';
  import type { RunResult } from '../../lib/types';

  let { result, error, running }: { result: RunResult | null; error: string | null; running: boolean } = $props();

  const table = $derived(result?.barcode.magnitude ?? null);
  const width = $derived(table ? Math.max(...table.ranks.map((row) => row.length)) : 0);
  const torsion = $derived(new Map((table?.torsion ?? []).map(([k, l, orders]) => [`${k},${l}`, orders])));
  /** sum_k (-1)^k rank MH_{k,l}: the coefficient of q^l in the magnitude. */
  const euler = $derived(
    table?.ranks.map((row) => (row.some((r) => r === null) ? null : row.reduce((s: number, r, k) => s + (k % 2 ? -1 : 1) * (r ?? 0), 0))) ?? [],
  );
  function entry(k: number, l: number): string {
    const r = table!.ranks[l][k];
    if (r === undefined) return '';
    let text = r === null ? '?' : String(r);
    for (const order of torsion.get(`${k},${l}`) ?? []) text += ` ⊕ ℤ/${order}`;
    return text;
  }
</script>

<div class="results">
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
  {#if table && !error}
    <div class="scroll">
      <table>
        <thead>
          <tr><th>l \ k</th>{#each Array(width) as _, k}<th>{k}</th>{/each}<th class="euler" title="Σₖ (−1)ᵏ rank: the coefficient of qˡ in the magnitude">χ</th></tr>
        </thead>
        <tbody>
          {#each table.ranks as row, l}
            <tr>
              <th>{l}</th>
              {#each Array(width) as _, k}
                <td class:zero={row[k] === 0 && !torsion.has(`${k},${l}`)} class:torsion={torsion.has(`${k},${l}`)}>{k < row.length ? entry(k, l) : ''}</td>
              {/each}
              <td class="euler">{euler[l] ?? '?'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <p class="note">
      The rank of magnitude homology MH<sub>k,l</sub> (Hepworth and Willerton) in degree k (across) and length l (down),
      {table.integral ? 'over the integers, with its torsion' : 'over ℤ/2'}; the last column is the coefficient of qˡ in
      the graph's magnitude. Every edge has length 1. As <code>phlite.networks.magnitude</code>.
    </p>
  {/if}
</div>

<style>
  .results { display: flex; flex-direction: column; gap: 0.6rem; }
  .status { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; min-height: 1.8rem; font-size: 0.9rem; }
  .time { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--muted); }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.9rem; }
  th, td { padding: 0.2rem 0.55rem; text-align: right; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; }
  td.zero { color: var(--muted); opacity: 0.6; }
  td.torsion { color: var(--accent); font-weight: 600; }
  .euler { border-left: 1px solid var(--line); }
</style>
