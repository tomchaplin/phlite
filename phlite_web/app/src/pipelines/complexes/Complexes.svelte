<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import ComplexView, { type Tool } from './ComplexView.svelte';
  import MatrixView from './MatrixView.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { COMPLEX, defaultsFor } from '../../lib/options';
  import { barDomain } from '../../lib/scale';
  import { randomSeed } from '../../lib/random';
  import type { Job } from '../../lib/types';
  import { EXAMPLES, FORMAT_HELP, randomComplex, toText, parseText, effectiveValues, drawable, vertexSets, type Complex } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let c = $state.raw<Complex>(EXAMPLES[0].make());
  let modulus = $state(2);
  let maxDim = $state<number | null>(null);
  let options = $state(defaultsFor(COMPLEX));
  const runner = new Runner();

  function useExample(id: string) {
    const ex = EXAMPLES.find((e) => e.id === id)!;
    exampleId = id;
    c = ex.make();
    modulus = ex.modulus;
  }

  let randomN = $state(30);
  let randomRadius = $state(0.3);
  let randomValues = $state<'rips' | 'height'>('rips');
  let seed = $state(1);
  function useRandom(newSeed = false) {
    if (newSeed) seed = randomSeed();
    c = randomComplex(randomN, randomRadius, randomValues, seed);
    // Without 3-cells, every tetrahedron's four triangles would make a spurious H₂ class
    maxDim = 1;
  }

  let uploadError = $state<string | null>(null);
  async function upload(file: File) {
    try {
      c = parseText(await file.text());
      uploadError = null;
    } catch (e) {
      uploadError = (e as Error).message;
    }
  }

  // ---- Building: by drawing, or by editing the text ----
  let mode = $state<'draw' | 'text'>('draw');
  let tool = $state<Tool>('vertex');
  let valuesBy = $state<'order' | 'length' | 'height' | 'text'>('order');
  let text = $state('');
  let textError = $state<string | null>(null);
  function startBuilding() {
    if (!drawable(c)) c = [];
    text = toText(c);
  }
  $effect(() => {
    if (source === 'draw') untrack(startBuilding);
  });
  function assignValues(cells: Complex): Complex {
    if (valuesBy === 'text') return cells;
    const verts = vertexSets(cells);
    return cells.map((x, i) => {
      let value: number | null = null;
      if (valuesBy === 'order') value = i;
      else if (valuesBy === 'height') value = x.dim === 0 && x.pos ? Math.round(100 * x.pos[1]) / 100 : null;
      else if (x.dim > 0) {
        let m = 0;
        for (const a of verts[i]) for (const b of verts[i]) m = Math.max(m, Math.hypot(cells[a].pos![0] - cells[b].pos![0], cells[a].pos![1] - cells[b].pos![1]));
        value = Math.round(1000 * m) / 1000;
      } else value = 0;
      return { ...x, value };
    });
  }
  function edited(cells: Complex) {
    c = assignValues(cells);
    text = toText(c);
  }
  function applyText() {
    try {
      c = parseText(text);
      valuesBy = 'text';
      textError = null;
    } catch (e) {
      textError = (e as Error).message;
    }
  }

  // ---- Computing ----
  const job = $derived.by((): Job | null => {
    if (c.length === 0) return null;
    const indptr = [0];
    const indices: number[] = [];
    const coefficients: number[] = [];
    for (const x of c) {
      for (const f of x.faces) {
        indices.push(f.cell);
        coefficients.push(f.coeff);
      }
      indptr.push(indices.length);
    }
    return {
      pipeline: 'complex',
      dimensions: Uint32Array.from(c.map((x) => x.dim)),
      indptr: Uint32Array.from(indptr),
      indices: Uint32Array.from(indices),
      coefficients: Int32Array.from(coefficients),
      values: Float64Array.from(c.map((x) => x.value ?? NaN)),
      modulus,
      maxDim,
      options: $state.snapshot(options),
    };
  });
  // Moving a vertex only changes values when they come from lengths or heights
  const jobKey = $derived(JSON.stringify([c.map((x) => [x.dim, x.faces, x.value]), modulus, maxDim, options]));
  $effect(() => {
    jobKey;
    const j = untrack(() => job);
    if (runner.auto) untrack(() => runner.schedule(j));
  });

  const values = $derived(effectiveValues(c));
  const canDraw = $derived(drawable(c) || source === 'draw');
  const domain = $derived(barDomain(runner.result?.barcode.bars ?? [], values));
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
  const describe = (cell: number[]) => c[cell[0]]?.label ?? `cell ${cell[0]}`;
  const topDim = $derived(Math.max(0, ...c.map((x) => x.dim)));
</script>

<div class="layout">
  <div>
    <div class="panel">
      {#if canDraw}
        <ComplexView {c} {values} t={runner.t} {bar} tool={source === 'draw' && mode === 'draw' ? tool : null} onedit={edited} />
        <div style="margin: 0.6rem 0">
          <FiltrationSlider bind:t={runner.t} {domain} label="Value" />
        </div>
      {/if}
      {#if c.length > 0 && c.length <= 90}
        <MatrixView {c} {values} {bar} {modulus} />
      {:else if c.length > 90}
        <p class="note">{c.length} cells: too many to draw the boundary matrix.</p>
      {/if}
      {#if !canDraw}
        <p class="note">This complex has no positions for its vertices (or cells above dimension 2), so only its matrix is drawn.</p>
      {/if}
    </div>

    <div class="panel secondary">
      <h3>Chain complex</h3>
      <SourceTabs bind:source drawLabel="Build" />
      {#if source === 'examples'}
        <div class="chips">
          {#each EXAMPLES as ex}
            <button class:on={exampleId === ex.id} onclick={() => useExample(ex.id)}>{ex.name}</button>
          {/each}
        </div>
        <p class="note">{EXAMPLES.find((e) => e.id === exampleId)?.about}</p>
      {:else if source === 'random'}
        <label class="field"><span>Vertices</span><input type="range" min="4" max="120" bind:value={randomN} onchange={() => useRandom()} /><span>{randomN}</span></label>
        <label class="field"><span>Edge length</span><input type="range" min="0.05" max="0.6" step="0.01" bind:value={randomRadius} onchange={() => useRandom()} /><span>≤ {randomRadius}</span></label>
        <label class="field"><span>Values</span>
          <select bind:value={randomValues} onchange={() => useRandom()}>
            <option value="rips">diameter (Rips-like)</option>
            <option value="height">random vertex heights (lower star)</option>
          </select>
        </label>
        <div class="field"><span>Seed {seed}</span><button onclick={() => useRandom(true)}>New complex</button></div>
        <p class="note">Random points, an edge for each close pair, and a triangle for every three mutually close points (no tetrahedra, so only H₀ and H₁ are meaningful).</p>
      {:else if source === 'upload'}
        <FileDrop accept=".txt,.cplx" onfile={upload}>
          {FORMAT_HELP}
          <pre>{`a 0 0 @ 0 0
b 0 0 @ 1 0
c 0 1 @ 0.5 0.8
ab 1 1 : -a b
bc 1 1 : -b c
ac 1 2 : -a c
abc 2 _ : ab bc -ac`}</pre>
        </FileDrop>
        {#if uploadError}<p class="error">{uploadError}</p>{/if}
      {:else}
        <div class="tools">
          <button class:on={mode === 'draw'} onclick={() => (mode = 'draw')}>Draw</button>
          <button class:on={mode === 'text'} onclick={() => { mode = 'text'; text = toText(c); }}>Edit as text</button>
        </div>
        {#if mode === 'draw'}
          <div class="tools" style="margin-top: 0.4rem">
            <button class:on={tool === 'vertex'} onclick={() => (tool = 'vertex')}>Vertices</button>
            <button class:on={tool === 'edge'} onclick={() => (tool = 'edge')}>Edges</button>
            <button class:on={tool === 'triangle'} onclick={() => (tool = 'triangle')}>Triangles</button>
            <button class:on={tool === 'move'} onclick={() => (tool = 'move')}>Move</button>
            <button class:on={tool === 'delete'} onclick={() => (tool = 'delete')}>Delete</button>
            <button onclick={() => edited([])}>Clear</button>
          </div>
          <label class="field"><span>Values</span>
            <select bind:value={valuesBy} onchange={() => edited(c)}>
              <option value="order">the order cells were added</option>
              <option value="length">diameter (Rips-like)</option>
              <option value="height">vertex height (lower star)</option>
              <option value="text" disabled={valuesBy !== 'text'}>as edited in the text</option>
            </select>
          </label>
          <p class="note">
            {tool === 'vertex' ? 'Click to add a vertex.' : tool === 'edge' ? 'Click two vertices to join them.' : tool === 'triangle' ? 'Click three vertices to fill a triangle (missing edges are added).' : tool === 'move' ? 'Drag vertices.' : 'Click a cell to delete it and everything on it.'}
          </p>
        {:else}
          <textarea class="code" rows="12" bind:value={text} spellcheck="false"></textarea>
          <div class="compute-row" style="margin-top: 0.3rem">
            <button onclick={applyText}>Apply</button>
            <span class="note" style="margin: 0">{FORMAT_HELP.split('\n')[0]}</span>
          </div>
          {#if textError}<p class="error">{textError}</p>{/if}
        {/if}
      {/if}
    </div>
  </div>

  <div>
    <div class="panel">
      <Results result={runner.result} error={runner.error} running={runner.running} {domain} bind:selected={runner.selected} bind:t={runner.t} {describe} name="complex" />
    </div>

    <div class="panel secondary">
      <h3>Compute</h3>
      <label class="field"><span>Coefficients</span>
        <select bind:value={modulus}>
          {#each [2, 3, 5, 7] as p}<option value={p}>Z/{p}</option>{/each}
        </select>
      </label>
      <label class="field"><span>Up to dimension</span>
        <select bind:value={maxDim}>
          <option value={null}>all (H₀–H{['₀', '₁', '₂', '₃', '₄', '₅'][topDim] ?? topDim})</option>
          {#each Array.from({ length: topDim + 1 }, (_, d) => d) as d}<option value={d}>{d}</option>{/each}
        </select>
      </label>
      <Options bind:options defaults={COMPLEX} />
      <div class="compute-row">
        <button class="primary" disabled={!job} onclick={() => runner.run(job)}>Compute</button>
        <label><input type="checkbox" bind:checked={runner.auto} /> recompute on every change</label>
      </div>
    </div>
  </div>
</div>

<style>
  textarea { width: 100%; font-size: 0.85rem; margin-top: 0.4rem; }
</style>
