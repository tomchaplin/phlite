<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import GraphView, { type Tool } from './GraphView.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { GRPPH, NETWORK, defaultsFor } from '../../lib/options';
  import { barDomain } from '../../lib/scale';
  import { randomSeed } from '../../lib/random';
  import type { Job, NetworkKind } from '../../lib/types';
  import { EXAMPLES, randomDigraph, parseEdges, layout, shortestPaths, type Digraph } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let g = $state.raw<Digraph>(EXAMPLES[0].make());
  type Complex = 'grpph' | 'pph' | NetworkKind;
  const COMPLEXES: { id: Complex; name: string; about: string }[] = [
    {
      id: 'grpph',
      name: 'GrPPH',
      about:
        "Grounded persistent path homology: the path homology of the shortest-path filtration, grounded at the graph's own edges (which are there from the start). H₀ and H₁; representatives are sums of nodes or of directed edges.",
    },
    {
      id: 'pph',
      name: 'PPH',
      about:
        'Persistent path homology: the path homology of the shortest-path filtration, a pair a→b entering at the length of the shortest path from a to b. H₀ and H₁.',
    },
    {
      id: 'flag',
      name: 'Directed flag',
      about:
        "The directed flag complex (as flagser): a simplex for each directed clique, a sequence of distinct nodes with an edge from each to every later one, entering with its last edge (at the edge's weight).",
    },
    {
      id: 'flag-sp',
      name: 'Directed flag (paths)',
      about: 'The directed flag complex of the shortest-path filtration: each pair a→b at the length of the shortest path from a to b.',
    },
    {
      id: 'grounded-flag',
      name: 'Grounded flag',
      about:
        "The directed flag complex of the shortest-path filtration, grounded at the graph's own edges (there from the start): H₁ bars are the graph's own cycles, born at 0, until they are filled in. H₀ and H₁.",
    },
    {
      id: 'tuples',
      name: 'Ordered tuples',
      about:
        'Directed Rips (Turner): the directed flag complex with repeated nodes allowed, so long as no node is next to itself, so a pair of opposite edges is filled in by (a, b, a) as soon as both appear.',
    },
    {
      id: 'dowker-sink',
      name: 'Dowker sink',
      about:
        'The Dowker sink complex: a set of nodes enters at t once some node x is within t of each of them (an edge of weight at most t to x, or x itself). Dense: best on small graphs.',
    },
    {
      id: 'dowker-source',
      name: 'Dowker source',
      about: 'The Dowker source complex: as the sink complex with the edges reversed. By Dowker duality, the same barcode as the sink complex.',
    },
  ];
  let complex = $state<Complex>('grpph');
  let maxDim = $state(1);
  const pathHomology = $derived(complex === 'grpph' || complex === 'pph');
  const shortestPath = $derived(pathHomology || complex === 'flag-sp' || complex === 'grounded-flag');
  let options = $state(defaultsFor(GRPPH));
  const runner = new Runner();
  function setComplex(c: Complex) {
    complex = c;
    options = defaultsFor(c === 'grpph' || c === 'pph' ? GRPPH : NETWORK);
  }

  function useExample(id: string) {
    exampleId = id;
    g = EXAMPLES.find((e) => e.id === id)!.make();
  }

  let randomN = $state(12);
  let randomP = $state(0.2);
  let minW = $state(0.5);
  let maxW = $state(3);
  let seed = $state(1);
  function useRandom(newSeed = false) {
    if (newSeed) seed = randomSeed();
    g = randomDigraph(randomN, randomP, Math.min(minW, maxW), Math.max(minW, maxW), seed);
  }

  let uploadError = $state<string | null>(null);
  async function upload(file: File) {
    try {
      g = parseEdges(await file.text());
      uploadError = null;
    } catch (e) {
      uploadError = (e as Error).message;
    }
  }

  let tool = $state<Tool>('node');
  let newWeight = $state(1);
  let selectedEdge = $state<number | null>(null);
  function setWeight(w: number) {
    if (selectedEdge === null || !(w >= 0)) return;
    g = { ...g, edges: g.edges.map((e, i) => (i === selectedEdge ? { ...e, w } : e)) };
  }

  // ---- Computing ----
  const job = $derived.by((): Job | null => {
    if (g.nodes.length === 0) return null;
    const graph = {
      n: g.nodes.length,
      sources: Uint32Array.from(g.edges.map((e) => e.s)),
      targets: Uint32Array.from(g.edges.map((e) => e.t)),
      weights: Float64Array.from(g.edges.map((e) => e.w)),
      options: $state.snapshot(options),
    };
    if (complex === 'grpph' || complex === 'pph') return { pipeline: 'grpph', grounded: complex === 'grpph', ...graph };
    return { pipeline: 'network', kind: complex, maxDim: complex === 'grounded-flag' ? Math.min(maxDim, 1) : maxDim, ...graph };
  });
  // Moving a node changes nothing to compute
  const jobKey = $derived(job ? JSON.stringify([g.nodes.length, g.edges, options, complex, maxDim]) : '');
  $effect(() => {
    jobKey;
    const j = untrack(() => job);
    if (runner.auto) untrack(() => runner.schedule(j));
  });

  const dist = $derived(shortestPath && g.nodes.length <= 150 ? shortestPaths(g) : null);
  const domain = $derived(barDomain(runner.result?.barcode.bars ?? [], [0]));
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
  const label = (v: number) => g.nodes[v]?.label ?? String(v);
  const describe = (c: number[]) => {
    if (c.length === 1) return `node ${label(c[0])}`;
    if (complex === 'dowker-sink' || complex === 'dowker-source') return `{${c.map(label).join(', ')}}`;
    if (!pathHomology) return `(${c.map(label).join(', ')})`;
    if (c.length === 2) return `${label(c[0])}→${label(c[1])}`;
    if (c.length === 3 && c[0] === c[2]) return `double edge ${label(c[0])}⇄${label(c[1])}`;
    if (c.length === 3) return `triangle ${c.map(label).join('→')}`;
    return `long square ${label(c[0])}→${label(c[1])}→${label(c[3])}, ${label(c[0])}→${label(c[2])}→${label(c[3])}`;
  };
</script>

<div class="layout">
  <div>
    <div class="panel">
      <GraphView {g} t={runner.t} {bar} {dist} tool={source === 'draw' ? tool : null} {newWeight} bind:selectedEdge onedit={(next) => (g = next)} />
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} {domain} label={shortestPath ? 'Path length' : 'Weight'} />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>{g.nodes.length} nodes, {g.edges.length} edges</span>
        {#if dist}<span><i style="background: #54a24b"></i>new edges: pairs joined by a path no longer than the slider</span>{/if}
        {#if bar}<span>coloured: the representative (dashed if not an edge of the graph) · red: what kills it</span>{/if}
      </div>
    </div>

    <div class="panel secondary">
      <h3>Weighted digraph</h3>
      <SourceTabs bind:source drawLabel="Build" />
      {#if source === 'examples'}
        <div class="chips">
          {#each EXAMPLES as ex}
            <button class:on={exampleId === ex.id} onclick={() => useExample(ex.id)}>{ex.name}</button>
          {/each}
        </div>
        <p class="note">{EXAMPLES.find((e) => e.id === exampleId)?.about}</p>
      {:else if source === 'random'}
        <label class="field"><span>Nodes</span><input type="range" min="3" max="80" bind:value={randomN} onchange={() => useRandom()} /><span>{randomN}</span></label>
        <label class="field"><span>Edge probability</span><input type="range" min="0.02" max="0.6" step="0.01" bind:value={randomP} onchange={() => useRandom()} /><span>{randomP}</span></label>
        <label class="field"><span>Weights</span>
          <input type="number" min="0" step="0.1" bind:value={minW} onchange={() => useRandom()} /> to
          <input type="number" min="0" step="0.1" bind:value={maxW} onchange={() => useRandom()} /></label>
        <div class="field"><span>Seed {seed}</span><button onclick={() => useRandom(true)}>New graph</button></div>
      {:else if source === 'upload'}
        <FileDrop accept=".csv,.txt,.tsv" onfile={upload}>
          An edge list, one directed edge per line: source, target and an optional weight (default 1). Nodes can be numbers
          or names.
          <pre>source,target,weight
a,b,1.5
b,c,1
c,a,2</pre>
        </FileDrop>
        {#if uploadError}<p class="error">{uploadError}</p>{/if}
      {:else}
        <div class="tools">
          <button class:on={tool === 'node'} onclick={() => (tool = 'node')}>Add nodes</button>
          <button class:on={tool === 'edge'} onclick={() => (tool = 'edge')}>Add edges</button>
          <button class:on={tool === 'select'} onclick={() => (tool = 'select')}>Move / select</button>
          <button class:on={tool === 'delete'} onclick={() => (tool = 'delete')}>Delete</button>
          <button onclick={() => (g = layout(g))}>Tidy</button>
          <button onclick={() => { g = { nodes: [], edges: [] }; selectedEdge = null; }}>Clear</button>
        </div>
        <label class="field"><span>New edge weight</span><input type="number" min="0" step="0.5" bind:value={newWeight} /></label>
        {#if selectedEdge !== null && g.edges[selectedEdge]}
          {@const e = g.edges[selectedEdge]}
          <label class="field"><span>Edge {g.nodes[e.s].label}→{g.nodes[e.t].label}</span>
            <input type="number" min="0" step="0.5" value={e.w} onchange={(ev) => setWeight(+ev.currentTarget.value)} /></label>
        {/if}
        <p class="note">Click to add a node; drag from one node to another to add an edge; click an edge to change its weight.</p>
      {/if}
    </div>
  </div>

  <div>
    <div class="panel">
      <Results result={runner.result} error={runner.error} running={runner.running} {domain} bind:selected={runner.selected} bind:t={runner.t} {describe} name={complex} />
    </div>

    <div class="panel secondary">
      <h3>Compute</h3>
      <label class="field"><span>Complex</span>
        <select value={complex} onchange={(e) => setComplex(e.currentTarget.value as Complex)}>
          {#each COMPLEXES as c}<option value={c.id}>{c.name}</option>{/each}
        </select>
      </label>
      {#if !pathHomology && complex !== 'grounded-flag'}
        <label class="field"><span>Up to dimension</span>
          <select bind:value={maxDim}>
            <option value={0}>H₀</option><option value={1}>H₁</option><option value={2}>H₂</option><option value={3}>H₃</option>
          </select>
        </label>
      {/if}
      <p class="note">{COMPLEXES.find((c) => c.id === complex)?.about}</p>
      <Options bind:options defaults={pathHomology ? GRPPH : NETWORK} />
      <div class="compute-row">
        <button class="primary" disabled={!job} onclick={() => runner.run(job)}>Compute</button>
        <label><input type="checkbox" bind:checked={runner.auto} /> recompute on every change</label>
      </div>
    </div>
  </div>
</div>
