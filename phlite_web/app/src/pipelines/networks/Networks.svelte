<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import ComputeRow from '../../components/ComputeRow.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import EditBar from '../../components/EditBar.svelte';
  import { History } from '../../lib/history.svelte';
  import GraphView, { type Tool } from './GraphView.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { GRPPH, NETWORK, COMPLEX, RIPS, defaultsFor } from '../../lib/options';
  import { barDomain } from '../../lib/scale';
  import { randomSeed } from '../../lib/random';
  import type { GraphKind, Job, NetworkKind } from '../../lib/types';
  import { EXAMPLES, randomDigraph, parseEdges, layout, type Digraph } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let g = $state.raw<Digraph>(EXAMPLES[0].make());
  /** The digraph's complexes, and (`u-`) those of the graph with its edges undirected. */
  type Complex = 'grpph' | NetworkKind | 'u-flag' | 'u-lower-star' | 'u-extended';
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
        'Persistent path homology (Chowdhury and Mémoli): the path homology of the digraph of edges present at t, in any dimension. Cells are regular paths, each entering once all its edges have. With the shortest-path filtration, H₀ and H₁ are GrPPH without the grounding.',
    },
    {
      id: 'flag',
      name: 'Directed flag',
      about:
        "The directed flag complex (as flagser): a simplex for each directed clique, a sequence of distinct nodes with an edge from each to every later one, entering with its last edge (at the edge's weight).",
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
    {
      id: 'u-flag',
      name: 'Flag (undirected)',
      about:
        "The flag (clique) complex of the graph with its edges undirected (an edge given both ways keeps its smaller weight): a simplex for every clique, entering with its heaviest edge, or, by shortest paths, Rips of the graph's metric. phlite.networks.flag.",
    },
    {
      id: 'u-lower-star',
      name: 'Lower-star by height (undirected)',
      about:
        "The flag complex of the undirected graph filtered by the nodes' heights on the canvas (0 at the bottom, 10 at the top): each simplex enters with its highest node. Drag nodes up and down to change it. phlite.networks.flag(G, weight=None, node_weight=...).",
    },
    {
      id: 'u-extended',
      name: 'Extended persistence by height',
      about:
        "Extended persistence of the nodes' heights on the undirected graph: up through the sublevel sets, then down, so every class dies. Ext⁺ bars run from each component's lowest node to its highest; Ext⁻ bars from each loop's highest node to its lowest. phlite.networks.extended.",
    },
  ];
  let complex = $state<Complex>('grpph');
  let maxDim = $state(1);
  /** Each pair at the length of the shortest path between them, rather than each edge at its weight. */
  let pathFiltration = $state(false);
  /** Whether the complex has a choice of filtration (GrPPH and the grounded flag complex are by shortest paths). */
  const undirected = $derived(complex.startsWith('u-'));
  /** Whether the filtration is by the nodes' heights on the canvas (so moving a node changes it). */
  const byHeight = $derived(complex === 'u-lower-star' || complex === 'u-extended');
  const choosesFiltration = $derived(complex !== 'grpph' && complex !== 'grounded-flag' && !byHeight);
  const pathHomology = $derived(complex === 'grpph' || complex === 'pph');
  const shortestPath = $derived(complex === 'grpph' || complex === 'grounded-flag' || (choosesFiltration && pathFiltration));
  /** PPH of the shortest-path filtration in H₀ and H₁ is GrPPH without the grounding (as Python's `pph`). */
  const viaGrpph = $derived(complex === 'grpph' || (complex === 'pph' && pathFiltration && maxDim <= 1));
  /** The undirected flag complex is listed explicitly (as phlite.complexes), but by shortest paths by Rips' machinery. */
  const defaultsOf = (c: Complex, paths: boolean) =>
    c === 'grpph' ? GRPPH : c === 'u-flag' && paths ? RIPS : c.startsWith('u-') ? COMPLEX : NETWORK;
  const defaults = $derived(viaGrpph ? GRPPH : defaultsOf(complex, pathFiltration));
  let options = $state(defaultsFor(GRPPH));
  const runner = new Runner();
  function setComplex(c: Complex) {
    complex = c;
    options = defaultsFor(defaultsOf(c, pathFiltration));
  }
  $effect(() => {
    // Keep the options to what the pipeline that runs supports
    const d = defaults;
    untrack(() => {
      if (!d.parallelSetup && options.parallelSetup) options.parallelSetup = false;
      if ((!d.pivot && options.columns === 'pivot') || (d.unionFind === undefined && options.unionFind)) options = defaultsFor(d);
    });
  });
  /** Each node's height on the canvas: 0 at the bottom, 10 at the top. */
  const heights = $derived(g.nodes.map((v) => Math.round(10000 * (1 - v.y)) / 1000));

  // ---- The digraph, with undo ----
  const history = new History<Digraph>();
  function replace(next: Digraph) {
    history.commit(g);
    g = next;
    selectedEdge = null;
  }
  function useExample(id: string) {
    exampleId = id;
    const ex = EXAMPLES.find((e) => e.id === id)!;
    replace(ex.make());
    if (ex.show) {
      setComplex(ex.show.complex);
      pathFiltration = ex.show.shortestPath;
      maxDim = ex.show.maxDim;
    }
  }

  let randomN = $state(12);
  let randomP = $state(0.2);
  let minW = $state(0.5);
  let maxW = $state(3);
  let seed = $state(1);
  function useRandom(newSeed = false) {
    if (newSeed) seed = randomSeed();
    replace(randomDigraph(randomN, randomP, Math.min(minW, maxW), Math.max(minW, maxW), seed));
  }

  let uploadError = $state<string | null>(null);
  async function upload(file: File) {
    try {
      replace(parseEdges(await file.text()));
      uploadError = null;
    } catch (e) {
      uploadError = (e as Error).message;
    }
  }

  let tool = $state<Tool | null>(null);
  let newWeight = $state(1);
  let selectedEdge = $state<number | null>(null);
  const TOOLS: [Tool, string, string][] = [
    ['select', 'Move / select', 'Drag nodes about; click an edge to change its weight'],
    ['node', 'Add nodes', 'Click to add a node'],
    ['edge', 'Add edges', 'Drag from one node to another to add an edge'],
    ['delete', 'Delete', 'Click a node or an edge to delete it'],
  ];
  function setWeight(w: number) {
    if (selectedEdge === null || !(w >= 0)) return;
    history.commit(g);
    g = { ...g, edges: g.edges.map((e, i) => (i === selectedEdge ? { ...e, w } : e)) };
  }
  function deleteSelected() {
    if (selectedEdge === null) return;
    history.commit(g);
    g = { ...g, edges: g.edges.filter((_, i) => i !== selectedEdge) };
    selectedEdge = null;
  }
  function undo() {
    const previous = history.undo(g);
    if (previous) [g, selectedEdge] = [previous, null];
  }
  function redo() {
    const next = history.redo(g);
    if (next) [g, selectedEdge] = [next, null];
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
    if (viaGrpph) return { pipeline: 'grpph', grounded: complex === 'grpph', ...graph };
    if (undirected) {
      const kind: GraphKind =
        complex === 'u-flag' ? (pathFiltration ? 'flag-sp' : 'flag') : complex === 'u-lower-star' ? 'lower-star' : 'extended';
      return { pipeline: 'graph', kind, heights: Float64Array.from(heights), maxDim: kind === 'extended' ? 1 : maxDim, ...graph };
    }
    return {
      pipeline: 'network',
      kind: complex as NetworkKind,
      shortestPath: choosesFiltration && pathFiltration,
      maxDim: complex === 'grounded-flag' ? Math.min(maxDim, 1) : maxDim,
      ...graph,
    };
  });
  // Moving a node changes nothing to compute, but for the filtrations by height
  const jobKey = $derived(
    job ? JSON.stringify([g.nodes.length, g.edges, options, complex, maxDim, pathFiltration, byHeight ? heights : null]) : '',
  );
  $effect(() => {
    jobKey;
    const j = untrack(() => job);
    if (runner.auto) untrack(() => runner.schedule(j));
  });

  // When each pair a → b enters the filtration (phlite's, sent with the barcode), while the result is for this digraph
  const dist = $derived.by(() => {
    const job = runner.resultJob;
    const edges = runner.result?.barcode.edges;
    const n = g.nodes.length;
    if (!edges || !job || !('n' in job) || job.n !== n || n > 150) return null;
    const d = new Float64Array(n * n).fill(Infinity);
    for (let k = 0; k < edges.values.length; k++) d[edges.pairs[2 * k] * n + edges.pairs[2 * k + 1]] = edges.values[k];
    return d;
  });
  const domain = $derived(barDomain(runner.result?.barcode.bars ?? [], [0]));
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
  const label = (v: number) => g.nodes[v]?.label ?? String(v);
  const describe = (c: number[]) => {
    if (c.length === 1) return `node ${label(c[0])}`;
    if (undirected || complex === 'dowker-sink' || complex === 'dowker-source') return `{${c.map(label).join(', ')}}`;
    if (!pathHomology) return `(${c.map(label).join(', ')})`;
    if (complex === 'pph') return `path ${c.map(label).join('→')}`;
    if (c.length === 2) return `${label(c[0])}→${label(c[1])}`;
    if (c.length === 3 && c[0] === c[2]) return `double edge ${label(c[0])}⇄${label(c[1])}`;
    if (c.length === 3) return `triangle ${c.map(label).join('→')}`;
    return `long square ${label(c[0])}→${label(c[1])}→${label(c[3])}, ${label(c[0])}→${label(c[2])}→${label(c[3])}`;
  };
</script>

<div class="layout">
  <div>
    <div class="panel">
      <GraphView
        {g}
        t={runner.t}
        {bar}
        {dist}
        {tool}
        {newWeight}
        {undirected}
        heightLine={byHeight ? runner.t : null}
        bind:selectedEdge
        onstart={() => history.begin(g)}
        onedit={(next) => {
          history.edited();
          g = next;
        }}
      />
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} {domain} label={byHeight ? 'Height' : shortestPath ? 'Path length' : 'Weight'} />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>{g.nodes.length} nodes, {g.edges.length} edges</span>
        {#if dist && shortestPath}<span><i style="background: var(--green)"></i>new edges: pairs joined by a path no longer than the slider</span>{/if}
        {#if bar}<span>coloured: the representative (dashed if not an edge of the graph) · red: what kills it</span>{/if}
      </div>
      <EditBar
        bind:tool
        tools={TOOLS}
        canUndo={history.past.length > 0}
        canRedo={history.future.length > 0}
        onundo={undo}
        onredo={redo}
        ondelete={deleteSelected}
      >
        {#if tool === 'edge' || tool === 'node'}
          <label class="field"><span>New edge weight</span><input type="number" min="0" step="0.5" bind:value={newWeight} /></label>
        {/if}
        {#if selectedEdge !== null && g.edges[selectedEdge]}
          {@const e = g.edges[selectedEdge]}
          <label class="field"><span>Edge {g.nodes[e.s].label}→{g.nodes[e.t].label}</span>
            <input type="number" min="0" step="0.5" value={e.w} onchange={(ev) => setWeight(+ev.currentTarget.value)} /></label>
          <button onclick={deleteSelected}>Delete edge</button>
        {/if}
        {#if tool}
          <button onclick={() => replace(layout(g))}>Tidy</button>
          <button disabled={g.nodes.length === 0} onclick={() => replace({ nodes: [], edges: [] })}>Clear</button>
        {/if}
      </EditBar>
    </div>

    <div class="panel secondary">
      <h3>Weighted digraph</h3>
      <SourceTabs bind:source drawLabel="New" />
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
          <button
            onclick={() => {
              replace({ nodes: [], edges: [] });
              tool = 'node';
            }}>Empty graph</button
          >
        </div>
        <p class="note">
          Start from nothing and build a digraph with the editing tools under the view: click to add nodes, drag between
          them to add edges, click an edge to change its weight. Any digraph (an example, a random one or your own) can
          be edited the same way.
        </p>
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
      {#if choosesFiltration}
        <label class="field"><span>Filtration</span>
          <select bind:value={pathFiltration}>
            <option value={false}>edge weights</option>
            <option value={true}>shortest paths</option>
          </select>
        </label>
      {/if}
      {#if complex !== 'grpph' && complex !== 'grounded-flag' && complex !== 'u-extended'}
        <label class="field"><span>Up to dimension</span>
          <select bind:value={maxDim}>
            <option value={0}>H₀</option><option value={1}>H₁</option><option value={2}>H₂</option><option value={3}>H₃</option>
          </select>
        </label>
      {/if}
      <p class="note">{COMPLEXES.find((c) => c.id === complex)?.about}</p>
      <Options bind:options {defaults} />
      <ComputeRow {runner} {job} />
    </div>
  </div>
</div>
