<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import MagnitudeTable from './MagnitudeTable.svelte';
  import MpssView from './MpssView.svelte';
  import ComputeRow from '../../components/ComputeRow.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import EditBar from '../../components/EditBar.svelte';
  import Segmented from '../../components/Segmented.svelte';
  import More from '../../components/More.svelte';
  import { JOINS } from '../../lib/joins';
  import { History } from '../../lib/history.svelte';
  import GraphView, { type Tool } from './GraphView.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { GRPPH, NETWORK, COMPLEX, RIPS, defaultsFor } from '../../lib/options';
  import { barDomain } from '../../lib/scale';
  import { randomSeed } from '../../lib/random';
  import type { Bar, GraphKind, Job, NetworkKind } from '../../lib/types';
  import { EXAMPLES, randomDigraph, parseEdges, layout, type Digraph } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let g = $state.raw<Digraph>(EXAMPLES[0].make());
  /** The digraph's complexes, and (`u-`) those of the graph with its edges undirected. */
  type Complex = 'grpph' | NetworkKind | 'u-flag' | 'magnitude' | 'mpss' | 'zz-clique' | 'zz-flag';
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
      id: 'magnitude',
      name: 'Magnitude homology',
      about:
        'Magnitude homology (Hepworth and Willerton): tuples of nodes, consecutive ones distinct, graded by the total length of shortest paths through them, every edge of length 1. A table of ranks, not a barcode; over the integers it finds torsion (rare: the smallest known graphs with any have 15 nodes).',
    },
    {
      id: 'mpss',
      name: 'Magnitude-path spectral sequence',
      about:
        'The spectral sequence of the reachability complex filtered by length (Asao; Hepworth and Roff): E¹ is magnitude homology, E² bigraded path homology, converging to reachability homology. Every entry as a group, over a field or the integers, with generators: chains of tuples of nodes. Every edge has length 1; best on small digraphs.',
    },
    {
      id: 'u-flag',
      name: 'Flag (undirected)',
      about:
        "The flag (clique) complex of the graph with its edges undirected (an edge given both ways keeps its smaller weight): a simplex for every clique, entering with its heaviest edge, or, by shortest paths, Rips of the graph's metric. phlite.networks.flag.",
    },
    {
      id: 'zz-clique',
      name: 'Zigzag: clique complexes over time',
      about:
        "A temporal network: each edge's weight is when it is active. Each snapshot is the edges active in a window of time (and the nodes they touch), as its clique complex, joined to the next snapshot by their union or intersection: a bar is a component or loop followed from window to window. Computed by FastZigzag (Dey and Hou), as phlite.networks.network_zigzag.",
    },
    {
      id: 'zz-flag',
      name: 'Zigzag: directed flag complexes over time',
      about:
        "As the clique zigzag, each snapshot as its directed flag complex (a simplex for each directed clique), so a cyclic triangle a→b→c→a is a loop rather than filled in. phlite.networks.network_zigzag(complex='directed_flag').",
    },
  ];
  let complex = $state<Complex>('grpph');
  /** The complex's explanation: its first sentence, and the rest folded away. */
  const about = $derived(COMPLEXES.find((c) => c.id === complex)?.about ?? '');
  const aboutCut = $derived(about.search(/\.(\s|$)/) + 1);
  const aboutLead = $derived(aboutCut > 0 ? about.slice(0, aboutCut) : about);
  const aboutRest = $derived(aboutCut > 0 ? about.slice(aboutCut).trim() : '');
  type Kind = 'persistence' | 'zigzag' | 'magnitude';
  const kindOf = (c: Complex): Kind =>
    c === 'zz-clique' || c === 'zz-flag' ? 'zigzag' : c === 'magnitude' || c === 'mpss' ? 'magnitude' : 'persistence';
  /** The last complex chosen of each kind, to go back to. */
  const lastOf: Record<Kind, Complex> = $state({ persistence: 'grpph', zigzag: 'zz-clique', magnitude: 'magnitude' });
  /** A zigzag over time: snapshots of the edges active in windows of `zzWidth`, starting every `zzStep`. */
  const zigzag = $derived(complex === 'zz-clique' || complex === 'zz-flag');
  let zzWidth = $state(4);
  let zzStep = $state(1);
  let zzIntersection = $state(false);
  const zzStart = $derived(g.edges.length ? Math.min(...g.edges.map((e) => e.w)) : 0);
  const zzEnd = $derived(g.edges.length ? Math.max(...g.edges.map((e) => e.w)) : 0);
  const zzCount = $derived(zzStep > 0 ? Math.floor((zzEnd - zzStart) / zzStep) + 1 : 1);
  let maxDim = $state(1);
  /** Each pair at the length of the shortest path between them, rather than each edge at its weight. */
  let pathFiltration = $state(false);
  /** Whether the complex has a choice of filtration (GrPPH and the grounded flag complex are by shortest paths). */
  const undirected = $derived(complex.startsWith('u-'));
  const choosesFiltration = $derived(complex !== 'grpph' && complex !== 'grounded-flag' && complex !== 'magnitude' && complex !== 'mpss' && complex !== 'zz-clique' && complex !== 'zz-flag');
  let mpssModulus = $state(0);
  let mpssTuple = $state<number[] | null>(null);
  /** A tuple of the MPSS to show on the graph, as a representative: its nodes and the steps between them. */
  const mpssBar = $derived.by((): Bar | null => {
    const t = mpssTuple;
    if (complex !== 'mpss' || !t) return null;
    // Each step along a shortest directed path (every edge of length 1), or straight if there is none
    const out = new Map<number, number[]>();
    for (const e of g.edges) out.set(e.s, [...(out.get(e.s) ?? []), e.t]);
    const geodesic = (a: number, b: number): number[] => {
      const previous = new Map<number, number>([[a, a]]);
      const queue = [a];
      while (queue.length && !previous.has(b)) {
        const u = queue.shift()!;
        for (const v of out.get(u) ?? []) if (!previous.has(v)) (previous.set(v, u), queue.push(v));
      }
      if (!previous.has(b)) return [a, b];
      const path = [b];
      while (path[0] !== a) path.unshift(previous.get(path[0])!);
      return path;
    };
    const rep: [number, number[]][] =
      t.length === 1
        ? [[1, [t[0]]]]
        : t.slice(1).flatMap((v, i) => {
            const path = geodesic(t[i], v);
            return path.slice(1).map((w, j): [number, number[]] => [1, [path[j], w]]);
          });
    return { dim: t.length - 1, birth: 0, death: null, birth_cell: [], death_cell: null, rep };
  });
  let lMax = $state(4);
  let integral = $state(false);
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
    lastOf[kindOf(c)] = c;
    if (c === 'zz-clique' || c === 'zz-flag') runner.t = 0;
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
      if (ex.show.lMax !== undefined) lMax = ex.show.lMax;
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
    if (complex === 'magnitude') return { pipeline: 'magnitude', n: graph.n, sources: graph.sources, targets: graph.targets, lMax, integral };
    if (zigzag)
      return {
        pipeline: 'network-zigzag', n: graph.n, sources: graph.sources, targets: graph.targets, weights: graph.weights,
        directed: complex === 'zz-flag', width: zzWidth, step: zzStep, maxDim: Math.min(maxDim, 2), intersection: zzIntersection,
      };
    if (complex === 'mpss') return { pipeline: 'mpss', n: graph.n, sources: graph.sources, targets: graph.targets, lMax, modulus: mpssModulus, rMax: 8 };
    if (viaGrpph) return { pipeline: 'grpph', grounded: complex === 'grpph', ...graph };
    if (undirected) {
      const kind: GraphKind = pathFiltration ? 'flag-sp' : 'flag';
      return { pipeline: 'graph', kind, maxDim, ...graph };
    }
    return {
      pipeline: 'network',
      kind: complex as NetworkKind,
      shortestPath: choosesFiltration && pathFiltration,
      maxDim: complex === 'grounded-flag' ? Math.min(maxDim, 1) : maxDim,
      ...graph,
    };
  });
  // Moving a node changes nothing to compute
  const jobKey = $derived(job ? JSON.stringify([g.nodes.length, g.edges, options, complex, maxDim, pathFiltration, lMax, integral, mpssModulus, zzWidth, zzStep, zzIntersection]) : '');
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
  const domain = $derived<[number, number]>(zigzag ? [0, Math.max(1, zzCount)] : barDomain(runner.result?.barcode.bars ?? [], [0]));
  /** In a zigzag, the edges outside the snapshot at the slider. */
  const inactive = $derived.by(() => {
    if (!zigzag || runner.t === null) return null;
    const k = Math.min(zzCount - 1, Math.max(0, Math.floor(runner.t)));
    const lo = zzStart + k * zzStep;
    return new Set(g.edges.flatMap((e, i) => (lo <= e.w && e.w < lo + zzWidth ? [] : [i])));
  });
  const bar = $derived(mpssBar ?? (runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null)));
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
        undirected={undirected || complex === 'zz-clique'}
        {inactive}
        bind:selectedEdge
        onstart={() => history.begin(g)}
        onedit={(next) => {
          history.edited();
          g = next;
        }}
      />
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} {domain} label={zigzag ? 'Snapshot' : shortestPath ? 'Path length' : 'Weight'} />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>{g.nodes.length} nodes, {g.edges.length} edges</span>
        {#if zigzag && inactive}{@const k = Math.min(zzCount - 1, Math.max(0, Math.floor(runner.t ?? 0)))}<span>snapshot {k + 1} of {zzCount}: edges active in [{+(zzStart + k * zzStep).toFixed(3)}, {+(zzStart + k * zzStep + zzWidth).toFixed(3)}) drawn solid</span>{/if}
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
      {#if complex === 'mpss'}
        <MpssView result={runner.result} error={runner.error} running={runner.running} onhighlight={(t) => (mpssTuple = t)} />
      {:else if complex === 'magnitude'}
        <MagnitudeTable result={runner.result} error={runner.error} running={runner.running} />
      {:else}
        <Results result={runner.result} error={runner.error} running={runner.running} {domain} bind:selected={runner.selected} bind:t={runner.t} {describe} name={complex} />
      {/if}
    </div>

    <div class="panel secondary">
      <h3>Compute</h3>
      <div class="field"><span>Compute</span>
        <Segmented
          label="What to compute"
          value={kindOf(complex)}
          options={[
            { id: 'persistence', label: 'Persistence', title: 'Persistent homology of the weighted digraph' },
            { id: 'zigzag', label: 'Zigzag over time', title: 'Edge weights as times: snapshots in sliding windows of time' },
            { id: 'magnitude', label: 'Magnitude', title: 'Magnitude homology and the magnitude-path spectral sequence' },
          ]}
          onchange={(k) => setComplex(lastOf[k])}
        />
      </div>
      {#if kindOf(complex) === 'persistence'}
        <label class="field"><span>Complex</span>
          <select value={complex} onchange={(e) => setComplex(e.currentTarget.value as Complex)}>
            <optgroup label="Path homology">
              {#each COMPLEXES.filter((c) => c.id === 'grpph' || c.id === 'pph') as c}<option value={c.id}>{c.name}</option>{/each}
            </optgroup>
            <optgroup label="Complexes of the digraph">
              {#each COMPLEXES.filter((c) => ['flag', 'grounded-flag', 'tuples', 'dowker-sink', 'dowker-source'].includes(c.id)) as c}<option value={c.id}>{c.name}</option>{/each}
            </optgroup>
            <optgroup label="Of the undirected graph">
              {#each COMPLEXES.filter((c) => c.id === 'u-flag') as c}<option value={c.id}>{c.name}</option>{/each}
            </optgroup>
          </select>
        </label>
      {:else if kindOf(complex) === 'zigzag'}
        <div class="field"><span>Complex</span>
          <Segmented
            label="Complex"
            value={complex}
            options={[
              { id: 'zz-clique', label: 'Clique' },
              { id: 'zz-flag', label: 'Directed flag' },
            ]}
            onchange={setComplex}
          />
        </div>
      {:else}
        <div class="field"><span>Table</span>
          <Segmented
            label="Magnitude"
            value={complex}
            options={[
              { id: 'magnitude', label: 'Magnitude homology' },
              { id: 'mpss', label: 'Spectral sequence' },
            ]}
            onchange={setComplex}
          />
        </div>
      {/if}
      {#if choosesFiltration}
        <label class="field"><span>Filtration</span>
          <select bind:value={pathFiltration}>
            <option value={false}>edge weights</option>
            <option value={true}>shortest paths</option>
          </select>
        </label>
      {/if}
      {#if complex === 'mpss'}
        <label class="field"><span>Tuples up to length</span><input type="range" min="1" max="8" bind:value={lMax} /><span>{lMax}</span></label>
        <label class="field"><span>Coefficients</span>
          <select bind:value={mpssModulus}>
            <option value={0}>ℤ (the integers: torsion)</option>
            {#each [2, 3, 5, 7] as p}<option value={p}>𝔽{p}</option>{/each}
          </select>
        </label>
      {:else if complex === 'magnitude'}
        <label class="field"><span>Up to length</span><input type="range" min="1" max="10" bind:value={lMax} /><span>{lMax}</span></label>
        <label class="field"><span>Over the integers</span><input type="checkbox" bind:checked={integral} /></label>
      {:else if zigzag}
        <label class="field"><span>Window of time</span><input type="number" min="0" step="0.5" bind:value={zzWidth} /></label>
        <label class="field"><span>Step</span><input type="number" min="0.01" step="0.5" bind:value={zzStep} /><span>{zzCount} snapshots</span></label>
        <div class="field"><span>Join</span>
          <Segmented label="Join" value={zzIntersection ? 'intersection' : 'union'} options={JOINS} onchange={(j) => (zzIntersection = j === 'intersection')} />
        </div>
        <label class="field"><span>Up to dimension</span>
          <select bind:value={maxDim}>
            <option value={0}>H₀</option><option value={1}>H₁</option><option value={2}>H₂</option>
          </select>
        </label>
      {:else if complex !== 'grpph' && complex !== 'grounded-flag'}
        <label class="field"><span>Up to dimension</span>
          <select bind:value={maxDim}>
            <option value={0}>H₀</option><option value={1}>H₁</option><option value={2}>H₂</option><option value={3}>H₃</option>
          </select>
        </label>
      {/if}
      <More>
        {#snippet lead()}{aboutLead}{/snippet}
        {#if aboutRest}{aboutRest}{/if}
      </More>
      {#if complex !== 'magnitude' && complex !== 'mpss' && !zigzag}<Options bind:options {defaults} />{/if}
      <ComputeRow {runner} {job} />
    </div>
  </div>
</div>
