<script lang="ts">
  import { dimColour, type Bar } from '../../lib/types';
  import type { Digraph } from './data';

  export type Tool = 'select' | 'node' | 'edge' | 'delete';

  let {
    g,
    t,
    bar,
    dist,
    tool = null,
    newWeight = 1,
    selectedEdge = $bindable(null),
    onedit,
  }: {
    g: Digraph;
    t: number | null;
    bar: Bar | null;
    /** All-pairs shortest path lengths, for the filtration's new edges, or null if too big to draw. */
    dist: Float64Array | null;
    tool?: Tool | null;
    newWeight?: number;
    selectedEdge?: number | null;
    onedit?: (g: Digraph) => void;
  } = $props();

  let width = $state(500);
  const height = $derived(Math.round(width * 0.8));
  const R = $derived(g.nodes.length > 60 ? 5 : g.nodes.length > 25 ? 8 : 12);
  const X = (x: number) => x * width;
  const Y = (y: number) => y * height;

  const has = $derived(new Set(g.edges.map((e) => `${e.s},${e.t}`)));
  /** An edge from a to b as a path: curved if the reverse edge is drawn too, stopping short of the target node. */
  function path(a: number, b: number, curved: boolean) {
    const [x1, y1, x2, y2] = [X(g.nodes[a].x), Y(g.nodes[a].y), X(g.nodes[b].x), Y(g.nodes[b].y)];
    const len = Math.hypot(x2 - x1, y2 - y1) || 1;
    const [ux, uy] = [(x2 - x1) / len, (y2 - y1) / len];
    const bend = curved ? Math.min(0.18 * len, 30) : 0;
    const [mx, my] = [(x1 + x2) / 2 - uy * bend, (y1 + y2) / 2 + ux * bend];
    // End where the curve meets the node's circle (approximately)
    const [ex, ey] = [x2 - ux * (R + 3) - uy * (curved ? bend * 0.25 : 0), y2 - uy * (R + 3) + ux * (curved ? bend * 0.25 : 0)];
    return { d: `M${x1},${y1} Q${mx},${my} ${ex},${ey}`, mx: (x1 + 2 * mx + ex) / 4, my: (y1 + 2 * my + ey) / 4 };
  }

  // The filtration's new edges at t: pairs joined by a path of length at most t
  const newEdges = $derived.by(() => {
    if (t === null || !dist) return [];
    const n = g.nodes.length;
    const out: [number, number][] = [];
    for (let a = 0; a < n; a++)
      for (let b = 0; b < n; b++) if (a !== b && !has.has(`${a},${b}`) && dist[a * n + b] <= t) out.push([a, b]);
    return out.slice(0, 3000);
  });

  const repEdges = $derived(bar?.rep ? bar.rep.filter(([, c]) => c.length === 2).map(([, c]) => c) : []);
  const repNodes = $derived(new Set(bar?.rep ? bar.rep.filter(([, c]) => c.length === 1).map(([, c]) => c[0]) : []));
  const colour = $derived(bar ? dimColour(bar.dim) : '#000');

  // ---- Editing ----
  let svg: SVGSVGElement;
  let dragging: { kind: 'move' | 'edge'; node: number } | null = $state(null);
  let pointer = $state<[number, number]>([0, 0]);
  function local(e: PointerEvent): [number, number] {
    const r = svg.getBoundingClientRect();
    return [(e.clientX - r.left) / width, (e.clientY - r.top) / height];
  }
  function nodeAt([x, y]: [number, number]) {
    let best = -1;
    let bestD = (R + 4) / width;
    g.nodes.forEach((v, i) => {
      const d = Math.hypot(v.x - x, (v.y - y) * (height / width));
      if (d < bestD) [best, bestD] = [i, d];
    });
    return best;
  }
  function down(e: PointerEvent) {
    if (!tool || !onedit) return;
    // Otherwise a drag can start selecting (then dragging) the labels' text, which cancels the pointer
    e.preventDefault();
    const p = local(e);
    pointer = p;
    const v = nodeAt(p);
    if (tool === 'node' && v < 0) {
      onedit({ ...g, nodes: [...g.nodes, { x: p[0], y: p[1], label: nextLabel() }] });
    } else if (tool === 'edge' && v >= 0) {
      dragging = { kind: 'edge', node: v };
      svg.setPointerCapture(e.pointerId);
    } else if ((tool === 'select' || tool === 'node') && v >= 0) {
      dragging = { kind: 'move', node: v };
      svg.setPointerCapture(e.pointerId);
    } else if (tool === 'delete' && v >= 0) {
      onedit({
        nodes: g.nodes.filter((_, i) => i !== v),
        edges: g.edges.filter((ed) => ed.s !== v && ed.t !== v).map((ed) => ({ ...ed, s: ed.s - +(ed.s > v), t: ed.t - +(ed.t > v) })),
      });
      selectedEdge = null;
    } else if (v < 0 && tool === 'select') selectedEdge = null;
  }
  function move(e: PointerEvent) {
    pointer = local(e);
    if (dragging?.kind === 'move' && onedit) {
      const nodes = g.nodes.slice();
      nodes[dragging.node] = { ...nodes[dragging.node], x: Math.min(0.98, Math.max(0.02, pointer[0])), y: Math.min(0.98, Math.max(0.02, pointer[1])) };
      onedit({ ...g, nodes });
    }
  }
  function up() {
    if (dragging?.kind === 'edge' && onedit) {
      const v = nodeAt(pointer);
      const s = dragging.node;
      if (v >= 0 && v !== s) {
        const edges = g.edges.filter((ed) => !(ed.s === s && ed.t === v));
        edges.push({ s, t: v, w: newWeight });
        onedit({ ...g, edges });
        selectedEdge = edges.length - 1;
      }
    }
    dragging = null;
  }
  function clickEdge(i: number, e: MouseEvent) {
    if (tool === 'delete' && onedit) {
      e.stopPropagation();
      onedit({ ...g, edges: g.edges.filter((_, j) => j !== i) });
      selectedEdge = null;
    } else if (tool) {
      e.stopPropagation();
      selectedEdge = i;
    }
  }
  function nextLabel() {
    const used = new Set(g.nodes.map((v) => v.label));
    let i = g.nodes.length;
    while (used.has(String(i))) i++;
    return String(i);
  }
</script>

<div class="view" bind:clientWidth={width} style="height: {height}px">
  <svg
    bind:this={svg}
    {width}
    {height}
    role="application"
    style="cursor: {tool === 'node' ? 'copy' : tool === 'edge' ? 'crosshair' : 'default'}"
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
  >
    <defs>
      {#each [['grey', 'var(--muted)'], ['new', '#54a24b'], ['rep', colour], ['sel', 'var(--accent)']] as [id, fill]}
        <marker id="arrow-{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style="fill: {fill}" />
        </marker>
      {/each}
    </defs>

    {#each newEdges as [a, b]}
      <path d={path(a, b, newEdges.length < 200).d} class="new" marker-end="url(#arrow-new)" />
    {/each}

    {#each g.edges as e, i}
      {@const p = path(e.s, e.t, has.has(`${e.t},${e.s}`))}
      <g role="presentation" onpointerdown={(ev) => tool && tool !== 'node' && tool !== 'edge' && ev.stopPropagation()} onclick={(ev) => clickEdge(i, ev)}>
        <path d={p.d} class="hit" />
        <path d={p.d} class="edge" class:sel={selectedEdge === i} marker-end="url(#arrow-{selectedEdge === i ? 'sel' : 'grey'})" />
        {#if g.edges.length <= 80}
          <text x={p.mx} y={p.my} class="w">{e.w}</text>
        {/if}
      </g>
    {/each}

    {#each repEdges as [a, b]}
      {#if g.nodes[a] && g.nodes[b]}
        <path d={path(a, b, has.has(`${b},${a}`)).d} class="rep" class:shortcut={!has.has(`${a},${b}`)} style="stroke: {colour}" marker-end="url(#arrow-rep)" />
      {/if}
    {/each}
    {#if bar?.death_cell && bar.death_cell.length >= 2}
      <polyline
        points={bar.death_cell.map((v) => `${X(g.nodes[v]?.x ?? 0)},${Y(g.nodes[v]?.y ?? 0)}`).join(' ')}
        class="death"
      />
    {/if}

    {#if dragging?.kind === 'edge'}
      <line x1={X(g.nodes[dragging.node].x)} y1={Y(g.nodes[dragging.node].y)} x2={X(pointer[0])} y2={Y(pointer[1])} class="drag" />
    {/if}

    {#each g.nodes as v, i}
      <g transform="translate({X(v.x)},{Y(v.y)})">
        <circle r={R} class="node" class:rep={repNodes.has(i)} style={repNodes.has(i) ? `fill: ${colour}` : ''} />
        {#if R >= 8}<text class="label" dy="0.32em">{v.label}</text>{/if}
      </g>
    {/each}
  </svg>
</div>

<style>
  .edge { fill: none; stroke: var(--muted); stroke-width: 1.4; }
  .edge.sel { stroke: var(--accent); stroke-width: 2.5; }
  .hit { fill: none; stroke: transparent; stroke-width: 12; cursor: pointer; }
  .new { fill: none; stroke: #54a24b; stroke-width: 1; stroke-dasharray: 4 3; opacity: 0.7; }
  .rep { fill: none; stroke-width: 4; opacity: 0.9; pointer-events: none; }
  .rep.shortcut { stroke-dasharray: 6 4; }
  .death { fill: none; stroke: #e45756; stroke-width: 2; stroke-dasharray: 3 3; pointer-events: none; }
  .drag { stroke: var(--accent); stroke-width: 2; stroke-dasharray: 4 3; }
  .node { fill: var(--panel); stroke: var(--fg); stroke-width: 1.5; }
  .node.rep { stroke-width: 2.5; }
  .label { font-size: 9px; text-anchor: middle; fill: var(--fg); pointer-events: none; }
  .w { font-size: 10px; fill: var(--muted); text-anchor: middle; paint-order: stroke; stroke: var(--panel); stroke-width: 3px; pointer-events: none; }
</style>
