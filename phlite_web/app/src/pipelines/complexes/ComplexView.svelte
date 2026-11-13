<script lang="ts">
  import { dimColour, type Bar } from '../../lib/types';
  import { vertexSets, type Complex } from './data';

  export type Tool = 'vertex' | 'edge' | 'triangle' | 'move' | 'delete';

  let {
    c,
    values,
    t,
    bar,
    tool = null,
    onedit,
  }: {
    c: Complex;
    values: number[];
    t: number | null;
    bar: Bar | null;
    tool?: Tool | null;
    onedit?: (c: Complex) => void;
  } = $props();

  let width = $state(500);
  const height = $derived(Math.round(width * 0.8));
  const verts = $derived(vertexSets(c));

  // Fit the vertices (or the unit square while building an empty complex)
  const box = $derived.by(() => {
    const ps = c.filter((x) => x.pos).map((x) => x.pos!);
    if (ps.length < 2 || tool) return tool && ps.length < 2 ? [0, 0, 1, 1] : fit(ps.length ? ps : [[0, 0], [1, 1]]);
    return fit(ps);
  });
  function fit(ps: [number, number][] | number[][]) {
    const xs = ps.map((p) => p[0]);
    const ys = ps.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const pad = Math.max(x1 - x0, y1 - y0, 0.2) * 0.12;
    return [x0 - pad, y0 - pad, x1 + pad, y1 + pad];
  }
  // While building, keep the box fixed so the picture doesn't jump about
  let frozen: number[] | null = $state(null);
  $effect(() => {
    if (tool && !frozen) frozen = box.slice();
    if (!tool) frozen = null;
  });
  const view = $derived(frozen ?? box);
  const scale = $derived(Math.min(width / (view[2] - view[0]), height / (view[3] - view[1])));
  const ox = $derived((width - scale * (view[2] - view[0])) / 2);
  const oy = $derived((height - scale * (view[3] - view[1])) / 2);
  const sx = (x: number) => ox + (x - view[0]) * scale;
  const sy = (y: number) => height - oy - (y - view[1]) * scale;
  const P = (v: number) => [sx(c[v].pos![0]), sy(c[v].pos![1])];
  const pts = (i: number) => verts[i].map(P).map((p) => p.join(',')).join(' ');

  const present = (i: number) => t === null || values[i] <= t;
  const repCells = $derived(new Map((bar?.rep ?? []).map(([coeff, cell]) => [cell[0], coeff])));
  const colour = $derived(bar ? dimColour(bar.dim) : '#000');
  const order = $derived(c.map((_, i) => i).sort((a, b) => c[b].dim - c[a].dim));
  const showLabels = $derived(c.filter((x) => x.dim === 0).length <= 40);

  // ---- Building ----
  let svg: SVGSVGElement;
  let picked = $state<number[]>([]);
  let dragging: number | null = null;
  $effect(() => {
    tool;
    picked = [];
  });
  function world(e: PointerEvent): [number, number] {
    const r = svg.getBoundingClientRect();
    return [view[0] + (e.clientX - r.left - ox) / scale, view[1] + (height - oy - (e.clientY - r.top)) / scale];
  }
  function vertexAt(p: [number, number]) {
    let best = -1;
    let bestD = 12 / scale;
    c.forEach((x, i) => {
      if (x.dim !== 0 || !x.pos) return;
      const d = Math.hypot(x.pos[0] - p[0], x.pos[1] - p[1]);
      if (d < bestD) [best, bestD] = [i, d];
    });
    return best;
  }
  function nextLabel(): string {
    const used = new Set(c.map((x) => x.label));
    for (let i = 0; ; i++) {
      const l = i < 26 ? 'abcdefghijklmnopqrstuvwxyz'[i] : `v${i}`;
      if (!used.has(l)) return l;
    }
  }
  /** The cell with exactly these vertices, adding it (and its missing faces) if needed. */
  function ensure(cells: Complex, vs: number[]): number {
    vs = [...vs].sort((a, b) => a - b);
    const sets = vertexSets(cells);
    const found = sets.findIndex((s, i) => cells[i].dim === vs.length - 1 && s.length === vs.length && s.every((v, k) => v === vs[k]));
    if (found >= 0) return found;
    const faces = vs.map((_, i) => ({ cell: ensure(cells, vs.filter((_, j) => j !== i)), coeff: i % 2 === 0 ? 1 : -1 }));
    const sep = vs.some((v) => cells[v].label.length > 1) ? '-' : '';
    cells.push({ label: vs.map((v) => cells[v].label).join(sep), dim: vs.length - 1, faces, value: null });
    return cells.length - 1;
  }
  /** Remove cell `i` and everything with it as a face. */
  function remove(cells: Complex, i: number): Complex {
    const gone = new Set([i]);
    for (let changed = true; changed; ) {
      changed = false;
      cells.forEach((x, j) => {
        if (!gone.has(j) && x.faces.some((f) => gone.has(f.cell))) {
          gone.add(j);
          changed = true;
        }
      });
    }
    const renumber: number[] = [];
    let next = 0;
    cells.forEach((_, j) => (renumber[j] = gone.has(j) ? -1 : next++));
    return cells.filter((_, j) => !gone.has(j)).map((x) => ({ ...x, faces: x.faces.map((f) => ({ ...f, cell: renumber[f.cell] })) }));
  }
  function inTriangle(p: number[], i: number) {
    const [a, b, d] = verts[i].map((v) => c[v].pos!);
    const s = (u: number[], v: number[], w: number[]) => (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
    const [s1, s2, s3] = [s(a, b, p), s(b, d, p), s(d, a, p)];
    return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
  }
  function nearEdge(p: number[], i: number) {
    const [a, b] = verts[i].map((v) => c[v].pos!);
    const [dx, dy] = [b[0] - a[0], b[1] - a[1]];
    const u = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
    return Math.hypot(a[0] + u * dx - p[0], a[1] + u * dy - p[1]) < 7 / scale;
  }
  function down(e: PointerEvent) {
    if (!tool || !onedit) return;
    // Otherwise a drag can start selecting (then dragging) the labels' text, which cancels the pointer
    e.preventDefault();
    const p = world(e);
    const v = vertexAt(p);
    if (tool === 'vertex') {
      if (v >= 0) {
        dragging = v;
        svg.setPointerCapture(e.pointerId);
      } else onedit([...c, { label: nextLabel(), dim: 0, faces: [], value: null, pos: p }]);
    } else if (tool === 'move' && v >= 0) {
      dragging = v;
      svg.setPointerCapture(e.pointerId);
    } else if ((tool === 'edge' || tool === 'triangle') && v >= 0) {
      const next = picked.includes(v) ? picked.filter((x) => x !== v) : [...picked, v];
      if (next.length === (tool === 'edge' ? 2 : 3)) {
        const cells = c.map((x) => ({ ...x }));
        ensure(cells, next);
        onedit(cells);
        picked = [];
      } else picked = next;
    } else if (tool === 'delete') {
      const target =
        v >= 0
          ? v
          : c.findIndex((x, i) => x.dim === 1 && nearEdge(p, i)) >= 0
            ? c.findIndex((x, i) => x.dim === 1 && nearEdge(p, i))
            : c.findIndex((x, i) => x.dim === 2 && inTriangle(p, i));
      if (target >= 0) onedit(remove(c, target));
    }
  }
  function move(e: PointerEvent) {
    if (dragging === null || !onedit) return;
    const cells = c.slice();
    cells[dragging] = { ...cells[dragging], pos: world(e) };
    onedit(cells);
  }
</script>

<div class="view" bind:clientWidth={width} style="height: {height}px">
  <svg bind:this={svg} {width} {height} role="application" onpointerdown={down} onpointermove={move} onpointerup={() => (dragging = null)}
    style="cursor: {tool === 'vertex' ? 'copy' : tool ? 'pointer' : 'default'}">
    {#each order as i (i)}
      {@const x = c[i]}
      {@const rep = repCells.has(i)}
      {#if x.dim === 2 && verts[i].length === 3}
        <polygon points={pts(i)} class="tri" class:absent={!present(i)} style={rep ? `fill: ${colour}; fill-opacity: 0.55` : ''} />
      {:else if x.dim === 1 && verts[i].length === 2}
        {@const [a, b] = verts[i].map(P)}
        <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} class="edge" class:absent={!present(i)} style={rep ? `stroke: ${colour}; stroke-width: 4.5` : ''} />
      {:else if x.dim === 0 && x.pos}
        {@const [px, py] = P(i)}
        <circle cx={px} cy={py} r={rep || picked.includes(i) ? 7 : 5} class="vertex" class:absent={!present(i)} class:picked={picked.includes(i)}
          style={rep ? `fill: ${colour}` : ''} />
        {#if showLabels}<text x={px + 8} y={py - 8} class="label">{x.label}</text>{/if}
      {/if}
    {/each}
    {#if bar?.death_cell && verts[bar.death_cell[0]]}
      {@const d = bar.death_cell[0]}
      {#if verts[d].length >= 2}
        <polygon points={pts(d)} class="death" />
      {/if}
    {/if}
  </svg>
</div>

<style>
  .tri { fill: #4c78a8; fill-opacity: 0.18; stroke: none; }
  .edge { stroke: var(--fg); stroke-width: 1.6; stroke-linecap: round; }
  .vertex { fill: var(--fg); }
  .vertex.picked { fill: var(--accent); }
  .absent { opacity: 0.12; }
  .label { font-size: 11px; fill: var(--muted); pointer-events: none; }
  .death { fill: none; stroke: #e45756; stroke-width: 2; stroke-dasharray: 4 3; pointer-events: none; }
</style>
