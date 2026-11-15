<script lang="ts">
  import { untrack } from 'svelte';
  import { dimColour, type Bar, type FiltrationEdges } from '../../lib/types';
  import { cssColour } from '../../lib/palette.svelte';
  import { activeComplex } from './data';

  /**
   * `select`: click a point to select it (shift to add to the selection), drag to move the selection, or drag from
   * empty space to select a box; `add`: click to add a point (and drag to place it); `spray`, `erase`: a round brush.
   */
  export type Tool = 'select' | 'add' | 'spray' | 'erase';

  let {
    points,
    t,
    filtration,
    bar,
    balls,
    edges,
    tool = null,
    brush = 0.08,
    selection = $bindable([]),
    fitKey = 0,
    onstart,
    onedit,
  }: {
    points: number[][];
    t: number | null;
    filtration: 'rips' | 'alpha';
    bar: Bar | null;
    balls: boolean;
    /** A flag filtration's edges (Rips, DTM-Rips), to draw its complex at t: edges and triangles. */
    edges: FiltrationEdges | null;
    tool?: Tool | null;
    /** The brush's radius, as a fraction of the view's width. */
    brush?: number;
    /** The selected points' indices. */
    selection?: number[];
    /** Changed when the cloud is replaced (not edited), to fit the view to it again. */
    fitKey?: unknown;
    /** A gesture that may edit the points has begun. */
    onstart?: () => void;
    onedit?: (points: number[][]) => void;
  } = $props();

  let width = $state(500);
  let canvas: HTMLCanvasElement;
  let hover: number[] | null = $state(null);
  const height = $derived(Math.round(width * 0.8));
  const EDGE_LIMIT = 30000;
  const TRIANGLE_LIMIT = 20000;
  // The complex at t, when the filtration is a flag filtration
  const complex = $derived(edges && t !== null ? activeComplex(edges, points.length, t, TRIANGLE_LIMIT) : null);

  function fit(pad: number) {
    if (points.length === 0) return [-1.25, -1, 1.25, 1];
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const [x, y] of points) {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    const p = Math.max(x1 - x0, y1 - y0, 1e-6) * pad;
    return [x0 - p, y0 - p, x1 + p, y1 + p];
  }
  // While editing, the view stays put (with room around the points to add more), so nothing moves under the pointer;
  // it fits the points again when the tool is put down or the cloud is replaced
  let frozen = $state<number[] | null>(null);
  $effect.pre(() => {
    const editing = tool !== null;
    void fitKey;
    untrack(() => (frozen = editing ? fit(0.2) : null));
  });
  const box = $derived(frozen ?? fit(0.08));
  // World to screen, keeping the aspect ratio
  const scale = $derived(Math.min(width / (box[2] - box[0]), height / (box[3] - box[1])));
  const ox = $derived((width - scale * (box[2] - box[0])) / 2);
  const oy = $derived((height - scale * (box[3] - box[1])) / 2);
  const sx = (x: number) => ox + (x - box[0]) * scale;
  const sy = (y: number) => height - oy - (y - box[1]) * scale;
  const wx = (px: number) => box[0] + (px - ox) / scale;
  const wy = (py: number) => box[1] + (height - oy - py) / scale;
  /** The brush's radius in world units. */
  const radius = $derived((brush * width) / scale);

  // ---- Editing ----
  type Drag =
    | { kind: 'move'; from: number[]; original: number[][]; moved: boolean }
    | { kind: 'box'; from: number[]; to: number[]; extend: boolean }
    | { kind: 'brush'; last: number[] | null };
  let drag = $state<Drag | null>(null);
  const selected = $derived(new Set(selection));

  function at(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  /** The point under the pointer (within a few pixels), or -1. */
  function pointAt(px: number[]) {
    let best = -1;
    let bestD = 9;
    points.forEach(([x, y], i) => {
      const d = Math.hypot(sx(x) - px[0], sy(y) - px[1]);
      if (d < bestD) [best, bestD] = [i, d];
    });
    return best;
  }
  const hovered = $derived(tool === 'select' && hover && !drag ? pointAt(hover) : -1);

  function down(e: PointerEvent) {
    if (!tool || !onedit || e.button !== 0) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const px = at(e);
    const w = [wx(px[0]), wy(px[1])];
    onstart?.();
    if (tool === 'select') {
      const i = pointAt(px);
      if (i < 0) {
        drag = { kind: 'box', from: px, to: px, extend: e.shiftKey };
        return;
      }
      if (e.shiftKey) {
        selection = selected.has(i) ? selection.filter((j) => j !== i) : [...selection, i];
        return;
      }
      if (!selected.has(i)) selection = [i];
      drag = { kind: 'move', from: w, original: points, moved: false };
    } else if (tool === 'add') {
      // Add a point, selected, and carry on dragging it into place
      // (Build the new cloud once: `points` is the parent's, so it is the new cloud as soon as `onedit` returns)
      const added = [...points, w];
      onedit(added);
      selection = [added.length - 1];
      drag = { kind: 'move', from: w, original: added, moved: true };
    } else {
      drag = { kind: 'brush', last: null };
      stroke(px);
    }
  }
  function move(e: PointerEvent) {
    hover = at(e);
    if (!drag || !onedit) return;
    if (drag.kind === 'box') drag.to = hover;
    else if (drag.kind === 'move') {
      const [dx, dy] = [wx(hover[0]) - drag.from[0], wy(hover[1]) - drag.from[1]];
      drag.moved = true;
      onedit(drag.original.map((p, i) => (selected.has(i) ? [p[0] + dx, p[1] + dy] : p)));
    } else stroke(hover);
  }
  function up() {
    if (drag?.kind === 'box') {
      const [a, b] = [drag.from, drag.to];
      const [x0, x1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
      const [y0, y1] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
      const inside: number[] = [];
      points.forEach(([x, y], i) => {
        if (sx(x) >= x0 && sx(x) <= x1 && sy(y) >= y0 && sy(y) <= y1) inside.push(i);
      });
      selection = drag.extend ? [...new Set([...selection, ...inside])] : inside;
    }
    drag = null;
  }
  /** Spray points about, or erase those under, the brush at `px`. */
  function stroke(px: number[]) {
    if (drag?.kind !== 'brush' || !onedit) return;
    const [x, y] = [wx(px[0]), wy(px[1])];
    if (tool === 'erase') {
      const kept = points.filter(([a, b]) => Math.hypot(a - x, b - y) > radius);
      if (kept.length !== points.length) {
        selection = [];
        onedit(kept);
      }
      return;
    }
    // Spray: a few points per brush-width moved
    if (drag.last && Math.hypot(px[0] - drag.last[0], px[1] - drag.last[1]) < brush * width * 0.5) return;
    drag.last = px;
    const added = Array.from({ length: 4 }, () => {
      const a = 2 * Math.PI * Math.random();
      const r = radius * Math.sqrt(Math.random());
      return [x + r * Math.cos(a), y + r * Math.sin(a)];
    });
    onedit([...points, ...added]);
  }

  // ---- Drawing ----
  $effect(() => {
    const dpr = devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const fg = cssColour('--fg');
    const muted = cssColour('--muted');
    const accent = cssColour('--accent');
    ctx.clearRect(0, 0, width, height);

    // The complex at t: balls of radius t/2 (Rips) or t (alpha), and the Rips edges
    if (t !== null && t > 0) {
      if (balls) {
        const r = (filtration === 'rips' ? t / 2 : t) * scale;
        ctx.fillStyle = 'rgba(76, 120, 168, 0.13)';
        ctx.beginPath();
        for (const [x, y] of points) {
          ctx.moveTo(sx(x) + r, sy(y));
          ctx.arc(sx(x), sy(y), r, 0, 2 * Math.PI);
        }
        ctx.fill();
      }
      if (complex && edges) {
        // Triangles, faintly (they overlap, so many look darker), then edges
        const tri = complex.triangles;
        ctx.fillStyle = cssColour('--edges');
        ctx.globalAlpha = tri.length / 3 > 2000 ? 0.08 : 0.2;
        ctx.beginPath();
        for (let k = 0; k < tri.length; k += 3) {
          const [a, b, c] = [points[tri[k]], points[tri[k + 1]], points[tri[k + 2]]];
          if (!a || !b || !c) continue;
          ctx.moveTo(sx(a[0]), sy(a[1]));
          ctx.lineTo(sx(b[0]), sy(b[1]));
          ctx.lineTo(sx(c[0]), sy(c[1]));
          ctx.closePath();
        }
        ctx.fill();
        const m = Math.min(complex.m, EDGE_LIMIT);
        ctx.strokeStyle = muted;
        ctx.globalAlpha = m > 3000 ? 0.25 : 0.55;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let k = 0; k < m; k++) {
          const a = points[edges.pairs[2 * k]];
          const b = points[edges.pairs[2 * k + 1]];
          if (!a || !b) continue;
          ctx.moveTo(sx(a[0]), sy(a[1]));
          ctx.lineTo(sx(b[0]), sy(b[1]));
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // The selected bar: its representative, and the simplex that kills it
    if (bar) {
      const colour = dimColour(bar.dim);
      const simplex = (cell: number[], fill: boolean) => {
        const ps = cell.map((v) => points[v]).filter(Boolean);
        if (ps.length === 0) return;
        ctx.beginPath();
        ctx.moveTo(sx(ps[0][0]), sy(ps[0][1]));
        for (const p of ps.slice(1)) ctx.lineTo(sx(p[0]), sy(p[1]));
        if (ps.length > 2) ctx.closePath();
        if (fill && ps.length > 2) ctx.fill();
        ctx.stroke();
      };
      if (bar.rep) {
        ctx.strokeStyle = colour;
        ctx.fillStyle = colour + '55';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        for (const [, cell] of bar.rep) {
          if (cell.length === 1) {
            const p = points[cell[0]];
            if (!p) continue;
            ctx.beginPath();
            ctx.arc(sx(p[0]), sy(p[1]), 6, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
          } else simplex(cell, true);
        }
      }
      if (bar.death_cell) {
        ctx.strokeStyle = fg;
        ctx.fillStyle = 'transparent';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        simplex(bar.death_cell, false);
        ctx.setLineDash([]);
      }
    }

    // The points, the selected ones ringed; with births (DTM-Rips), those not yet born at t hollow
    const pr = points.length > 1000 ? 1.6 : points.length > 300 ? 2.2 : 3;
    const births = edges?.births && t !== null ? edges.births : null;
    ctx.fillStyle = fg;
    ctx.beginPath();
    points.forEach(([x, y], i) => {
      if (births && births[i] > t!) return;
      ctx.moveTo(sx(x) + pr, sy(y));
      ctx.arc(sx(x), sy(y), pr, 0, 2 * Math.PI);
    });
    ctx.fill();
    if (births) {
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      points.forEach(([x, y], i) => {
        if (births[i] <= t!) return;
        ctx.moveTo(sx(x) + pr, sy(y));
        ctx.arc(sx(x), sy(y), pr, 0, 2 * Math.PI);
      });
      ctx.stroke();
    }
    if (tool && (selection.length > 0 || hovered >= 0)) {
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const i of hovered >= 0 ? [...selection, hovered] : selection) {
        const p = points[i];
        if (!p) continue;
        ctx.moveTo(sx(p[0]) + pr + 3, sy(p[1]));
        ctx.arc(sx(p[0]), sy(p[1]), pr + 3, 0, 2 * Math.PI);
      }
      ctx.stroke();
    }

    // The box being selected, and the brush
    if (drag?.kind === 'box') {
      const [a, b] = [drag.from, drag.to];
      ctx.strokeStyle = accent;
      ctx.fillStyle = accent + '22';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      ctx.setLineDash([]);
    }
    if ((tool === 'spray' || tool === 'erase') && hover) {
      ctx.strokeStyle = tool === 'erase' ? cssColour('--danger') : muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hover[0], hover[1], brush * width, 0, 2 * Math.PI);
      ctx.stroke();
    }
  });

  const cursor = $derived(
    tool === 'select' ? (drag?.kind === 'move' ? 'grabbing' : hovered >= 0 ? 'grab' : 'default') : tool === 'add' ? 'copy' : tool ? 'none' : 'default',
  );
</script>

<div class="view" bind:clientWidth={width} style="height: {height}px">
  <canvas
    bind:this={canvas}
    style="width: {width}px; height: {height}px; cursor: {cursor}; touch-action: {tool ? 'none' : 'auto'}"
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
    onpointerleave={() => (hover = null)}
  ></canvas>
</div>
