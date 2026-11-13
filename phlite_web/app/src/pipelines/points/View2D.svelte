<script lang="ts">
  import { dimColour, type Bar } from '../../lib/types';
  import { countUpTo } from './data';

  export type Tool = 'spray' | 'point' | 'erase';

  let {
    points,
    t,
    filtration,
    bar,
    balls,
    edges,
    tool = null,
    brush = 0.08,
    box: fixedBox = null,
    onedit,
  }: {
    points: number[][];
    t: number | null;
    filtration: 'rips' | 'alpha';
    bar: Bar | null;
    balls: boolean;
    edges: { pairs: Uint32Array; lengths: Float64Array } | null;
    tool?: Tool | null;
    brush?: number;
    /** A fixed view box [x0, y0, x1, y1] (while drawing), or null to fit the points. */
    box?: number[] | null;
    onedit?: (points: number[][]) => void;
  } = $props();

  let width = $state(500);
  let canvas: HTMLCanvasElement;
  let hover: number[] | null = $state(null);
  const height = $derived(Math.round(width * 0.8));
  const EDGE_LIMIT = 30000;

  const box = $derived.by(() => {
    if (fixedBox) return fixedBox;
    if (points.length === 0) return [-1, -1, 1, 1];
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const [x, y] of points) {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    const pad = Math.max(x1 - x0, y1 - y0, 1e-6) * 0.08;
    return [x0 - pad, y0 - pad, x1 + pad, y1 + pad];
  });
  // World to screen, keeping the aspect ratio
  const scale = $derived(Math.min(width / (box[2] - box[0]), height / (box[3] - box[1])));
  const ox = $derived((width - scale * (box[2] - box[0])) / 2);
  const oy = $derived((height - scale * (box[3] - box[1])) / 2);
  const sx = (x: number) => ox + (x - box[0]) * scale;
  const sy = (y: number) => height - oy - (y - box[1]) * scale;
  const wx = (px: number) => box[0] + (px - ox) / scale;
  const wy = (py: number) => box[1] + (height - oy - py) / scale;

  $effect(() => {
    const dpr = devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(canvas);
    const fg = css.getPropertyValue('--fg').trim() || '#222';
    const muted = css.getPropertyValue('--muted').trim() || '#888';
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
      if (filtration === 'rips' && edges) {
        const m = Math.min(countUpTo(edges.lengths, t), EDGE_LIMIT);
        ctx.strokeStyle = muted;
        ctx.globalAlpha = m > 3000 ? 0.25 : 0.55;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let k = 0; k < m; k++) {
          const a = points[edges.pairs[2 * k]];
          const b = points[edges.pairs[2 * k + 1]];
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

    // The points
    const pr = points.length > 1000 ? 1.6 : points.length > 300 ? 2.2 : 3;
    ctx.fillStyle = fg;
    ctx.beginPath();
    for (const [x, y] of points) {
      ctx.moveTo(sx(x) + pr, sy(y));
      ctx.arc(sx(x), sy(y), pr, 0, 2 * Math.PI);
    }
    ctx.fill();

    // The brush
    if (tool && tool !== 'point' && hover) {
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hover[0], hover[1], brush * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }
  });

  // ---- Drawing ----
  let down = false;
  let last: number[] | null = null;
  function at(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  function apply(px: number[], first: boolean) {
    if (!tool || !onedit) return;
    const [x, y] = [wx(px[0]), wy(px[1])];
    if (tool === 'point') {
      if (first) onedit([...points, [x, y]]);
      return;
    }
    if (tool === 'erase') {
      const kept = points.filter(([a, b]) => Math.hypot(a - x, b - y) > brush);
      if (kept.length !== points.length) onedit(kept);
      return;
    }
    // Spray: a few points per brush-width moved
    if (!first && last && Math.hypot(px[0] - last[0], px[1] - last[1]) < brush * scale * 0.5) return;
    last = px;
    const added = Array.from({ length: 4 }, () => {
      const a = 2 * Math.PI * Math.random();
      const r = brush * Math.sqrt(Math.random());
      return [x + r * Math.cos(a), y + r * Math.sin(a)];
    });
    onedit([...points, ...added]);
  }
</script>

<div class="view" bind:clientWidth={width} style="height: {height}px">
  <canvas
    bind:this={canvas}
    style="width: {width}px; height: {height}px; cursor: {tool ? 'crosshair' : 'default'}"
    onpointerdown={(e) => { if (!tool) return; down = true; last = null; canvas.setPointerCapture(e.pointerId); apply(at(e), true); }}
    onpointermove={(e) => { hover = at(e); if (down) apply(hover, false); }}
    onpointerup={() => (down = false)}
    onpointerleave={() => (hover = null)}
  ></canvas>
</div>
