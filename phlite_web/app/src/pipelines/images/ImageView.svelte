<script lang="ts">
  import { dimColour, type Bar } from '../../lib/types';
  import type { Img } from './data';

  let {
    img,
    t,
    superlevel,
    bar,
    painting = false,
    brush = 4,
    value = 1,
    onpaint,
  }: {
    img: Img;
    t: number | null;
    superlevel: boolean;
    bar: Bar | null;
    painting?: boolean;
    brush?: number;
    value?: number;
    onpaint?: (img: Img) => void;
  } = $props();

  let width = $state(500);
  let canvas: HTMLCanvasElement;
  let hover: number[] | null = $state(null);
  const scale = $derived(Math.min(width / img.width, 560 / img.height));
  const w = $derived(Math.round(img.width * scale));
  const h = $derived(Math.round(img.height * scale));

  // The image, and the sub/superlevel set at t tinted, as an image of the same size
  const base = $derived.by(() => {
    const c = new OffscreenCanvas(img.width, img.height);
    const ctx = c.getContext('2d')!;
    const data = ctx.createImageData(img.width, img.height);
    for (let i = 0; i < img.pixels.length; i++) {
      const v = Math.round(255 * Math.min(1, Math.max(0, img.pixels[i])));
      const inSet = t !== null && (superlevel ? img.pixels[i] >= t : img.pixels[i] <= t);
      data.data[4 * i] = inSet ? Math.round(v * 0.45 + 76 * 0.55) : v;
      data.data[4 * i + 1] = inSet ? Math.round(v * 0.45 + 120 * 0.55) : v;
      data.data[4 * i + 2] = inSet ? Math.round(v * 0.45 + 168 * 0.55 + 40) : v;
      data.data[4 * i + 3] = 255;
    }
    ctx.putImageData(data, 0, 0);
    return c;
  });

  $effect(() => {
    const dpr = devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(base, 0, 0, w, h);

    // Doubled grid coordinates to screen: pixel (X/2, Y/2) at the centre of its square
    const px = (X: number) => (X / 2 + 0.5) * scale;
    const cell = (c: number[], kind: 'rep' | 'death') => {
      const [X, Y] = c;
      const oddX = X % 2 === 1;
      const oddY = Y % 2 === 1;
      if (oddX && oddY) {
        // A square, between four pixels
        ctx.strokeRect(px(X - 1), px(Y - 1), scale, scale);
        if (kind === 'rep') ctx.fillRect(px(X - 1), px(Y - 1), scale, scale);
      } else if (oddX || oddY) {
        ctx.beginPath();
        ctx.moveTo(px(oddX ? X - 1 : X), px(oddY ? Y - 1 : Y));
        ctx.lineTo(px(oddX ? X + 1 : X), px(oddY ? Y + 1 : Y));
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(px(X), px(Y), Math.max(2.5, scale * 0.3), 0, 2 * Math.PI);
        if (kind === 'rep') ctx.fill();
        ctx.stroke();
      }
    };
    if (bar) {
      const colour = dimColour(bar.dim);
      ctx.lineCap = 'round';
      if (bar.rep) {
        ctx.strokeStyle = colour;
        ctx.fillStyle = colour;
        ctx.lineWidth = Math.max(2, Math.min(5, scale * 0.45));
        for (const [, c] of bar.rep) cell(c, 'rep');
      }
      // The pixel where the class is born, and the cell that kills it
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#e45756';
      ctx.setLineDash([3, 2]);
      if (bar.death_cell) cell(bar.death_cell, 'death');
      ctx.setLineDash([]);
    }
    if (painting && hover) {
      ctx.strokeStyle = '#e45756';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hover[0], hover[1], brush * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }
  });

  // ---- Painting: pull pixels towards `value`, with a soft round brush ----
  let down = false;
  function at(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  function paint(p: number[]) {
    if (!onpaint) return;
    const cx = p[0] / scale - 0.5;
    const cy = p[1] / scale - 0.5;
    const pixels = img.pixels.slice();
    const r = brush;
    for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(img.height - 1, Math.ceil(cy + r)); y++)
      for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(img.width - 1, Math.ceil(cx + r)); x++) {
        const d = Math.hypot(x - cx, y - cy) / r;
        if (d >= 1) continue;
        const strength = 0.6 * (1 - d * d);
        const i = y * img.width + x;
        pixels[i] += (value - pixels[i]) * strength;
      }
    onpaint({ ...img, pixels });
  }
</script>

<div class="wrap" bind:clientWidth={width}>
  <div class="view" style="width: {w}px; height: {h}px">
    <canvas
      bind:this={canvas}
      style="width: {w}px; height: {h}px; cursor: {painting ? 'none' : 'default'}"
      onpointerdown={(e) => { if (!painting) return; down = true; canvas.setPointerCapture(e.pointerId); paint(at(e)); }}
      onpointermove={(e) => { hover = at(e); if (down) paint(hover); }}
      onpointerup={() => (down = false)}
      onpointerleave={() => (hover = null)}
    ></canvas>
  </div>
</div>

<style>
  .wrap { width: 100%; display: flex; justify-content: center; }
</style>
