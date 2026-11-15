<script lang="ts">
  import { dimColour, type Bar } from '../../lib/types';
  import { cssColour, rgb } from '../../lib/palette.svelte';
  import type { Img } from './data';

  /** `paint`: pull pixels towards a value with a soft round brush; `smooth`: blur under the brush; `pick`: read a
   * pixel's value (to paint with). */
  export type Tool = 'paint' | 'smooth' | 'pick';

  let {
    img,
    t,
    superlevel,
    construction = 'T',
    bar,
    tool = null,
    brush = 4,
    value = 1,
    onstart,
    onpaint,
    onpick,
  }: {
    img: Img;
    t: number | null;
    superlevel: boolean;
    /** Pixels are squares ('T') or vertices ('V') of the complex. */
    construction?: 'T' | 'V';
    bar: Bar | null;
    tool?: Tool | null;
    /** The brush's radius, in pixels. */
    brush?: number;
    value?: number;
    /** A stroke that may edit the image has begun. */
    onstart?: () => void;
    onpaint?: (img: Img) => void;
    onpick?: (value: number) => void;
  } = $props();
  const painting = $derived(tool === 'paint' || tool === 'smooth');

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
    const tint = rgb(dimColour(0));
    for (let i = 0; i < img.pixels.length; i++) {
      const v = Math.round(255 * Math.min(1, Math.max(0, img.pixels[i])));
      const inSet = t !== null && (superlevel ? img.pixels[i] >= t : img.pixels[i] <= t);
      data.data[4 * i] = inSet ? Math.round(v * 0.45 + tint[0] * 0.55) : v;
      data.data[4 * i + 1] = inSet ? Math.round(v * 0.45 + tint[1] * 0.55) : v;
      data.data[4 * i + 2] = inSet ? Math.round(v * 0.45 + tint[2] * 0.55) : v;
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

    // Doubled grid coordinates to screen. V: pixel p is the vertex 2p, at the centre of its square on screen;
    // T: pixel p is the square 2p + 1, so the vertex 2p is its corner
    const offset = construction === 'V' ? 0.5 : 0;
    const px = (X: number) => (X / 2 + offset) * scale;
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
      ctx.strokeStyle = cssColour('--danger');
      ctx.setLineDash([3, 2]);
      if (bar.death_cell) cell(bar.death_cell, 'death');
      ctx.setLineDash([]);
    }
    if (painting && hover) {
      ctx.strokeStyle = cssColour('--danger');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hover[0], hover[1], brush * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }
  });

  // ---- Painting: pull pixels towards `value` (or their neighbours' mean), with a soft round brush ----
  let down = false;
  function at(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  function pick(p: number[]) {
    const x = Math.min(img.width - 1, Math.max(0, Math.floor(p[0] / scale)));
    const y = Math.min(img.height - 1, Math.max(0, Math.floor(p[1] / scale)));
    onpick?.(img.pixels[y * img.width + x]);
  }
  function paint(p: number[]) {
    if (!onpaint) return;
    const cx = p[0] / scale - 0.5;
    const cy = p[1] / scale - 0.5;
    const src = img.pixels;
    const pixels = src.slice();
    const r = brush;
    const get = (x: number, y: number) => src[Math.min(img.height - 1, Math.max(0, y)) * img.width + Math.min(img.width - 1, Math.max(0, x))];
    for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(img.height - 1, Math.ceil(cy + r)); y++)
      for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(img.width - 1, Math.ceil(cx + r)); x++) {
        const d = Math.hypot(x - cx, y - cy) / r;
        if (d >= 1) continue;
        const i = y * img.width + x;
        const target =
          tool === 'smooth'
            ? (get(x - 1, y - 1) + get(x, y - 1) + get(x + 1, y - 1) + get(x - 1, y) + get(x, y) + get(x + 1, y) + get(x - 1, y + 1) + get(x, y + 1) + get(x + 1, y + 1)) / 9
            : value;
        pixels[i] += (target - pixels[i]) * 0.6 * (1 - d * d);
      }
    onpaint({ ...img, pixels });
  }
  function start(e: PointerEvent) {
    if (!tool || e.button !== 0) return;
    e.preventDefault();
    if (tool === 'pick') {
      pick(at(e));
      return;
    }
    down = true;
    canvas.setPointerCapture(e.pointerId);
    onstart?.();
    paint(at(e));
  }
</script>

<div class="wrap" bind:clientWidth={width}>
  <div class="view" style="width: {w}px; height: {h}px">
    <canvas
      bind:this={canvas}
      style="width: {w}px; height: {h}px; cursor: {painting ? 'none' : tool === 'pick' ? 'crosshair' : 'default'}; touch-action: {tool ? 'none' : 'auto'}"
      onpointerdown={start}
      onpointermove={(e) => { hover = at(e); if (down) paint(hover); }}
      onpointerup={() => (down = false)}
      onpointercancel={() => (down = false)}
      onpointerleave={() => (hover = null)}
    ></canvas>
  </div>
</div>

<style>
  .wrap { width: 100%; display: flex; justify-content: center; }
</style>
