<!-- The landing page's barcode, with its bars growing and shrinking: shown while phlite computes. -->
<script lang="ts">
  import { dimColour } from '../lib/palette.svelte';

  let { size = 26 }: { size?: number } = $props();
  // [x, y, width, dimension, opacity] as on the landing page, scaled to a 44 × 24 box
  const bars: [number, number, number, number, number][] = [
    [0, 0, 40, 0, 1],
    [0, 4.5, 14, 0, 0.75],
    [0, 9, 7, 0, 0.55],
    [11, 14, 26, 1, 1],
    [19, 18.5, 8, 1, 0.6],
  ];
</script>

<svg class="spinner" width={size * (44 / 24)} height={size} viewBox="0 0 44 24" role="img" aria-label="Computing">
  {#each bars as [x, y, w, dim, opacity], i}
    <rect {x} {y} width={w} height="3" rx="1" fill={dimColour(dim)} {opacity} style="animation-delay: {i * -0.22}s; transform-origin: {x}px {y}px" />
  {/each}
</svg>

<style>
  .spinner { display: inline-block; vertical-align: middle; overflow: visible; }
  rect { animation: grow 1.3s ease-in-out infinite alternate; }
  @keyframes grow {
    from { transform: scaleX(0.15); }
    to { transform: scaleX(1); }
  }
  @media (prefers-reduced-motion: reduce) {
    rect { animation: none; }
  }
</style>
