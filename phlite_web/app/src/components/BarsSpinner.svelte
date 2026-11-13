<!-- The landing page's barcode, with its bars growing and shrinking: shown while phlite computes. -->
<script lang="ts">
  let { size = 26 }: { size?: number } = $props();
  // [x, y, width, colour, opacity] as on the landing page, scaled to a 44 × 24 box
  const bars: [number, number, number, string, number][] = [
    [0, 0, 40, '#4c78a8', 1],
    [0, 4.5, 14, '#4c78a8', 0.75],
    [0, 9, 7, '#4c78a8', 0.55],
    [11, 14, 26, '#f58518', 1],
    [19, 18.5, 8, '#f58518', 0.6],
  ];
</script>

<svg class="spinner" width={size * (44 / 24)} height={size} viewBox="0 0 44 24" role="img" aria-label="Computing">
  {#each bars as [x, y, w, fill, opacity], i}
    <rect {x} {y} width={w} height="3" rx="1" {fill} {opacity} style="animation-delay: {i * -0.22}s; transform-origin: {x}px {y}px" />
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
