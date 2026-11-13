<script lang="ts">
  import { fmt } from '../lib/format';

  let {
    t = $bindable(),
    domain,
    reversed = false,
    label = 'Filtration value',
  }: { t: number | null; domain: [number, number]; reversed?: boolean; label?: string } = $props();

  let playing = $state(false);
  let frame = 0;
  // The slider runs in the direction of the filtration
  const lo = $derived(reversed ? -domain[1] : domain[0]);
  const hi = $derived(reversed ? -domain[0] : domain[1]);
  const pos = $derived(t === null ? hi : reversed ? -t : t);

  function set(v: number) {
    t = reversed ? -v : v;
  }

  function play() {
    if (playing) {
      playing = false;
      cancelAnimationFrame(frame);
      return;
    }
    playing = true;
    if (pos >= hi - (hi - lo) * 0.01) set(lo);
    let last = performance.now();
    const tick = (now: number) => {
      if (!playing) return;
      // About 8 seconds for the whole filtration
      const next = pos + ((now - last) / 8000) * (hi - lo);
      last = now;
      if (next >= hi) {
        set(hi);
        playing = false;
        return;
      }
      set(next);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }
</script>

<div class="slider">
  <button class="play" onclick={play} title={playing ? 'Pause' : 'Play the filtration'} aria-label="Play">{playing ? '❚❚' : '▶'}</button>
  <span class="label">{label}</span>
  <input
    type="range"
    min={lo}
    max={hi}
    step={(hi - lo) / 1000 || 0.001}
    value={pos}
    oninput={(e) => { playing = false; set(+e.currentTarget.value); }}
  />
  <span class="value">{t === null ? 'all' : fmt(t)}</span>
  <button class="link" onclick={() => { playing = false; t = null; }} disabled={t === null}>all</button>
</div>

<style>
  .slider { display: flex; align-items: center; gap: 0.5rem; }
  .label { color: var(--muted); font-size: 0.85rem; white-space: nowrap; }
  input { flex: 1; min-width: 4rem; }
  .value { font-variant-numeric: tabular-nums; min-width: 3.5rem; font-size: 0.9rem; }
  .play { width: 2rem; padding: 0.15rem 0; }
</style>
