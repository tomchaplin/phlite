<script lang="ts">
  import type { Runner } from '../lib/runner.svelte';
  import type { Job } from '../lib/types';

  /**
   * Compute, and when to recompute: on every change (a moment after the last one, cancelling a computation still
   * running), or live (as the data is edited: each computation runs to the end, and then the latest edit is computed).
   */
  let { runner, job, disabled = false }: { runner: Runner; job: Job | null; disabled?: boolean } = $props();
</script>

<div class="compute-row">
  <button class="primary" disabled={!job || disabled} onclick={() => runner.run(job)}>Compute</button>
  <label><input type="checkbox" bind:checked={runner.auto} /> recompute on every change</label>
  <label
    class:off={!runner.auto}
    title="Recompute while you edit the data (drag a point, paint a stroke), not just after: best when a computation takes a few milliseconds"
  >
    <input type="checkbox" bind:checked={runner.live} disabled={!runner.auto} /> live
  </label>
</div>

<style>
  .off { opacity: 0.5; }
</style>
