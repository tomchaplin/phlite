<script lang="ts">
  import { untrack } from 'svelte';
  import { engine } from '../lib/engine.svelte';
  import { defaultAlgorithm, defaultsFor, isDefault, isPreset, type PipelineDefaults } from '../lib/options';
  import type { AlgoOptions } from '../lib/types';

  let {
    options = $bindable(),
    defaults,
    allowHomology = true,
  }: { options: Required<AlgoOptions>; defaults: PipelineDefaults; allowHomology?: boolean } = $props();

  let advanced = $state(false);
  let threads = $state(engine.requested);

  function setAlgorithm(algorithm: AlgoOptions['algorithm']) {
    // The dependent options follow the algorithm's preset
    const preset = defaultsFor(defaults, algorithm);
    options = { ...preset, method: options.method, representatives: options.representatives };
  }

  // When the default algorithm changes (the parallel switch, or a fallback to one thread), options that were the old
  // default's preset follow it
  let previous = defaultAlgorithm();
  $effect(() => {
    const algorithm = defaultAlgorithm();
    untrack(() => {
      if (algorithm !== previous && isPreset(defaults, options, previous)) setAlgorithm(algorithm);
      previous = algorithm;
    });
  });

  const columnsHelp: Record<string, string> = {
    merged: 'merges the added columns lazily: the safe choice',
    heap: 'one heap of every entry added: good when few, short columns are added',
    sorted: 'a sorted vector, merged in batches: fast when additions stay near the pivot',
    pivot: 'a dense bit-tree over all rows (as PHAT): fast when many columns are added',
  };
  const book = '../book/';
</script>

<div class="options">
  <div class="row">
    <label for="algo">Algorithm</label>
    <select id="algo" value={options.algorithm} onchange={(e) => setAlgorithm(e.currentTarget.value as AlgoOptions['algorithm'])}>
      <option value="clearing">Clearing (serial)</option>
      <option value="parallel" disabled={engine.single}>Parallel clearing{engine.single ? ' (no threads here)' : ''}</option>
      <option value="standard">Standard (no optimisations)</option>
    </select>
  </div>
  <div class="row">
    <label for="method">Method</label>
    <select id="method" bind:value={options.method} disabled={!allowHomology}>
      <option value="cohomology">Cohomology</option>
      <option value="homology">Homology</option>
    </select>
  </div>
  <div class="row">
    <label class="check"><input type="checkbox" bind:checked={options.representatives} /> Representatives</label>
  </div>

  <button class="link" onclick={() => (advanced = !advanced)}>{advanced ? '▾' : '▸'} Advanced options</button>
  {#if advanced}
    <div class="advanced">
      <div class="row">
        <label for="cols">Columns</label>
        <select id="cols" bind:value={options.columns}>
          <option value="merged">merged</option>
          <option value="heap">heap</option>
          <option value="sorted">sorted</option>
          <option value="pivot" disabled={!defaults.pivot || options.algorithm !== 'clearing'}>pivot</option>
        </select>
      </div>
      <p class="hint">{columnsHelp[options.columns]}. <a href="{book}columns/choosing.html" target="_blank">Choosing columns</a></p>
      <div class="row">
        <label class="check"><input type="checkbox" bind:checked={options.cache} /> Cache columns</label>
      </div>
      <div class="row">
        <label class="check" class:disabled={options.algorithm !== 'clearing'}>
          <input type="checkbox" bind:checked={options.storeR} disabled={options.algorithm !== 'clearing'} /> Store R
        </label>
      </div>
      <div class="row">
        <label class="check" class:disabled={options.algorithm === 'standard'}>
          <input type="checkbox" bind:checked={options.skipZeroApparentPairs} disabled={options.algorithm === 'standard'} />
          Skip zero-length apparent pairs
        </label>
      </div>
      {#if options.algorithm === 'parallel'}
        <div class="row">
          <label for="storage">Storage</label>
          <select id="storage" bind:value={options.storage}>
            <option value="locking">locking</option>
            <option value="lock-free">lock-free</option>
          </select>
        </div>
        <div class="row">
          <label for="threads">Threads</label>
          <input id="threads" type="number" min="1" max={engine.maxThreads} bind:value={threads} />
          <button disabled={threads === engine.threads || threads < 1} onclick={() => engine.start(threads)}>Restart</button>
        </div>
      {/if}
      <p class="hint">
        How these work: <a href="{book}algorithms/clearing.html" target="_blank">clearing</a>,
        <a href="{book}algorithms/apparent-pairs.html" target="_blank">apparent pairs</a>,
        <a href="{book}algorithms/store-r.html" target="_blank">storing R</a>,
        <a href="{book}algorithms/parallel.html" target="_blank">parallel</a>,
        <a href="{book}columns/caching.html" target="_blank">caching</a>.
      </p>
    </div>
  {/if}
  <div class="row foot">
    {#if isDefault(defaults, options)}
      <span class="badge">defaults</span>
    {:else}
      <button class="link" onclick={() => (options = defaultsFor(defaults))}>Reset to defaults</button>
    {/if}
  </div>
</div>

<style>
  .options { display: flex; flex-direction: column; gap: 0.35rem; }
  .row { display: flex; align-items: center; gap: 0.5rem; }
  .row > label:first-child:not(.check) { width: 6.5rem; color: var(--muted); }
  .row select, .row input[type='number'] { flex: 1; min-width: 0; }
  .row input[type='number'] { max-width: 5rem; }
  .check { display: flex; align-items: center; gap: 0.4rem; }
  .disabled { opacity: 0.5; }
  .advanced { display: flex; flex-direction: column; gap: 0.35rem; padding: 0.4rem 0 0.2rem 0.8rem; border-left: 2px solid var(--line); }
  .hint { margin: 0; font-size: 0.82rem; color: var(--muted); }
  .options > .link { align-self: flex-start; }
  .foot { justify-content: flex-end; min-height: 1.4rem; }
  .badge { font-size: 0.75rem; color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 0 0.5rem; }
</style>
