<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import ImageView from './ImageView.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { COMPLEX, defaultsFor } from '../../lib/options';
  import { randomSeed } from '../../lib/random';
  import type { Job } from '../../lib/types';
  import { EXAMPLES, smoothNoise, loadImage, blur, type Img } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let img = $state.raw<Img>(EXAMPLES[0].make());
  let superlevel = $state(EXAMPLES[0].superlevel);
  let options = $state(defaultsFor(COMPLEX));
  const runner = new Runner();

  function useExample(id: string) {
    const ex = EXAMPLES.find((e) => e.id === id)!;
    exampleId = id;
    img = ex.make();
    superlevel = ex.superlevel;
  }

  let noiseSize = $state(64);
  let noiseScale = $state(16);
  let noiseSeed = $state(1);
  function useNoise(newSeed = false) {
    if (newSeed) noiseSeed = randomSeed();
    img = smoothNoise(noiseSize, noiseScale, noiseSeed);
  }

  let maxSide = $state(96);
  let uploadError = $state<string | null>(null);
  let lastFile: File | null = null;
  async function upload(file: File | null) {
    if (!file) return;
    lastFile = file;
    try {
      img = await loadImage(file, maxSide);
      uploadError = null;
    } catch (e) {
      uploadError = `Couldn't read that image: ${(e as Error).message}`;
    }
  }

  let brush = $state(4);
  let paintValue = $state(1);
  let canvasSize = $state(64);
  function blank(fill: number) {
    img = { width: canvasSize, height: canvasSize, pixels: new Float64Array(canvasSize * canvasSize).fill(fill) };
  }
  const invert = () => (img = { ...img, pixels: img.pixels.map((v) => 1 - v) });

  // ---- Computing ----
  const job = $derived<Job>({
    pipeline: 'image',
    pixels: img.pixels,
    width: img.width,
    height: img.height,
    superlevel,
    options: $state.snapshot(options),
  });
  $effect(() => {
    const j = job;
    if (runner.auto) untrack(() => runner.schedule(j));
  });

  const range = $derived.by(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of img.pixels) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return lo === hi ? ([lo, lo + 1] as [number, number]) : ([lo, hi] as [number, number]);
  });
  const domain = $derived<[number, number]>([range[0], range[1] + (range[1] - range[0]) * 0.04]);
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
  const describe = (c: number[]) => {
    const [X, Y] = c;
    const p = (a: number) => a / 2;
    if (X % 2 === 0 && Y % 2 === 0) return `pixel (${p(X)}, ${p(Y)})`;
    if (X % 2 === 1 && Y % 2 === 1) return `square (${p(X - 1)}–${p(X + 1)}, ${p(Y - 1)}–${p(Y + 1)})`;
    return X % 2 === 1 ? `edge (${p(X - 1)}–${p(X + 1)}, ${p(Y)})` : `edge (${p(X)}, ${p(Y - 1)}–${p(Y + 1)})`;
  };
</script>

<div class="layout">
  <div>
    <div class="panel">
      <ImageView {img} t={runner.t} {superlevel} {bar} painting={source === 'draw'} {brush} value={paintValue} onpaint={(i) => (img = i)} />
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} {domain} reversed={superlevel} label="Intensity" />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>{img.width}×{img.height} pixels</span>
        <span><i style="background: #4c78a8"></i>the {superlevel ? 'superlevel' : 'sublevel'} set at the slider's value</span>
        {#if bar}<span>coloured: the representative · red dashes: the cell that kills it</span>{/if}
      </div>
    </div>

    <div class="panel secondary">
      <h3>Image</h3>
      <SourceTabs bind:source drawLabel="Paint" />
      {#if source === 'examples'}
        <div class="chips">
          {#each EXAMPLES as ex}
            <button class:on={exampleId === ex.id} onclick={() => useExample(ex.id)}>{ex.name}</button>
          {/each}
        </div>
        <p class="note">{EXAMPLES.find((e) => e.id === exampleId)?.about}</p>
      {:else if source === 'random'}
        <label class="field"><span>Size</span><input type="range" min="16" max="192" step="8" bind:value={noiseSize} onchange={() => useNoise()} /><span>{noiseSize}²</span></label>
        <label class="field"><span>Feature size</span><input type="range" min="3" max="48" bind:value={noiseScale} onchange={() => useNoise()} /><span>{noiseScale}px</span></label>
        <div class="field"><span>Seed {noiseSeed}</span><button onclick={() => useNoise(true)}>New terrain</button></div>
      {:else if source === 'upload'}
        <FileDrop accept="image/*" onfile={upload}>Any image your browser can open. It is turned to greyscale (brightness) and scaled down.</FileDrop>
        <label class="field"><span>Longest side</span>
          <select bind:value={maxSide} onchange={() => upload(lastFile)}>
            {#each [32, 64, 96, 128, 192, 256] as s}<option value={s}>{s} pixels</option>{/each}
          </select>
        </label>
        {#if uploadError}<p class="error">{uploadError}</p>{/if}
      {:else}
        <div class="tools">
          <button onclick={() => blank(0)}>New black</button>
          <button onclick={() => blank(1)}>New white</button>
          <select bind:value={canvasSize}>
            {#each [32, 48, 64, 96, 128] as s}<option value={s}>{s}×{s}</option>{/each}
          </select>
          <button onclick={() => (img = blur(img))}>Blur</button>
          <button onclick={invert}>Invert</button>
        </div>
        <label class="field"><span>Brush</span><input type="range" min="1" max="16" step="0.5" bind:value={brush} /><span>{brush}px</span></label>
        <label class="field"><span>Paint value</span><input type="range" min="0" max="1" step="0.05" bind:value={paintValue} />
          <span class="swatch" style="background: rgb({paintValue * 255},{paintValue * 255},{paintValue * 255})"></span></label>
        <p class="note">Paint on the image below; the barcode follows.</p>
      {/if}
    </div>
  </div>

  <div>
    <div class="panel">
      <Results
        result={runner.result}
        error={runner.error}
        running={runner.running}
        {domain}
        reversed={superlevel}
        bind:selected={runner.selected}
        bind:t={runner.t}
        {describe}
        name="image"
      />
    </div>

    <div class="panel secondary">
      <h3>Compute</h3>
      <div class="field"><span>Filtration</span>
        <div class="tools">
          <button class:on={!superlevel} onclick={() => (superlevel = false)}>Sublevel (dark first)</button>
          <button class:on={superlevel} onclick={() => (superlevel = true)}>Superlevel (bright first)</button>
        </div>
      </div>
      <p class="note">
        The cubical complex with a vertex per pixel, an edge between neighbouring pixels and a square for each 2×2 block;
        each cell enters with the {superlevel ? 'darkest' : 'brightest'} of its pixels. Reduced as an explicit complex.
      </p>
      <Options bind:options defaults={COMPLEX} />
      <div class="compute-row">
        <button class="primary" onclick={() => runner.run(job)}>Compute</button>
        <label><input type="checkbox" bind:checked={runner.auto} /> recompute on every change</label>
      </div>
    </div>
  </div>
</div>

<style>
  .swatch { width: 1.4rem; height: 1.1rem; border: 1px solid var(--line); border-radius: 3px; }
</style>
