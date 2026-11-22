<script lang="ts">
  // Zigzag persistence of videos: each frame cut at a threshold is a cubical complex, consecutive frames are joined
  // (K_0 -> K_0 ∪ K_1 <- K_1 -> ...), and the barcode follows features from frame to frame.
  import { untrack } from 'svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import ComputeRow from '../../components/ComputeRow.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import ImageView from '../images/ImageView.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import type { Job } from '../../lib/types';
  import { EXAMPLES, decodeVideo, recordCamera, concat, type Video } from './data';

  type Source = 'examples' | 'upload' | 'camera';
  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let video = $state.raw<Video>(EXAMPLES[0].make());
  let threshold = $state(EXAMPLES[0].threshold);
  let superlevel = $state(EXAMPLES[0].superlevel);
  let construction = $state<'T' | 'V'>('T');
  let periodic = $state(false);
  let intersection = $state(false);
  const runner = new Runner();
  runner.t = 0;

  function useExample(id: string) {
    const ex = EXAMPLES.find((e) => e.id === id)!;
    exampleId = id;
    video = ex.make();
    threshold = ex.threshold;
    superlevel = ex.superlevel;
    runner.t = 0;
  }

  let nFrames = $state(40);
  let maxSide = $state(64);
  let loading = $state<string | null>(null);
  let loadError = $state<string | null>(null);
  let lastFile = $state.raw<File | null>(null);
  async function upload(file: File | null) {
    if (!file) return;
    lastFile = file;
    loading = 'Decoding the video…';
    loadError = null;
    try {
      video = await decodeVideo(file, nFrames, maxSide);
      runner.t = 0;
    } catch (e) {
      loadError = (e as Error).message;
    } finally {
      loading = null;
    }
  }
  async function record() {
    loading = 'Recording…';
    loadError = null;
    try {
      video = await recordCamera(nFrames, 100, maxSide);
      runner.t = 0;
    } catch (e) {
      loadError = `The camera isn't available: ${(e as Error).message}`;
    } finally {
      loading = null;
    }
  }

  // ---- Computing ----
  const job = $derived<Job>({
    pipeline: 'video-zigzag',
    pixels: concat(video),
    width: video.width,
    height: video.height,
    nFrames: video.frames.length,
    threshold,
    superlevel,
    construction,
    periodic,
    intersection,
  });
  $effect(() => {
    const j = job;
    if (runner.auto) untrack(() => runner.schedule(j));
  });

  // ---- Viewing: the slider runs over frames (a union between frames i and i + 1 at i + 0.5) ----
  const domain = $derived<[number, number]>([0, Math.max(1, video.frames.length)]);
  const frameIndex = $derived(Math.min(video.frames.length - 1, Math.max(0, Math.floor(runner.t ?? 0))));
  const frame = $derived(video.frames[frameIndex]);
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
</script>

<div class="layout">
  <div>
    <div class="panel">
      {#if frame}
        <ImageView img={frame} t={threshold} {superlevel} {construction} bar={null} />
      {/if}
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} domain={[0, Math.max(1, video.frames.length - 1)]} label="Frame" />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>frame {frameIndex + 1} of {video.frames.length}, {video.width}×{video.height}</span>
        <span><i style="background: var(--dim0)"></i>pixels {superlevel ? 'at least' : 'at most'} the threshold</span>
        {#if bar}<span>selected bar: frames {bar.birth} to {bar.death === null ? 'the end' : bar.death}</span>{/if}
      </div>
    </div>

    <div class="panel secondary">
      <h3>Video</h3>
      <div class="tools" role="tablist">
        <button class:on={source === 'examples'} onclick={() => (source = 'examples')}>Examples</button>
        <button class:on={source === 'upload'} onclick={() => (source = 'upload')}>Upload</button>
        <button class:on={source === 'camera'} onclick={() => (source = 'camera')}>Camera</button>
      </div>
      {#if source === 'examples'}
        <div class="chips">
          {#each EXAMPLES as ex}
            <button class:on={exampleId === ex.id} onclick={() => useExample(ex.id)}>{ex.name}</button>
          {/each}
        </div>
        <p class="note">{EXAMPLES.find((e) => e.id === exampleId)?.about}</p>
      {:else}
        <label class="field"><span>Frames</span><input type="range" min="4" max="120" bind:value={nFrames} /><span>{nFrames}</span></label>
        <label class="field"><span>Longest side</span>
          <select bind:value={maxSide}>
            {#each [24, 32, 48, 64, 96, 128] as s}<option value={s}>{s} pixels</option>{/each}
          </select>
        </label>
        {#if source === 'upload'}
          <FileDrop accept="video/*" onfile={upload}>
            Any video your browser can play: {nFrames} frames are taken evenly through it, turned to greyscale and scaled
            down. Nothing leaves your computer.
          </FileDrop>
          {#if lastFile}<button onclick={() => upload(lastFile)}>Take the frames again</button>{/if}
        {:else}
          <button onclick={record} disabled={loading !== null}>Record {nFrames} frames ({(nFrames / 10).toFixed(1)} s)</button>
          <p class="note">Wave a dark object in front of the camera: it becomes a component that the barcode follows. Frames stay in your browser.</p>
        {/if}
        {#if loading}<p class="note">{loading}</p>{/if}
        {#if loadError}<p class="error">{loadError}</p>{/if}
      {/if}
    </div>
  </div>

  <div>
    <div class="panel">
      <Results result={runner.result} error={runner.error} running={runner.running} {domain} bind:selected={runner.selected} bind:t={runner.t} name="video-zigzag" />
    </div>

    <div class="panel secondary">
      <h3>Compute</h3>
      <label class="field"><span>Threshold</span><input type="range" min="0" max="1" step="0.01" bind:value={threshold} /><span>{threshold.toFixed(2)}</span></label>
      <div class="field"><span>Pixels</span>
        <div class="tools">
          <button class:on={!superlevel} onclick={() => (superlevel = false)}>Dark (at most)</button>
          <button class:on={superlevel} onclick={() => (superlevel = true)}>Bright (at least)</button>
        </div>
      </div>
      <div class="field"><span>Construction</span>
        <div class="tools">
          <button class:on={construction === 'T'} onclick={() => (construction = 'T')}>T: pixels as squares</button>
          <button class:on={construction === 'V'} onclick={() => (construction = 'V')}>V: pixels as vertices</button>
        </div>
      </div>
      <div class="field"><span>Join</span>
        <div class="tools">
          <button class:on={!intersection} onclick={() => (intersection = false)}>Unions</button>
          <button class:on={intersection} onclick={() => (intersection = true)}>Intersections</button>
        </div>
      </div>
      <label class="field"><span>Periodic</span><input type="checkbox" bind:checked={periodic} /><span class="note">wrap the frames round (a torus)</span></label>
      <p class="note">
        <strong>Zigzag persistence</strong> of the video: each frame cut at the threshold is a cubical complex, and each is
        joined to the next by their union (K₀ → K₀ ∪ K₁ ← K₁ → …) or their intersection (K₀ ← K₀ ∩ K₁ → K₁ → …).
        A bar is a component (H₀) or a loop (H₁) followed from frame to frame, for as long as it overlaps itself (with
        unions) or what it shares with the next frame carries it (with intersections): merging ends a bar, splitting
        starts one. The axis counts frames (a bar ending at i + 0.5 lasted into the join of frames i and i + 1). Computed by FastZigzag (Dey and Hou), as
        <code>phlite.images.video_zigzag</code>.
      </p>
      <ComputeRow {runner} {job} />
    </div>
  </div>
</div>
