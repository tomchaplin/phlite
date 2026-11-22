<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import ComputeRow from '../../components/ComputeRow.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import EditBar from '../../components/EditBar.svelte';
  import Segmented from '../../components/Segmented.svelte';
  import More from '../../components/More.svelte';
  import { JOINS } from '../../lib/joins';
  import { History } from '../../lib/history.svelte';
  import View2D, { type Tool } from './View2D.svelte';
  import View3D from './View3D.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { RIPS, COMPLEX, defaultsFor } from '../../lib/options';
  import { barDomain } from '../../lib/scale';
  import { randomSeed } from '../../lib/random';
  import type { Job, Metric, ZigzagCloudComplex } from '../../lib/types';
  import { EXAMPLES, RANDOM_KINDS, MOTIONS, randomCloud, parseCloud, parseFrames, moveCloud, flatten, type Cloud, type Motion, type RandomKind } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let cloud = $state.raw<Cloud>(EXAMPLES[0].make());
  /** Bumped whenever the cloud is replaced rather than edited, to fit the view to it. */
  let fitKey = $state(0);
  const history = new History<Cloud>();
  /** Replace the cloud (undoably). */
  function replace(next: Cloud) {
    history.commit(cloud);
    cloud = next;
    selection = [];
    fitKey++;
  }
  type Filtration = 'rips' | 'dtm' | 'alpha' | 'cech' | 'sixpack' | 'zigzag' | 'moving';
  /** For the sliding-window zigzag: the number of consecutive points in a window. */
  let windowSize = $state(20);
  /** For the zigzag: join consecutive windows by intersections rather than unions. */
  let intersection = $state(false);
  /** For the zigzag: windows start every `windowStep` points. */
  let windowStep = $state(1);
  /** For the zigzag: each window's complex. */
  let zzComplex = $state<ZigzagCloudComplex>('rips');
  const ZZ_COMPLEXES: { id: ZigzagCloudComplex; name: string; scale: string }[] = [
    { id: 'rips', name: 'Vietoris–Rips', scale: 'the largest distance within a simplex' },
    { id: 'cech', name: 'Čech', scale: 'the radius of the balls (a simplex enters when they all meet)' },
    { id: 'alpha', name: 'Alpha', scale: 'the alpha radius (points in the plane or in space)' },
    { id: 'delaunay-cech', name: 'Delaunay–Čech', scale: 'the radius of the minimal enclosing ball (points in the plane or in space)' },
    { id: 'witness', name: 'Witness', scale: 'α²: each window’s points are landmarks, every point of the cloud a witness' },
  ];
  /** The points of the window at the slider (its index, rounded down). */
  const zzActive = $derived.by((): [number, number] | null => {
    if (filtration !== 'zigzag' || runner.t === null) return null;
    const n = cloud.points.length;
    const w = Math.min(windowSize, n);
    const last = Math.floor((n - w) / Math.max(1, windowStep));
    const k = Math.min(last, Math.max(0, Math.floor(runner.t)));
    return [k * windowStep, k * windowStep + w];
  });
  let filtration = $state<Filtration>('rips');
  let dtmK = $state(10);
  const isDelaunay = $derived(filtration === 'alpha' || filtration === 'cech');
  let maxDim = $state(EXAMPLES[0].maxDim);
  let threshold = $state('');
  let metric = $state<Metric>('euclidean');
  let options = $state(defaultsFor(RIPS));
  let showBalls = $state(false);
  /** Balls show the complex for Rips in the Euclidean metric, alpha and Čech. */
  const ballsShow = $derived(isDelaunay || (filtration === 'rips' && metric === 'euclidean'));
  const runner = new Runner();
  /** Either zigzag: windows of the cloud, or the cloud's points moving. */
  const isZigzag = $derived(filtration === 'zigzag' || filtration === 'moving');
  /** What kind of thing is computed: a filtration's barcode, a zigzag, or the maps of a subset. */
  const kind = $derived(isZigzag ? 'zigzag' : filtration === 'sixpack' ? 'maps' : 'filtration');
  // ---- Moving points: frames of the cloud moving (or uploaded) ----
  let motion = $state<Motion>('breathe');
  let nMovFrames = $state(20);
  let amount = $state(0.6);
  let movSeed = $state(1);
  let uploadedFrames = $state.raw<{ dim: 2 | 3; frames: number[][][] } | null>(null);
  let framesError = $state<string | null>(null);
  async function uploadFrames(file: File) {
    try {
      const parsed = parseFrames(await file.text());
      replace({ dim: parsed.dim, points: parsed.frames[0] });
      uploadedFrames = parsed; // used while the cloud is still their first frame
      framesError = null;
    } catch (e) {
      framesError = (e as Error).message;
    }
  }
  /** The frames: uploaded ones while the cloud is still their first frame, or the cloud moved by the motion. */
  const frames = $derived.by((): number[][][] => {
    if (filtration !== 'moving') return [];
    if (uploadedFrames && uploadedFrames.frames[0] === cloud.points) return uploadedFrames.frames;
    return moveCloud(cloud, motion, nMovFrames, amount, movSeed);
  });
  const frameIndex = $derived(Math.min(frames.length - 1, Math.max(0, Math.floor(runner.t ?? 0))));
  /** The points to draw: the slider's frame, when moving. */
  const shown = $derived(filtration === 'moving' && frames.length && runner.t !== null ? frames[frameIndex] : cloud.points);

  // ---- Data sources ----
  function useExample(id: string) {
    const ex = EXAMPLES.find((e) => e.id === id)!;
    exampleId = id;
    replace(ex.make());
    maxDim = ex.maxDim;
  }

  let randomKind = $state<RandomKind>('uniform');
  let randomN = $state(80);
  let randomNoise = $state(0.08);
  let randomSeedValue = $state(1);
  function useRandom(newSeed = false) {
    if (newSeed) randomSeedValue = randomSeed();
    replace(randomCloud(randomKind, randomN, randomNoise, randomSeedValue));
  }

  let uploadError = $state<string | null>(null);
  async function upload(file: File) {
    try {
      replace(parseCloud(await file.text()));
      uploadError = null;
    } catch (e) {
      uploadError = (e as Error).message;
    }
  }

  // ---- Editing (points in the plane) ----
  let tool = $state<Tool | null>(null);
  let brush = $state(0.035);
  let selection = $state<number[]>([]);
  const TOOLS: [Tool, string, string][] = [
    ['select', 'Select / move', 'Click or drag a box to select points (shift adds); drag a selected point to move them all'],
    ['add', 'Add', 'Click to add a point; keep the button down to drag it into place'],
    ['spray', 'Spray', 'Spray points with a round brush'],
    ['erase', 'Erase', 'Erase the points under a round brush'],
  ];
  function edited(points: number[][]) {
    history.edited();
    cloud = { dim: 2, points };
  }
  function deleteSelected() {
    if (selection.length === 0) return;
    const gone = new Set(selection);
    history.commit(cloud);
    cloud = { dim: cloud.dim, points: cloud.points.filter((_, i) => !gone.has(i)) };
    selection = [];
  }
  function undo() {
    const previous = history.undo(cloud);
    if (previous) {
      cloud = previous;
      selection = [];
    }
  }
  function redo() {
    const next = history.redo(cloud);
    if (next) {
      cloud = next;
      selection = [];
    }
  }
  // A new cloud: an empty plane to draw on
  function blank() {
    replace({ dim: 2, points: [] });
    tool = 'spray';
  }
  $effect(() => {
    if (cloud.dim !== 2) untrack(() => (tool = null));
  });

  function setFiltration(f: Filtration) {
    filtration = f;
    options = defaultsFor(f === 'rips' || f === 'dtm' || f === 'sixpack' ? RIPS : COMPLEX);
    if (f === 'zigzag' || f === 'moving') runner.t = 0;
    if (f === 'moving' && zzComplex === 'witness') zzComplex = 'rips';
    if (f === 'sixpack') {
      options.representatives = true;
      tool = cloud.dim === 2 ? 'select' : null;
    }
    showBalls = f === 'alpha' || f === 'cech';
  }

  // ---- Computing ----
  const thresholdValue = $derived(threshold.trim() === '' ? null : Number(threshold));
  const problem = $derived.by(() => {
    if (cloud.points.length === 0) return 'No points yet.';
    if (isDelaunay && cloud.dim !== 2 && cloud.dim !== 3) return 'Alpha and Delaunay–Čech persistence are for points in the plane or in space.';
    if (thresholdValue !== null && !(thresholdValue > 0)) return 'The threshold must be a positive number.';
    if (filtration === 'sixpack' && cloud.dim !== 2) return 'Subset maps are for clouds in the plane, where you can select the subset.';
    if (filtration === 'sixpack' && selection.length === 0) return 'Select some points (with the Select tool, or a box) to be the subset.';
    if (isZigzag && thresholdValue === null) return 'Give a threshold: the scale of the complexes.';
    if (filtration === 'moving' && zzComplex === 'witness') return 'Witness complexes are for sliding windows (each window\'s points the landmarks).';
    if (isZigzag && (zzComplex === 'alpha' || zzComplex === 'delaunay-cech') && cloud.dim !== 2 && cloud.dim !== 3)
      return 'Alpha and Delaunay–Čech complexes are for points in the plane or in space.';
    if (filtration === 'zigzag' && zzComplex === 'cech' && Math.min(windowSize, cloud.points.length) > 60 && maxDim >= 1)
      return 'Čech complexes list every simplex of up to four points within the radius: slow for windows of more than about 60 points. Compute anyway?';
    if (filtration === 'zigzag' && !(windowSize >= 1 && windowSize <= cloud.points.length)) return `The window must have between 1 and ${cloud.points.length} points.`;
    if (filtration === 'dtm' && !(dtmK >= 1 && dtmK <= cloud.points.length)) return `k must be between 1 and the number of points (${cloud.points.length}).`;
    if (options.algorithm === 'standard' && !isDelaunay && cloud.points.length > 120)
      return 'The standard algorithm builds every simplex, which takes a very long time for more than about 120 points. Compute anyway?';
    return null;
  });
  const job = $derived.by((): Job | null => {
    if (cloud.points.length === 0) return null;
    const opts = $state.snapshot(options);
    if (isDelaunay) return { pipeline: 'delaunay', coords: flatten(cloud), pointDim: cloud.dim, cech: filtration === 'cech', options: opts };
    if (filtration === 'moving') {
      if (thresholdValue === null || frames.length === 0 || zzComplex === 'witness') return null;
      return {
        pipeline: 'moving-zigzag',
        coords: Float64Array.from(frames.flat(2)),
        pointDim: cloud.dim,
        nFrames: frames.length,
        threshold: thresholdValue,
        maxDim: Math.min(maxDim, 2),
        metric,
        complex: zzComplex,
        intersection,
      };
    }
    if (filtration === 'zigzag') {
      if (thresholdValue === null) return null;
      return {
        pipeline: 'cloud-zigzag',
        coords: flatten(cloud),
        pointDim: cloud.dim,
        window: Math.min(windowSize, cloud.points.length),
        step: Math.max(1, windowStep),
        threshold: thresholdValue,
        maxDim: Math.min(maxDim, 2),
        metric,
        complex: zzComplex,
        intersection,
      };
    }
    if (filtration === 'sixpack') {
      if (selection.length === 0) return null;
      return {
        pipeline: 'rips-six-pack',
        coords: flatten(cloud),
        pointDim: cloud.dim,
        subset: Uint32Array.from([...selection].sort((a, b) => a - b)),
        maxDim: Math.min(maxDim, 2),
        threshold: thresholdValue,
        metric,
        options: opts,
      };
    }
    return {
      pipeline: 'rips',
      coords: flatten(cloud),
      pointDim: cloud.dim,
      maxDim,
      threshold: thresholdValue,
      metric,
      dtmK: filtration === 'dtm' ? Math.min(dtmK, cloud.points.length) : null,
      options: opts,
    };
  });
  const blocked = $derived(problem !== null && !problem.endsWith('anyway?'));
  const risky = $derived(problem?.endsWith('anyway?') ?? false);
  $effect(() => {
    const j = job;
    if (runner.auto && !blocked && !risky) untrack(() => runner.schedule(j));
  });

  // ---- Viewing ----
  // Rips and DTM-Rips filter the complete complex on the points (a flag filtration): draw its complex at t
  const isFlag = $derived(filtration === 'rips' || filtration === 'dtm' || filtration === 'sixpack');
  // The edges are phlite's, sent with the barcode, so the complex drawn is the one computed. Only while the result is
  // for these points (live, a result one edit behind with the same number of points is drawn on the moved points)
  const edges = $derived.by(() => {
    const job = runner.resultJob;
    const edges = runner.result?.barcode.edges;
    if (!isFlag || !edges || (job?.pipeline !== 'rips' && job?.pipeline !== 'rips-six-pack')) return null;
    if (job.coords.length !== cloud.points.length * cloud.dim) return null;
    return edges;
  });
  const domain = $derived<[number, number]>(
    filtration === 'moving' ? [0, Math.max(1, frames.length)] : barDomain(runner.result?.barcode.bars ?? [], [0]),
  );
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
  const describe = (cell: number[]) => (cell.length === 1 ? `point ${cell[0]}` : `[${cell.join(' ')}]`);
</script>

<div class="layout">
  <div>
    <div class="panel">
      {#if cloud.dim === 2}
        <View2D
          points={shown}
          t={runner.t}
          filtration={isDelaunay ? 'alpha' : 'rips'}
          {bar}
          balls={showBalls && ballsShow}
          {edges}
          tool={filtration === 'moving' ? null : tool}
          {brush}
          bind:selection
          active={zzActive}
          {fitKey}
          onstart={() => history.begin(cloud)}
          onedit={edited}
        />
      {:else}
        <View3D points={shown} t={runner.t} {bar} {edges} />
      {/if}
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} {domain} label={filtration === 'zigzag' ? 'Window' : filtration === 'moving' ? 'Frame' : isDelaunay ? 'Radius' : 'Diameter'} />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>{cloud.points.length} points{cloud.dim === 3 ? ' in space' : ''}</span>
        {#if cloud.dim === 2 && ballsShow}<label><input type="checkbox" bind:checked={showBalls} /> balls of radius {filtration === 'rips' ? 'd/2' : 'r'}</label>{/if}
        {#if isFlag && edges}<span><i style="background: var(--edges)"></i>the complex at the slider's value: edges and (shaded) triangles{filtration === 'dtm' ? '; hollow points are not yet born' : ''}</span>{/if}
        {#if bar}<span>coloured: the representative · dashed: the simplex that kills it</span>{/if}
      </div>
      <EditBar
        bind:tool
        tools={cloud.dim === 2 ? TOOLS : []}
        canUndo={history.past.length > 0}
        canRedo={history.future.length > 0}
        onundo={undo}
        onredo={redo}
        ondelete={deleteSelected}
      >
        {#if cloud.dim !== 2}
          <span class="note">Editing is for clouds in the plane; this one is in space.</span>
        {:else if tool === 'spray' || tool === 'erase'}
          <label class="field"><span>Brush size</span><input type="range" min="0.005" max="0.12" step="any" bind:value={brush} /></label>
        {:else if tool === 'select'}
          <span class="note">{selection.length} selected</span>
          <button disabled={selection.length === 0} onclick={deleteSelected}>Delete selected</button>
          <button disabled={cloud.points.length === 0} onclick={() => (selection = cloud.points.map((_, i) => i))}>Select all</button>
        {/if}
        {#if cloud.dim === 2 && tool}
          <button disabled={cloud.points.length === 0} onclick={() => replace({ dim: 2, points: [] })}>Clear</button>
        {/if}
      </EditBar>
    </div>

    <div class="panel secondary">
      <h3>Point cloud</h3>
      <SourceTabs bind:source drawLabel="New" />
      {#if source === 'examples'}
        <div class="chips">
          {#each EXAMPLES as ex}
            <button class:on={exampleId === ex.id} onclick={() => useExample(ex.id)}>{ex.name}</button>
          {/each}
        </div>
        <p class="note">{EXAMPLES.find((e) => e.id === exampleId)?.about}</p>
      {:else if source === 'random'}
        <label class="field"><span>Distribution</span>
          <select bind:value={randomKind} onchange={() => useRandom()}>
            {#each RANDOM_KINDS as [k, name]}<option value={k}>{name}</option>{/each}
          </select>
        </label>
        <label class="field"><span>Points</span><input type="range" min="5" max="1000" bind:value={randomN} onchange={() => useRandom()} /><span>{randomN}</span></label>
        <label class="field"><span>Noise</span><input type="range" min="0" max="0.4" step="0.01" bind:value={randomNoise} onchange={() => useRandom()} /><span>{randomNoise}</span></label>
        <div class="field"><span>Seed {randomSeedValue}</span><button onclick={() => useRandom(true)}>New sample</button></div>
      {:else if source === 'upload'}
        <FileDrop accept=".csv,.txt,.tsv,.xyz" onfile={upload}>
          A text or CSV file with one point per line: two numbers (a point in the plane) or three (in space), separated by
          commas or spaces. A header line is skipped.
          <pre>x,y
0.0,1.0
0.5,0.87</pre>
        </FileDrop>
        {#if uploadError}<p class="error">{uploadError}</p>{/if}
      {:else}
        <div class="tools">
          <button onclick={blank}>Empty plane</button>
        </div>
        <p class="note">
          Start from nothing and draw a cloud with the editing tools under the view. Any cloud in the plane (an example, a
          random sample or your own) can be edited the same way.
        </p>
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
        bind:selected={runner.selected}
        bind:t={runner.t}
        {describe}
        name={filtration}
      />
    </div>

    <div class="panel secondary">
      <h3>Compute</h3>
      <div class="field"><span>Compute</span>
        <Segmented
          label="What to compute"
          value={kind}
          options={[
            { id: 'filtration', label: 'Persistence', title: 'The persistent homology of a filtration of the cloud' },
            { id: 'zigzag', label: 'Zigzag', title: 'Zigzag persistence: windows of the cloud, or its points moving' },
            { id: 'maps', label: 'Subset maps', title: 'The image, kernel and cokernel of the map from the Rips filtration of the selected points to that of all of them' },
          ]}
          onchange={(k) => setFiltration(k === 'filtration' ? 'rips' : k === 'zigzag' ? 'zigzag' : 'sixpack')}
        />
      </div>
      {#if kind === 'filtration'}
        <div class="field"><span>Filtration</span>
          <Segmented
            label="Filtration"
            value={filtration}
            options={[
              { id: 'rips', label: 'Vietoris–Rips' },
              { id: 'dtm', label: 'DTM-Rips' },
              { id: 'alpha', label: 'Alpha' },
              { id: 'cech', label: 'Delaunay–Čech' },
            ]}
            onchange={setFiltration}
          />
        </div>
      {:else if kind === 'zigzag'}
        <div class="field"><span>Frames</span>
          <Segmented
            label="Frames"
            value={filtration}
            options={[
              { id: 'zigzag', label: 'Sliding windows', title: 'Windows of consecutive points' },
              { id: 'moving', label: 'Moving points', title: "The cloud's points moving, frame by frame" },
            ]}
            onchange={setFiltration}
          />
        </div>
      {/if}
      {#if !isDelaunay}
        <label class="field"><span>Up to dimension</span>
          <select bind:value={maxDim}>
            <option value={0}>H₀</option><option value={1}>H₁</option><option value={2}>H₂</option><option value={3}>H₃</option>
          </select>
        </label>
        {#if isZigzag}
          <label class="field"><span>Complex</span>
            <select bind:value={zzComplex}>
              {#each ZZ_COMPLEXES.filter((c) => filtration === 'zigzag' || c.id !== 'witness') as c}<option value={c.id}>{c.name}</option>{/each}
            </select>
          </label>
        {/if}
        {#if !isZigzag || zzComplex === 'rips'}<label class="field"><span>Metric</span>
          <select bind:value={metric}>
            <option value="euclidean">Euclidean</option>
            <option value="manhattan">Manhattan (L¹)</option>
            <option value="chebyshev">Chebyshev (L∞)</option>
          </select>
        </label>{/if}
        <label class="field"><span>Threshold</span><input type="text" placeholder="none (the enclosing radius)" bind:value={threshold} /></label>
        {#if filtration === 'zigzag'}
          <label class="field"><span>Window (points)</span><input type="range" min="2" max={Math.max(2, cloud.points.length)} bind:value={windowSize} /><span>{windowSize}</span></label>
          <label class="field"><span>Step (points)</span><input type="range" min="1" max={Math.max(1, Math.min(windowSize, cloud.points.length))} bind:value={windowStep} /><span>{windowStep}</span></label>
          <div class="field"><span>Join</span>
            <Segmented label="Join" value={intersection ? 'intersection' : 'union'} options={JOINS} onchange={(j) => (intersection = j === 'intersection')} />
          </div>
          <More>
            {#snippet lead()}A bar is a feature followed from window to window; the slider's window is drawn solid.{/snippet}
            The {ZZ_COMPLEXES.find((c) => c.id === zzComplex)?.name} complexes, at the threshold's scale
            ({ZZ_COMPLEXES.find((c) => c.id === zzComplex)?.scale}), of windows of {windowSize} consecutive points (in the
            cloud's order) starting every {windowStep}, each joined to the next by their
            {intersection ? 'intersection (K₀ ← K₀ ∩ K₁ → K₁ …)' : 'union (K₀ → K₀ ∪ K₁ ← K₁ …)'}: a feature lasts while it
            {intersection ? 'is carried by what one window shares with the next' : 'overlaps itself from window to window'}. By
            FastZigzag (Dey and Hou), as <code>phlite.clouds.cloud_zigzag</code>.
          </More>
        {/if}
        {#if filtration === 'moving'}
          {#if uploadedFrames && uploadedFrames.frames[0] === cloud.points}
            <p class="note">{uploadedFrames.frames.length} uploaded frames of {cloud.points.length} points. <button onclick={() => (uploadedFrames = null)}>Move the cloud instead</button></p>
          {:else}
            <label class="field"><span>Motion</span>
              <select bind:value={motion}>{#each MOTIONS as m}<option value={m.id}>{m.name}</option>{/each}</select>
            </label>
            <label class="field"><span>Frames</span><input type="range" min="2" max="80" bind:value={nMovFrames} /><span>{nMovFrames}</span></label>
            <label class="field"><span>How far</span><input type="range" min="0" max="1" step="0.05" bind:value={amount} /><span>{amount}</span></label>
            {#if motion === 'drift'}<div class="field"><span>Seed {movSeed}</span><button onclick={() => (movSeed = randomSeed())}>New walk</button></div>{/if}
            <p class="note">{MOTIONS.find((m) => m.id === motion)?.about}</p>
          {/if}
          <details class="upload">
            <summary>Upload frames instead</summary>
            <FileDrop accept=".csv,.txt,.tsv" onfile={uploadFrames}>
              One point per line, <code>frame, x, y</code> (or <code>frame, x, y, z</code>), the same number of points in
              every frame, the j-th point of each frame the same point.
            </FileDrop>
          </details>
          {#if framesError}<p class="error">{framesError}</p>{/if}
          <div class="field"><span>Join</span>
            <Segmented label="Join" value={intersection ? 'intersection' : 'union'} options={JOINS} onchange={(j) => (intersection = j === 'intersection')} />
          </div>
          <More>
            {#snippet lead()}A bar is a feature followed from frame to frame; the slider moves through the frames.{/snippet}
            Each frame's {ZZ_COMPLEXES.find((c) => c.id === zzComplex)?.name} complex at the threshold's scale
            ({ZZ_COMPLEXES.find((c) => c.id === zzComplex)?.scale}), on the same points throughout, joined to the next
            frame's by their {intersection ? 'intersection' : 'union'}. As <code>phlite.clouds.cloud_zigzag</code> with a
            frame per time.
          </More>
        {/if}
        {#if filtration === 'sixpack'}
          <More>
            {#snippet lead()}Select points (the subset) to see what of their homology survives among all the points.{/snippet}
            The <em>six-pack</em> of the inclusion of the Rips filtration of the subset into that of all the points: the
            barcodes of the <strong>image</strong> (what of the subset's homology is really there), the
            <strong>kernel</strong> (what of it dies among all the points), the <strong>cokernel</strong> (what the
            subset doesn't see), the subset's and the whole's own barcodes, and the pair's. With live on, the barcodes
            follow the selection. As <code>phlite.clouds.rips_six_pack</code>.
          </More>
        {/if}
        {#if filtration === 'dtm'}
          <label class="field"><span>DTM neighbours k</span><input type="range" min="1" max={Math.max(1, Math.min(50, cloud.points.length))} bind:value={dtmK} /><span>{dtmK}</span></label>
          <More>
            {#snippet lead()}Rips weighted so that outliers come late, and noise matters less.{/snippet}
            Each point is weighted by its distance to measure (the root mean square distance to its k nearest neighbours,
            itself included): a point appears at twice its weight, and an edge when the balls about its ends, of radius
            the scale less the weights, meet.
          </More>
        {/if}
      {:else if filtration === 'alpha'}
        <More>
          {#snippet lead()}The Čech barcode, from the much smaller Delaunay triangulation.{/snippet}
          The exact Delaunay triangulation (in the plane or in space), each cell entering at the radius of its smallest
          empty circumsphere.
        </More>
      {:else}
        <More>
          {#snippet lead()}The Delaunay triangulation, each cell entering when the balls about its vertices first meet.{/snippet}
          Each cell enters at the radius of its minimal enclosing ball (as GUDHI's <code>DelaunayCechComplex</code>), in
          the plane or in space.
        </More>
      {/if}
      {#if !isZigzag}<Options bind:options defaults={isDelaunay ? COMPLEX : RIPS} />{/if}
      <ComputeRow {runner} {job} disabled={blocked} />
      {#if problem}<p class="note" class:error={blocked}>{problem}</p>{/if}
    </div>
  </div>
</div>
