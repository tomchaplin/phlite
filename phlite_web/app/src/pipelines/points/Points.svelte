<script lang="ts">
  import { untrack } from 'svelte';
  import SourceTabs, { type Source } from '../../components/SourceTabs.svelte';
  import FileDrop from '../../components/FileDrop.svelte';
  import Options from '../../components/Options.svelte';
  import Results from '../../components/Results.svelte';
  import FiltrationSlider from '../../components/FiltrationSlider.svelte';
  import View2D, { type Tool } from './View2D.svelte';
  import View3D from './View3D.svelte';
  import { Runner } from '../../lib/runner.svelte';
  import { RIPS, COMPLEX, defaultsFor } from '../../lib/options';
  import { barDomain } from '../../lib/scale';
  import { randomSeed } from '../../lib/random';
  import type { Job } from '../../lib/types';
  import { EXAMPLES, RANDOM_KINDS, randomCloud, parseCloud, sortedEdges, flatten, type Cloud, type RandomKind } from './data';

  let source = $state<Source>('examples');
  let exampleId = $state(EXAMPLES[0].id);
  let cloud = $state.raw<Cloud>(EXAMPLES[0].make());
  type Filtration = 'rips' | 'dtm' | 'alpha' | 'cech';
  let filtration = $state<Filtration>('rips');
  let dtmK = $state(10);
  const isDelaunay = $derived(filtration === 'alpha' || filtration === 'cech');
  let maxDim = $state(EXAMPLES[0].maxDim);
  let threshold = $state('');
  let options = $state(defaultsFor(RIPS));
  let showBalls = $state(false);
  const runner = new Runner();

  // ---- Data sources ----
  function useExample(id: string) {
    const ex = EXAMPLES.find((e) => e.id === id)!;
    exampleId = id;
    cloud = ex.make();
    maxDim = ex.maxDim;
  }

  let randomKind = $state<RandomKind>('uniform');
  let randomN = $state(80);
  let randomNoise = $state(0.08);
  let randomSeedValue = $state(1);
  function useRandom(newSeed = false) {
    if (newSeed) randomSeedValue = randomSeed();
    cloud = randomCloud(randomKind, randomN, randomNoise, randomSeedValue);
  }

  let uploadError = $state<string | null>(null);
  async function upload(file: File) {
    try {
      cloud = parseCloud(await file.text());
      uploadError = null;
    } catch (e) {
      uploadError = (e as Error).message;
    }
  }

  let tool = $state<Tool>('spray');
  let brush = $state(0.08);
  let drawBox = $state<number[] | null>(null);
  function startDrawing() {
    if (cloud.dim === 3) cloud = { dim: 2, points: [] };
    // Fix the view while drawing, so it doesn't move under the brush
    const ps = cloud.points;
    if (ps.length < 2) drawBox = [-1.25, -1, 1.25, 1];
    else {
      const xs = ps.map((p) => p[0]);
      const ys = ps.map((p) => p[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const pad = Math.max(x1 - x0, y1 - y0) * 0.15;
      drawBox = [x0 - pad, y0 - pad, x1 + pad, y1 + pad];
    }
    const span = Math.max(drawBox[2] - drawBox[0], drawBox[3] - drawBox[1]);
    brush = span * 0.035;
  }
  $effect(() => {
    if (source === 'draw') untrack(startDrawing);
    else drawBox = null;
  });

  function setFiltration(f: Filtration) {
    filtration = f;
    options = defaultsFor(f === 'rips' || f === 'dtm' ? RIPS : COMPLEX);
    showBalls = f === 'alpha' || f === 'cech';
  }

  // ---- Computing ----
  const thresholdValue = $derived(threshold.trim() === '' ? null : Number(threshold));
  const problem = $derived.by(() => {
    if (cloud.points.length === 0) return 'No points yet.';
    if (isDelaunay && cloud.dim !== 2 && cloud.dim !== 3) return 'Alpha and Delaunay–Čech persistence are for points in the plane or in space.';
    if (thresholdValue !== null && !(thresholdValue > 0)) return 'The threshold must be a positive number.';
    if (filtration === 'dtm' && !(dtmK >= 1 && dtmK <= cloud.points.length)) return `k must be between 1 and the number of points (${cloud.points.length}).`;
    if (options.algorithm === 'standard' && !isDelaunay && cloud.points.length > 120)
      return 'The standard algorithm builds every simplex, which takes a very long time for more than about 120 points. Compute anyway?';
    return null;
  });
  const job = $derived.by((): Job | null => {
    if (cloud.points.length === 0) return null;
    const opts = $state.snapshot(options);
    if (isDelaunay) return { pipeline: 'delaunay', coords: flatten(cloud), pointDim: cloud.dim, cech: filtration === 'cech', options: opts };
    return {
      pipeline: 'rips',
      coords: flatten(cloud),
      pointDim: cloud.dim,
      maxDim,
      threshold: thresholdValue,
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
  const edges = $derived(filtration === 'rips' && cloud.points.length <= 1200 ? sortedEdges(cloud.points) : null);
  const domain = $derived(barDomain(runner.result?.barcode.bars ?? [], [0]));
  const bar = $derived(runner.selected === null ? null : (runner.result?.barcode.bars[runner.selected] ?? null));
  const describe = (cell: number[]) => (cell.length === 1 ? `point ${cell[0]}` : `[${cell.join(' ')}]`);
</script>

<div class="layout">
  <div>
    <div class="panel">
      {#if cloud.dim === 2}
        <View2D
          points={cloud.points}
          t={runner.t}
          filtration={isDelaunay ? 'alpha' : 'rips'}
          {bar}
          balls={showBalls && filtration !== 'dtm'}
          {edges}
          tool={source === 'draw' ? tool : null}
          {brush}
          box={source === 'draw' ? drawBox : null}
          onedit={(points) => (cloud = { dim: 2, points })}
        />
      {:else}
        <View3D points={cloud.points} t={runner.t} {bar} edges={filtration === 'rips' ? edges : null} />
      {/if}
      <div style="margin-top: 0.6rem">
        <FiltrationSlider bind:t={runner.t} {domain} label={isDelaunay ? 'Radius' : 'Diameter'} />
      </div>
      <div class="legend" style="margin-top: 0.4rem">
        <span>{cloud.points.length} points{cloud.dim === 3 ? ' in space' : ''}</span>
        {#if cloud.dim === 2 && filtration !== 'dtm'}<label><input type="checkbox" bind:checked={showBalls} /> balls of radius {filtration === 'rips' ? 'd/2' : 'r'}</label>{/if}
        {#if filtration === 'rips'}<span>edges: pairs within the diameter</span>{/if}
        {#if bar}<span>coloured: the representative · dashed: the simplex that kills it</span>{/if}
      </div>
    </div>

    <div class="panel secondary">
      <h3>Point cloud</h3>
      <SourceTabs bind:source drawLabel="Brush" />
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
          <button class:on={tool === 'spray'} onclick={() => (tool = 'spray')}>Spray</button>
          <button class:on={tool === 'point'} onclick={() => (tool = 'point')}>Single points</button>
          <button class:on={tool === 'erase'} onclick={() => (tool = 'erase')}>Erase</button>
          <button onclick={() => (cloud = { dim: 2, points: [] })}>Clear</button>
        </div>
        {#if tool !== 'point' && drawBox}
          <label class="field"><span>Brush size</span><input type="range" min={(drawBox[2] - drawBox[0]) * 0.005} max={(drawBox[2] - drawBox[0]) * 0.12} step="any" bind:value={brush} /></label>
        {/if}
        <p class="note">Draw on the canvas below. The barcode updates as you go.</p>
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
      <div class="field"><span>Filtration</span>
        <div class="tools">
          <button class:on={filtration === 'rips'} onclick={() => setFiltration('rips')}>Vietoris–Rips</button>
          <button class:on={filtration === 'dtm'} onclick={() => setFiltration('dtm')}>DTM-Rips</button>
          <button class:on={filtration === 'alpha'} onclick={() => setFiltration('alpha')}>Alpha</button>
          <button class:on={filtration === 'cech'} onclick={() => setFiltration('cech')}>Delaunay–Čech</button>
        </div>
      </div>
      {#if !isDelaunay}
        <label class="field"><span>Up to dimension</span>
          <select bind:value={maxDim}>
            <option value={0}>H₀</option><option value={1}>H₁</option><option value={2}>H₂</option><option value={3}>H₃</option>
          </select>
        </label>
        <label class="field"><span>Threshold</span><input type="text" placeholder="none (the enclosing radius)" bind:value={threshold} /></label>
        {#if filtration === 'dtm'}
          <label class="field"><span>DTM neighbours k</span><input type="range" min="1" max={Math.max(1, Math.min(50, cloud.points.length))} bind:value={dtmK} /><span>{dtmK}</span></label>
          <p class="note">
            Weighted Rips, each point weighted by its distance to measure (the root mean square distance to its k nearest
            neighbours, itself included): a point appears at twice its weight, and an edge when the balls about its ends,
            of radius the scale less the weights, meet. Outliers come late, so noise matters less.
          </p>
        {/if}
      {:else if filtration === 'alpha'}
        <p class="note">The exact Delaunay triangulation (in the plane or in space), each cell entering at the radius of its smallest empty circumsphere: the same barcode as the Čech filtration, from a small complex.</p>
      {:else}
        <p class="note">The exact Delaunay triangulation (in the plane or in space), each cell entering at the radius of its minimal enclosing ball, when the balls about its vertices first meet.</p>
      {/if}
      <Options bind:options defaults={isDelaunay ? COMPLEX : RIPS} />
      <div class="compute-row">
        <button class="primary" disabled={!job || blocked} onclick={() => runner.run(job)}>Compute</button>
        <label><input type="checkbox" bind:checked={runner.auto} /> recompute on every change</label>
      </div>
      {#if problem}<p class="note" class:error={blocked}>{problem}</p>{/if}
    </div>
  </div>
</div>
