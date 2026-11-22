// Smoke tests of the playground's WebAssembly (`just smoke`, after `just wasm` or `just wasm-dev`): each pipeline the
// app calls, on an input whose barcode is known. Run with Node; the threaded build's worker glue needs `self`.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

globalThis.self = { addEventListener() {}, removeEventListener() {} };
const dir = new URL('../pkg/', import.meta.url);
const m = await import(new URL('phlite_web.js', dir));
m.initSync({ module: readFileSync(new URL('phlite_web_bg.wasm', dir)) });

let failures = 0;
function check(name, f) {
  try {
    f();
    console.log(`ok   ${name}`);
  } catch (e) {
    failures++;
    console.log(`FAIL ${name}\n     ${e.message.split('\n').join('\n     ')}`);
  }
}
const bars = (b, dim) => b.bars.filter((x) => x.dim === dim).map((x) => [x.birth, x.death]);
const circle = (n, turns = 1) => {
  const c = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * turns * i) / n;
    [c[2 * i], c[2 * i + 1]] = [Math.cos(t), Math.sin(t)];
  }
  return c;
};
const serial = { algorithm: 'clearing' };

check('rips: a circle has one long loop', () => {
  const b = m.rips(circle(24), 2, 1, undefined, 'euclidean', undefined, serial);
  const loops = bars(b, 1).filter(([s, d]) => (d ?? Infinity) - s > 1);
  assert.equal(loops.length, 1);
});

check('cloud_zigzag: windows going round a circle once each keep the loop', () => {
  // (Not alpha or Delaunay–Čech: a point of one window sits on one of the next, and the union's two loops differ by a
  // 4-cycle that only a small triangle would fill, which the Delaunay triangulation of cocircular points needn't have;
  // they are checked on moving points below)
  for (const complex of ['rips', 'cech']) {
    // 40 points twice round: a window of 20 spans the circle exactly once
    const t = complex === 'rips' ? 0.7 : 0.35;
    const b = m.cloud_zigzag(circle(40, 2), 2, 20, 1, t, 1, 'euclidean', complex, false);
    assert.deepEqual(bars(b, 1), [[0, null]], complex);
    assert.deepEqual(bars(b, 0), [[0, null]], complex);
  }
  // A distance matrix gives Rips too
  const c = circle(40, 2);
  const d = new Float64Array(40 * 40);
  for (let i = 0; i < 40; i++) for (let j = 0; j < 40; j++) d[i * 40 + j] = Math.hypot(c[2 * i] - c[2 * j], c[2 * i + 1] - c[2 * j + 1]);
  assert.deepEqual(bars(m.cloud_zigzag(d, 0, 20, 1, 0.7, 1, 'euclidean', 'rips', true), 1), [[0, null]]);
});

check('moving_zigzag: a circle that stays put keeps its loop; pulled apart it breaks', () => {
  const c = circle(20);
  const still = Float64Array.from([...c, ...c, ...c]);
  for (const complex of ['rips', 'cech', 'alpha', 'delaunay-cech']) {
    assert.deepEqual(bars(m.moving_zigzag(still, 2, 3, 0.5, 1, 'euclidean', complex, false), 1), [[0, null]], complex);
  }
  const apart = Float64Array.from([...c, ...c.map((x, i) => (i % 2 === 0 ? x * 3 : x))]);
  assert.deepEqual(bars(m.moving_zigzag(apart, 2, 2, 0.5, 1, 'euclidean', 'rips', false), 1), [[0, 1]]);
});

check('network_zigzag: a triangle of contacts is a loop while a window holds it', () => {
  const s = Uint32Array.from([0, 1, 2]);
  const t = Uint32Array.from([1, 2, 0]);
  const w = Float64Array.from([0, 1, 2]);
  // Windows of width 3 from 0: only the first holds all three edges, and its union with the second is still the
  // triangle's boundary, so the loop lasts to 1
  const flag = m.network_zigzag(3, s, t, w, true, 3, 1, 1, false);
  assert.deepEqual(bars(flag, 1), [[0, 1]]);
  // Joined by intersections, it is gone at once
  assert.deepEqual(bars(m.network_zigzag(3, s, t, w, true, 3, 1, 1, true), 1), [[0, 0.5]]);
  // A cyclic triangle is filled in the clique complex
  assert.deepEqual(bars(m.network_zigzag(3, s, t, w, false, 3, 1, 1, false), 1), []);
});

check('timed_zigzag: an edge that comes and goes makes a loop each time', () => {
  // Cells: a b c ab bc ac; ac lives [2, 3) and [4, 6)
  const dims = Uint32Array.from([0, 0, 0, 1, 1, 1]);
  const indptr = Uint32Array.from([0, 0, 0, 0, 2, 4, 6]);
  const indices = Uint32Array.from([0, 1, 1, 2, 0, 2]);
  const lifeIndptr = Uint32Array.from([0, 1, 2, 3, 4, 5, 7]);
  const starts = Float64Array.from([0, 0, 0, 0, 1, 2, 4]);
  const ends = Float64Array.from([Infinity, Infinity, Infinity, Infinity, Infinity, 3, 6]);
  const b = m.timed_zigzag(dims, indptr, indices, lifeIndptr, starts, ends, undefined, false);
  assert.deepEqual(bars(b, 1), [[2, 3], [4, 6]]);
  assert.throws(() => m.timed_zigzag(dims, indptr, indices, lifeIndptr, Float64Array.from([0, 0, 5, 0, 1, 2, 4]), ends, undefined, false));
});

check('video_zigzag: a blob drifting across keeps one component', () => {
  const [w, h, f] = [12, 8, 5];
  const px = new Float64Array(w * h * f).fill(1);
  for (let k = 0; k < f; k++) for (let r = 2; r < 5; r++) for (let c = k; c < k + 3; c++) px[k * w * h + r * w + c] = 0;
  for (const intersection of [false, true]) {
    assert.deepEqual(bars(m.video_zigzag(px, w, h, f, 0.5, false, 'T', false, intersection), 0), [[0, null]]);
  }
});

if (failures) {
  console.log(`${failures} failed`);
  process.exit(1);
}
console.log('all passed');
