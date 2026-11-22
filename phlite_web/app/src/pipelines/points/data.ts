import type { FiltrationEdges } from '../../lib/types';
import { rng, type Rng } from '../../lib/random';

/** A point cloud in the plane or in space. */
export interface Cloud {
  dim: 2 | 3;
  points: number[][];
}

const TAU = 2 * Math.PI;

function circle(r: Rng, n: number, radius: number, cx: number, cy: number, noise: number): number[][] {
  return Array.from({ length: n }, (_, i) => {
    const a = (TAU * (i + 0.3 * r.next())) / n;
    return [cx + radius * Math.cos(a) + noise * r.normal(), cy + radius * Math.sin(a) + noise * r.normal()];
  });
}

function blobs(r: Rng, n: number, k: number, spread: number): number[][] {
  const centres = Array.from({ length: k }, () => [r.uniform(-1, 1), r.uniform(-1, 1)]);
  return Array.from({ length: n }, (_, i) => {
    const c = centres[i % k];
    return [c[0] + spread * r.normal(), c[1] + spread * r.normal()];
  });
}

function torus(r: Rng, n: number, big: number, small: number, noise: number): number[][] {
  return Array.from({ length: n }, () => {
    const u = TAU * r.next();
    const v = TAU * r.next();
    return [
      (big + small * Math.cos(v)) * Math.cos(u) + noise * r.normal(),
      (big + small * Math.cos(v)) * Math.sin(u) + noise * r.normal(),
      small * Math.sin(v) + noise * r.normal(),
    ];
  });
}

function sphere(r: Rng, n: number, noise: number): number[][] {
  return Array.from({ length: n }, () => {
    const p = [r.normal(), r.normal(), r.normal()];
    const len = Math.hypot(...p) || 1;
    return p.map((x) => x / len + noise * r.normal());
  });
}

export interface PointExample {
  id: string;
  name: string;
  about: string;
  /** The top dimension of homology to start with. */
  maxDim: number;
  make: () => Cloud;
}

export const EXAMPLES: PointExample[] = [
  {
    id: 'circle',
    name: 'Noisy circle',
    about: 'One long H₁ bar for the loop; the noise only makes short bars.',
    maxDim: 1,
    make: () => ({ dim: 2, points: circle(rng(1), 60, 1, 0, 0, 0.06) }),
  },
  {
    id: 'two-circles',
    name: 'Two circles',
    about: 'Two loops of different sizes: two H₁ bars, the bigger loop living longer.',
    maxDim: 1,
    make: () => {
      const r = rng(2);
      return { dim: 2, points: [...circle(r, 50, 1, -1.3, 0, 0.04), ...circle(r, 26, 0.45, 1.1, 0.2, 0.03)] };
    },
  },
  {
    id: 'figure-eight',
    name: 'Figure eight',
    about: 'Two loops sharing a point.',
    maxDim: 1,
    make: () => {
      const r = rng(3);
      const points = Array.from({ length: 90 }, (_, i) => {
        const a = (TAU * i) / 90;
        return [Math.sin(a) + 0.03 * r.normal(), Math.sin(a) * Math.cos(a) + 0.03 * r.normal()];
      });
      return { dim: 2, points };
    },
  },
  {
    id: 'annulus',
    name: 'Annulus',
    about: 'Points filling an annulus: one H₁ bar, born late (the points are dense, the hole is big).',
    maxDim: 1,
    make: () => {
      const r = rng(4);
      return {
        dim: 2,
        points: Array.from({ length: 220 }, () => {
          const a = TAU * r.next();
          const rad = Math.sqrt(r.uniform(0.36, 1));
          return [rad * Math.cos(a), rad * Math.sin(a)];
        }),
      };
    },
  },
  {
    id: 'clusters',
    name: 'Clusters',
    about: 'Four Gaussian clusters: four long H₀ bars (one essential), and little else.',
    maxDim: 1,
    make: () => ({ dim: 2, points: blobs(rng(9), 160, 4, 0.12) }),
  },
  {
    id: 'sphere',
    name: 'Sphere (3D)',
    about: 'Points on a sphere: an H₂ bar for the enclosed void.',
    maxDim: 2,
    make: () => ({ dim: 3, points: sphere(rng(6), 140, 0.02) }),
  },
  {
    id: 'torus',
    name: 'Torus (3D)',
    about: 'Points on a torus: two H₁ bars (the two circles) and an H₂ bar (the void), if the sample is dense enough.',
    maxDim: 2,
    make: () => ({ dim: 3, points: torus(rng(7), 400, 1, 0.45, 0.0) }),
  },
  {
    id: 'trefoil',
    name: 'Trefoil knot (3D)',
    about: 'A knotted loop is still a loop: one H₁ bar.',
    maxDim: 1,
    make: () => {
      const r = rng(8);
      const points = Array.from({ length: 120 }, (_, i) => {
        const a = (TAU * i) / 120;
        return [
          Math.sin(a) + 2 * Math.sin(2 * a) + 0.05 * r.normal(),
          Math.cos(a) - 2 * Math.cos(2 * a) + 0.05 * r.normal(),
          -Math.sin(3 * a) + 0.05 * r.normal(),
        ];
      });
      return { dim: 3, points };
    },
  },
];

export type RandomKind = 'uniform' | 'blobs' | 'circle' | 'cube' | 'sphere';
export const RANDOM_KINDS: [RandomKind, string][] = [
  ['uniform', 'Uniform in a square'],
  ['blobs', 'Gaussian blobs'],
  ['circle', 'Noisy circle'],
  ['cube', 'Uniform in a cube (3D)'],
  ['sphere', 'Noisy sphere (3D)'],
];

export function randomCloud(kind: RandomKind, n: number, noise: number, seed: number): Cloud {
  const r = rng(seed);
  switch (kind) {
    case 'uniform':
      return { dim: 2, points: Array.from({ length: n }, () => [r.uniform(-1, 1), r.uniform(-1, 1)]) };
    case 'blobs':
      return { dim: 2, points: blobs(r, n, 2 + r.int(4), 0.05 + noise) };
    case 'circle':
      return { dim: 2, points: circle(r, n, 1, 0, 0, noise) };
    case 'cube':
      return { dim: 3, points: Array.from({ length: n }, () => [r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)]) };
    case 'sphere':
      return { dim: 3, points: sphere(r, n, noise) };
  }
}

/** Points from CSV or whitespace-separated text: the first two (or three) numeric columns of each line. */
// ---- Moving points ----

export type Motion = 'breathe' | 'split' | 'drift' | 'shear';
export const MOTIONS: { id: Motion; name: string; about: string }[] = [
  { id: 'breathe', name: 'Breathe', about: 'The cloud shrinks towards its centre and grows back: a loop closes up as the points crowd together and opens again.' },
  { id: 'split', name: 'Pull apart', about: 'The left and right halves move apart: components split, and a loop across the middle breaks.' },
  { id: 'drift', name: 'Drift', about: 'Every point wanders at random (a random walk), so small features flicker in and out while big ones last.' },
  { id: 'shear', name: 'Shear', about: 'The cloud is sheared sideways, more and more: distances along the shear stretch, and loops flatten out.' },
];

/** `nFrames` frames of `cloud`'s points moving by `motion`, by `amount` (0 to 1, in units of the cloud's size). */
export function moveCloud(cloud: Cloud, motion: Motion, nFrames: number, amount: number, seed: number): number[][][] {
  const pts = cloud.points;
  if (pts.length === 0) return [];
  const d = cloud.dim;
  const centre = Array.from({ length: d }, (_, k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const size = Math.max(...Array.from({ length: d }, (_, k) => Math.max(...pts.map((p) => p[k])) - Math.min(...pts.map((p) => p[k])))) || 1;
  const frames: number[][][] = [];
  if (motion === 'drift') {
    const r = rng(seed);
    const step = (amount * size) / (4 * Math.sqrt(nFrames));
    let current = pts.map((p) => [...p]);
    for (let f = 0; f < nFrames; f++) {
      frames.push(current);
      current = current.map((p) => p.map((x) => x + step * r.normal()));
    }
    return frames;
  }
  for (let f = 0; f < nFrames; f++) {
    const s = nFrames === 1 ? 0 : f / (nFrames - 1);
    frames.push(
      pts.map((p) => {
        if (motion === 'breathe') {
          const k = 1 - amount * 0.8 * Math.sin(Math.PI * s);
          return p.map((x, i) => centre[i] + k * (x - centre[i]));
        }
        if (motion === 'split') {
          const side = p[0] < centre[0] ? -1 : 1;
          return p.map((x, i) => (i === 0 ? x + side * amount * size * 0.6 * s : x));
        }
        // shear
        return p.map((x, i) => (i === 0 ? x + amount * 1.5 * s * (p[1] - centre[1]) : x));
      }),
    );
  }
  return frames;
}

/**
 * Frames of moving points from text: one point per line, `frame, x, y[, z]` (frames numbered from anything, in any
 * order); every frame must have the same number of points, the j-th point of each the same point.
 */
export function parseFrames(text: string, maxPoints = 20000): { dim: 2 | 3; frames: number[][][] } {
  const byFrame = new Map<number, number[][]>();
  let total = 0;
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const cols = trimmed.split(/[\s,;]+/).map(Number);
    if (cols.length < 3 || cols.some((c) => !Number.isFinite(c))) {
      if (total === 0) continue; // a header
      throw new Error(`Line "${trimmed.slice(0, 40)}" isn't a row of numbers: frame, x, y (and z)`);
    }
    const [frame, ...xyz] = cols;
    if (!byFrame.has(frame)) byFrame.set(frame, []);
    byFrame.get(frame)!.push(xyz);
    if (++total > maxPoints) throw new Error(`More than ${maxPoints} points in all is too many for the playground`);
  }
  if (total === 0) throw new Error('No points found: expected frame, x, y (and z) per line');
  const frames = [...byFrame.entries()].sort((a, b) => a[0] - b[0]).map(([, f]) => f);
  if (frames.some((f) => f.length !== frames[0].length)) throw new Error('Every frame needs the same number of points');
  const dim = frames.every((f) => f.every((p) => p.length >= 3)) ? 3 : 2;
  return { dim, frames: frames.map((f) => f.map((p) => p.slice(0, dim))) };
}

export function parseCloud(text: string, maxPoints = 5000): Cloud {
  const rows: number[][] = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const cols = trimmed.split(/[\s,;]+/).map(Number);
    if (cols.length < 2 || cols.some((c) => !Number.isFinite(c))) {
      // A header line
      if (rows.length === 0) continue;
      throw new Error(`Line "${trimmed.slice(0, 40)}" isn't a row of numbers`);
    }
    rows.push(cols);
  }
  if (rows.length === 0) throw new Error('No points found: expected two or three numbers per line');
  if (rows.length > maxPoints) throw new Error(`${rows.length} points is too many for the playground (at most ${maxPoints})`);
  const dim = rows.every((r) => r.length >= 3) ? 3 : 2;
  return { dim, points: rows.map((r) => r.slice(0, dim)) };
}


/**
 * The simplicial complex of a flag filtration at `t`, from its edges (as phlite computed them): how many of the sorted
 * `edges` are in, and its triangles (every three points pairwise joined), as `[a, b, c]` triples, at most `limit` of
 * them (`truncated` if there were more).
 */
export function activeComplex(edges: FiltrationEdges, n: number, t: number, limit = 20000) {
  const m = countUpTo(edges.values, t);
  const adjacent: number[][] = Array.from({ length: n }, () => []);
  for (let k = 0; k < m; k++) {
    const a = edges.pairs[2 * k];
    const b = edges.pairs[2 * k + 1];
    adjacent[a].push(b);
    adjacent[b].push(a);
  }
  // A triangle once, from its smallest vertex: c > b > a, both b and c neighbours of a, and c of b
  const triangles: number[] = [];
  let truncated = false;
  const mark = new Int32Array(n).fill(-1);
  outer: for (let a = 0; a < n; a++) {
    for (const b of adjacent[a]) if (b > a) mark[b] = a;
    for (const b of adjacent[a]) {
      if (b <= a) continue;
      for (const c of adjacent[b]) {
        if (c > b && mark[c] === a) {
          if (triangles.length >= 3 * limit) {
            truncated = true;
            break outer;
          }
          triangles.push(a, b, c);
        }
      }
    }
  }
  return { m, triangles: Uint32Array.from(triangles), truncated };
}

/** How many entries of the sorted `lengths` are at most `t`. */
export function countUpTo(lengths: Float64Array, t: number): number {
  let lo = 0;
  let hi = lengths.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (lengths[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function flatten(cloud: Cloud): Float64Array {
  const out = new Float64Array(cloud.points.length * cloud.dim);
  cloud.points.forEach((p, i) => p.forEach((x, k) => (out[i * cloud.dim + k] = x)));
  return out;
}
