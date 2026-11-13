import { rng } from '../../lib/random';

/** A weighted digraph, with positions for drawing (in the unit square, roughly). */
export interface Digraph {
  nodes: { x: number; y: number; label: string }[];
  edges: { s: number; t: number; w: number }[];
}

const ring = (n: number, cx = 0.5, cy = 0.5, r = 0.35, start = -Math.PI / 2) =>
  Array.from({ length: n }, (_, i) => ({
    x: cx + r * Math.cos(start + (2 * Math.PI * i) / n),
    y: cy + r * Math.sin(start + (2 * Math.PI * i) / n),
  }));

function cycle(n: number, both = false): Digraph {
  const nodes = ring(n).map((p, i) => ({ ...p, label: String(i) }));
  const edges = nodes.flatMap((_, i) => {
    const e = [{ s: i, t: (i + 1) % n, w: 1 }];
    if (both) e.push({ s: (i + 1) % n, t: i, w: 1 });
    return e;
  });
  return { nodes, edges };
}

export interface NetworkExample {
  id: string;
  name: string;
  about: string;
  make: () => Digraph;
}

export const EXAMPLES: NetworkExample[] = [
  {
    id: 'book',
    name: 'The notebook example',
    about:
      'From the GrPPH notebook: a directed 6-cycle and 4-cycle, two paths from c3 to t2, and a bidirected edge. Four H₁ bars, all born at 0, dying at 0.5, 1, 2 and 3.',
    make: () => {
      const labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 's1', 's2', 's3', 't1', 't2', 't3', 'b'];
      const at: Record<string, [number, number]> = {
        c0: [0.42, 0.18], c1: [0.62, 0.22], c2: [0.7, 0.42], c3: [0.6, 0.6], c4: [0.4, 0.6], c5: [0.3, 0.4],
        s1: [0.3, 0.05], s2: [0.12, 0.12], s3: [0.18, 0.3],
        t1: [0.78, 0.72], t2: [0.66, 0.9], t3: [0.5, 0.8], b: [0.86, 0.95],
      };
      const ix = (l: string) => labels.indexOf(l);
      const e = (a: string, b: string, w = 1) => ({ s: ix(a), t: ix(b), w });
      return {
        nodes: labels.map((label) => ({ x: at[label][0], y: at[label][1], label })),
        edges: [
          ...[0, 1, 2, 3, 4, 5].map((i) => e(`c${i}`, `c${(i + 1) % 6}`)),
          e('c0', 's1'), e('s1', 's2'), e('s2', 's3'), e('s3', 'c0'),
          e('c3', 't1'), e('t1', 't2'), e('c3', 't3'), e('t3', 't2'),
          e('t2', 'b', 0.5), e('b', 't2', 0.5),
        ],
      };
    },
  },
  {
    id: 'cycle',
    name: 'Directed 5-cycle',
    about: 'A directed cycle of unit edges: an H₁ bar born at 0 that lives until shortcut edges (paths of about half its length) appear to fill it in.',
    make: () => cycle(5),
  },
  {
    id: 'bicycle',
    name: 'Bidirected 5-cycle',
    about: 'With edges both ways the loop still appears, but now dies when the first shortcuts appear.',
    make: () => cycle(5, true),
  },
  {
    id: 'squares',
    name: 'Square vs long square',
    about:
      'Left: a directed 4-cycle, which needs new edges to die. Right: two directed paths of length 2 from one node to another (a long square), which GrPPH fills in at once.',
    make: () => {
      const left = ring(4, 0.27, 0.5, 0.2, -Math.PI / 4);
      const right = ring(4, 0.73, 0.5, 0.2, Math.PI);
      const nodes = [...left, ...right].map((p, i) => ({ ...p, label: i < 4 ? `a${i}` : `b${i - 4}` }));
      return {
        nodes,
        edges: [
          { s: 0, t: 1, w: 1 }, { s: 1, t: 2, w: 1 }, { s: 2, t: 3, w: 1 }, { s: 3, t: 0, w: 1 },
          { s: 4, t: 5, w: 1 }, { s: 5, t: 6, w: 1 }, { s: 4, t: 7, w: 1 }, { s: 7, t: 6, w: 1 },
        ],
      };
    },
  },
  {
    id: 'weighted',
    name: 'Weighted cycles',
    about: 'Three directed triangles with different edge weights: heavier loops live longer.',
    make: () => {
      const nodes: Digraph['nodes'] = [];
      const edges: Digraph['edges'] = [];
      [1, 2, 3].forEach((w, k) => {
        const base = nodes.length;
        ring(3, 0.2 + 0.3 * k, 0.5, 0.12).forEach((p, i) => nodes.push({ ...p, label: `${'xyz'[k]}${i}` }));
        for (let i = 0; i < 3; i++) edges.push({ s: base + i, t: base + ((i + 1) % 3), w });
      });
      return { nodes, edges };
    },
  },
];

/** A random digraph: each ordered pair is an edge with probability `p`, with a uniform random weight. */
export function randomDigraph(n: number, p: number, minW: number, maxW: number, seed: number): Digraph {
  const r = rng(seed);
  const nodes = ring(n).map((pos, i) => ({ ...pos, label: String(i) }));
  const edges: Digraph['edges'] = [];
  for (let s = 0; s < n; s++)
    for (let t = 0; t < n; t++)
      if (s !== t && r.next() < p) edges.push({ s, t, w: Math.round(100 * r.uniform(minW, maxW)) / 100 });
  return layout({ nodes, edges }, seed);
}

/** A few hundred steps of a simple force-directed layout, starting from the current positions. */
export function layout(g: Digraph, seed = 1, steps = 300): Digraph {
  const r = rng(seed);
  const n = g.nodes.length;
  if (n < 2) return g;
  const pos = g.nodes.map((v) => [v.x + 0.01 * r.normal(), v.y + 0.01 * r.normal()]);
  const k = 0.6 / Math.sqrt(n);
  for (let step = 0; step < steps; step++) {
    const temp = 0.05 * (1 - step / steps) + 0.002;
    const force = pos.map(() => [0, 0]);
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dx = pos[i][0] - pos[j][0];
        const dy = pos[i][1] - pos[j][1];
        const d2 = Math.max(dx * dx + dy * dy, 1e-6);
        const f = (k * k) / d2;
        force[i][0] += dx * f; force[i][1] += dy * f;
        force[j][0] -= dx * f; force[j][1] -= dy * f;
      }
    for (const { s, t } of g.edges) {
      const dx = pos[s][0] - pos[t][0];
      const dy = pos[s][1] - pos[t][1];
      const d = Math.max(Math.hypot(dx, dy), 1e-6);
      const f = d / k;
      force[s][0] -= dx * f; force[s][1] -= dy * f;
      force[t][0] += dx * f; force[t][1] += dy * f;
    }
    for (let i = 0; i < n; i++) {
      // Pull gently to the centre, so separate components don't drift apart
      force[i][0] += (0.5 - pos[i][0]) * 0.5;
      force[i][1] += (0.5 - pos[i][1]) * 0.5;
      const len = Math.hypot(force[i][0], force[i][1]) || 1;
      pos[i][0] += (force[i][0] / len) * Math.min(len, temp);
      pos[i][1] += (force[i][1] / len) * Math.min(len, temp);
    }
  }
  // Fit into [0.06, 0.94]²
  const xs = pos.map((p) => p[0]);
  const ys = pos.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const span = Math.max(x1 - x0, y1 - y0, 1e-6);
  return {
    nodes: g.nodes.map((v, i) => ({ ...v, x: 0.06 + (0.88 * (pos[i][0] - x0)) / span, y: 0.06 + (0.88 * (pos[i][1] - y0)) / span })),
    edges: g.edges,
  };
}

/** An edge list: one `source,target[,weight]` per line (names or numbers; weight 1 if missing). */
export function parseEdges(text: string): Digraph {
  const names = new Map<string, number>();
  const nodes: Digraph['nodes'] = [];
  const node = (name: string) => {
    if (!names.has(name)) {
      names.set(name, nodes.length);
      nodes.push({ x: 0, y: 0, label: name });
    }
    return names.get(name)!;
  };
  const edges = new Map<string, Digraph['edges'][number]>();
  text.split(/\r?\n/).forEach((line, i) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;
    const cols = trimmed.split(/[\s,;]+/);
    if (cols.length < 2) throw new Error(`Line ${i + 1}: expected "source,target[,weight]"`);
    const w = cols.length > 2 ? Number(cols[2]) : 1;
    if (!Number.isFinite(w)) {
      if (i === 0) return; // a header
      throw new Error(`Line ${i + 1}: the weight "${cols[2]}" isn't a number`);
    }
    if (w < 0) throw new Error(`Line ${i + 1}: weights must be non-negative`);
    if (cols[0] === cols[1]) return; // GrPPH ignores self-loops
    const s = node(cols[0]);
    const t = node(cols[1]);
    edges.set(`${s},${t}`, { s, t, w });
  });
  if (nodes.length === 0) throw new Error('No edges found');
  if (nodes.length > 400) throw new Error(`${nodes.length} nodes is too many for the playground (at most 400)`);
  const g = { nodes, edges: [...edges.values()] };
  // Start on a circle, then relax
  const start = ring(nodes.length);
  g.nodes.forEach((v, i) => Object.assign(v, start[i]));
  return layout(g);
}

/** All-pairs shortest path lengths (Floyd–Warshall), `Infinity` where there is no path. */
export function shortestPaths(g: Digraph): Float64Array {
  const n = g.nodes.length;
  const d = new Float64Array(n * n).fill(Infinity);
  for (let i = 0; i < n; i++) d[i * n + i] = 0;
  for (const { s, t, w } of g.edges) d[s * n + t] = Math.min(d[s * n + t], w);
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++) {
      const dik = d[i * n + k];
      if (dik === Infinity) continue;
      for (let j = 0; j < n; j++) {
        const via = dik + d[k * n + j];
        if (via < d[i * n + j]) d[i * n + j] = via;
      }
    }
  return d;
}
