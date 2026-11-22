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
  /** What to compute, if the example is about a particular complex: chosen with it. */
  show?: { complex: 'pph' | 'dowker-sink' | 'mpss' | 'zz-clique' | 'zz-flag'; shortestPath: boolean; maxDim: number; lMax?: number };
}

export const EXAMPLES: NetworkExample[] = [
  {
    id: 'contacts',
    name: 'Contacts over time',
    about:
      "A temporal network: each edge's weight is the time of a contact. A square a→b→c→d→a (times 0–3) gains its diagonals (4, 5), then a triangle e→f→g→e (6–8) and a transitive one e→g, g→h, e→h (9–11). Shown as a zigzag of windows of time: the square's loop lives while a window holds it, until the diagonals fill it; the cyclic triangle is a loop of the directed flag complex but filled in the clique complex.",
    make: () => {
      const labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
      const at: [number, number][] = [
        [0.15, 0.2], [0.4, 0.2], [0.4, 0.5], [0.15, 0.5], [0.6, 0.6], [0.85, 0.5], [0.8, 0.85], [0.55, 0.9],
      ];
      const e = (a: string, b: string, w: number) => ({ s: labels.indexOf(a), t: labels.indexOf(b), w });
      return {
        nodes: labels.map((label, i) => ({ x: at[i][0], y: at[i][1], label })),
        edges: [
          e('a', 'b', 0), e('b', 'c', 1), e('c', 'd', 2), e('d', 'a', 3),
          e('a', 'c', 4), e('b', 'd', 5),
          e('e', 'f', 6), e('f', 'g', 7), e('g', 'e', 8),
          e('e', 'g', 9), e('g', 'h', 10), e('e', 'h', 11),
        ],
      };
    },
    show: { complex: 'zz-flag', shortestPath: false, maxDim: 1 },
  },
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
    id: 'mpss-cycle',
    name: 'Directed 4-cycle (MPSS)',
    about:
      "Hepworth and Roff's example: in the magnitude-path spectral sequence of a directed m-cycle, E¹ has rank m in bidegrees (2i, mi) and (2i+1, mi+1), E² = … = E^(m−1), and E^m is a single generator at (0, 0). Step through the pages and look at the generators.",
    make: () => cycle(4),
    show: { complex: 'mpss', shortestPath: false, maxDim: 1, lMax: 6 },
  },
  {
    id: 'mpss-square',
    name: 'Directed square (MPSS)',
    about:
      'Two paths a → b → d and a → c → d: the square is filled in path homology (E² on the diagonal k = l is path homology), but magnitude homology (E¹) sees the two geodesics. Click entries for generators, and tuples to see them on the graph.',
    make: () => ({
      nodes: [{ x: 0.2, y: 0.5, label: 'a' }, { x: 0.5, y: 0.2, label: 'b' }, { x: 0.5, y: 0.8, label: 'c' }, { x: 0.8, y: 0.5, label: 'd' }],
      edges: [{ s: 0, t: 1, w: 1 }, { s: 0, t: 2, w: 1 }, { s: 1, t: 3, w: 1 }, { s: 2, t: 3, w: 1 }],
    }),
    show: { complex: 'mpss', shortestPath: false, maxDim: 1, lMax: 4 },
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
    id: 'sphere',
    name: 'Path-homology sphere',
    about:
      'A directed 4-cycle (weight 1), coned off from s (weight 2) and then from t (weight 3): in PPH the loop is born at 1 and filled by the first cone at 2, and the two cones make a sphere, an H₂ bar born at 3 (its representative is the eight triangles), filled at 4 by the edge s→t. The Dowker complexes only see the loop.',
    make: () => {
      const at = [[0.2, 0.5], [0.5, 0.36], [0.8, 0.5], [0.5, 0.64], [0.5, 0.06], [0.5, 0.94]];
      const nodes = at.map(([x, y], i) => ({ x, y, label: i < 4 ? `a${i}` : i === 4 ? 's' : 't' }));
      const edges = [
        ...[0, 1, 2, 3].map((i) => ({ s: i, t: (i + 1) % 4, w: 1 })),
        ...[0, 1, 2, 3].map((i) => ({ s: 4, t: i, w: 2 })),
        ...[0, 1, 2, 3].map((i) => ({ s: 5, t: i, w: 3 })),
        { s: 4, t: 5, w: 4 },
      ];
      return { nodes, edges };
    },
    show: { complex: 'pph', shortestPath: false, maxDim: 2 },
  },
  {
    id: 'dowker-cycle',
    name: 'Cycle network (Dowker)',
    about:
      "A directed 8-cycle of unit edges, each pair at its shortest path's length (a cycle network, as Chowdhury and Mémoli study): its Dowker complex has an H₁ bar [1, 4), three H₂ bars [4, 5) and an H₃ bar [5, 6), as the balls about the nodes grow round the cycle. The sink and source complexes agree (Dowker duality).",
    make: () => cycle(8),
    show: { complex: 'dowker-sink', shortestPath: true, maxDim: 3 },
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
