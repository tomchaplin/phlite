/** A bar as returned by the WebAssembly module. What a cell (`number[]`) means depends on the pipeline. */
export interface Bar {
  dim: number;
  birth: number;
  /** `null` for an essential bar. */
  death: number | null;
  birth_cell: number[];
  death_cell: number[] | null;
  /** A representative cycle as `[coefficient, cell]` pairs, if representatives were computed. */
  rep: [number, number[]][] | null;
  /** For extended persistence, which kind of bar (its death may then be below its birth). */
  kind?: ExtendedKind;
}

export type ExtendedKind = 'ordinary' | 'relative' | 'extended+' | 'extended-';

export interface Barcode {
  bars: Bar[];
  summary: string;
  /** The filtration's edges, if the pipeline draws its complex (see `FiltrationEdges`). */
  edges?: FiltrationEdges | null;
}

/**
 * The edges of the filtration a barcode was computed from, by phlite, sorted by when they enter: `pairs[2k] →
 * pairs[2k + 1]` at `values[k]`, with each point's birth if not every point is born at 0 (weighted Rips).
 */
export interface FiltrationEdges {
  pairs: Uint32Array;
  values: Float64Array;
  births: Float64Array | null;
}

export interface RunResult {
  barcode: Barcode;
  /** Time spent in the WebAssembly call, in milliseconds. */
  ms: number;
}

/** How to compute, as understood by the WebAssembly module (missing fields take the pipeline's defaults). */
export interface AlgoOptions {
  algorithm: 'standard' | 'clearing' | 'parallel';
  columns?: 'merged' | 'heap' | 'sorted' | 'pivot';
  cache?: boolean;
  storeR?: boolean;
  skipZeroApparentPairs?: boolean;
  storage?: 'locking' | 'lock-free';
  method: 'cohomology' | 'homology';
  representatives: boolean;
  /** Find dimension 0 by union-find (Rips and images). */
  unionFind?: boolean;
  /** Build the matrices in parallel (by default, exactly when the algorithm is parallel clearing). */
  parallelSetup?: boolean;
}

export type Metric = 'euclidean' | 'manhattan' | 'chebyshev';

/** The complexes of `phlite_web::network`. */
/** The complexes of `phlite_web::graph` (the flag complex of an undirected graph). */
export type GraphKind = 'flag' | 'flag-sp';

export type NetworkKind = 'pph' | 'flag' | 'grounded-flag' | 'tuples' | 'dowker-sink' | 'dowker-source';

export type Job =
  | {
      pipeline: 'rips';
      coords: Float64Array;
      pointDim: number;
      maxDim: number;
      threshold: number | null;
      metric: Metric;
      /** DTM-Rips: weight each point by its distance to measure over this many nearest neighbours. */
      dtmK: number | null;
      options: AlgoOptions;
    }
  | { pipeline: 'delaunay'; coords: Float64Array; pointDim: number; cech: boolean; options: AlgoOptions }
  | { pipeline: 'grpph'; n: number; sources: Uint32Array; targets: Uint32Array; weights: Float64Array; grounded: boolean; options: AlgoOptions }
  | {
      pipeline: 'network';
      kind: NetworkKind;
      /** Each pair at the length of the shortest path between them, rather than each edge at its weight. */
      shortestPath: boolean;
      n: number;
      sources: Uint32Array;
      targets: Uint32Array;
      weights: Float64Array;
      maxDim: number;
      options: AlgoOptions;
    }
  | {
      pipeline: 'graph';
      kind: GraphKind;
      n: number;
      sources: Uint32Array;
      targets: Uint32Array;
      weights: Float64Array;
      maxDim: number;
      options: AlgoOptions;
    }
  | {
      pipeline: 'image';
      pixels: Float64Array;
      width: number;
      height: number;
      superlevel: boolean;
      construction: 'T' | 'V';
      options: AlgoOptions;
    }
  | {
      pipeline: 'complex';
      dimensions: Uint32Array;
      indptr: Uint32Array;
      indices: Uint32Array;
      coefficients: Int32Array;
      values: Float64Array;
      modulus: number;
      maxDim: number | null;
      /** Extended persistence instead (bars with a `kind`). */
      extended: boolean;
      options: AlgoOptions;
    }
  /** Crash on purpose, to test the recovery (`window.__phlite.run({pipeline: 'crash', how: 'pool'})`). */
  | { pipeline: 'crash'; how: 'panic' | 'pool' | 'oom' };

/** The colour of homology dimension `d` (Catppuccin, as the book's figures; see `palette.svelte.ts`). */
export { dimColour } from './palette.svelte';
