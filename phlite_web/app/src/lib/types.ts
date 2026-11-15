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
}

export interface Barcode {
  bars: Bar[];
  summary: string;
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
      options: AlgoOptions;
    }
  /** Crash on purpose, to test the recovery (`window.__phlite.run({pipeline: 'crash', how: 'pool'})`). */
  | { pipeline: 'crash'; how: 'panic' | 'pool' | 'oom' };

/** The colour of homology dimension `d` (Catppuccin, as the book's figures; see `palette.svelte.ts`). */
export { dimColour } from './palette.svelte';
