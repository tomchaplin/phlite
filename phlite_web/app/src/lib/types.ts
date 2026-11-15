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
}

/** The complexes of `phlite_web::network`. */
export type NetworkKind = 'pph' | 'flag' | 'flag-sp' | 'grounded-flag' | 'tuples' | 'dowker-sink' | 'dowker-source';

export type Job =
  | {
      pipeline: 'rips';
      coords: Float64Array;
      pointDim: number;
      maxDim: number;
      threshold: number | null;
      /** DTM-Rips: weight each point by its distance to measure over this many nearest neighbours. */
      dtmK: number | null;
      options: AlgoOptions;
    }
  | { pipeline: 'delaunay'; coords: Float64Array; pointDim: number; cech: boolean; options: AlgoOptions }
  | { pipeline: 'grpph'; n: number; sources: Uint32Array; targets: Uint32Array; weights: Float64Array; grounded: boolean; options: AlgoOptions }
  | {
      pipeline: 'network';
      kind: NetworkKind;
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

/** Colours of homology dimensions 0, 1, 2, 3 (Tableau 10, as in the book's figures). */
export const DIM_COLOURS = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#b279a2'];
export const dimColour = (d: number) => DIM_COLOURS[Math.min(d, DIM_COLOURS.length - 1)];
