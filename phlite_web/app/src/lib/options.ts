import type { AlgoOptions } from './types';

/** A pipeline's defaults, mirroring its Python package's presets (and `phlite_web`'s `Defaults`). */
export interface PipelineDefaults {
  algorithm: AlgoOptions['algorithm'];
  columns: NonNullable<AlgoOptions['columns']>;
  cache: boolean;
  storeR: boolean;
  /** Whether pivot columns are possible (the matrices know the rank of every row). */
  pivot: boolean;
}

export const RIPS: PipelineDefaults = { algorithm: 'clearing', columns: 'heap', cache: false, storeR: false, pivot: false };
export const GRPPH: PipelineDefaults = { algorithm: 'clearing', columns: 'merged', cache: true, storeR: false, pivot: false };
export const COMPLEX: PipelineDefaults = { algorithm: 'clearing', columns: 'pivot', cache: false, storeR: true, pivot: true };

/** Every option, set to the defaults for `algorithm`. */
export function defaultsFor(d: PipelineDefaults, algorithm = d.algorithm): Required<AlgoOptions> {
  return {
    algorithm,
    columns: d.columns === 'pivot' && algorithm !== 'clearing' ? 'merged' : d.columns,
    cache: d.cache,
    storeR: d.storeR && algorithm === 'clearing',
    skipZeroApparentPairs: algorithm !== 'standard',
    storage: 'locking',
    method: 'cohomology',
    representatives: true,
  };
}

/** Whether `options` differ from the pipeline's defaults (for the "defaults" badge). */
export function isDefault(d: PipelineDefaults, o: Required<AlgoOptions>): boolean {
  const def = defaultsFor(d);
  return (Object.keys(def) as (keyof AlgoOptions)[]).every((k) => def[k] === o[k]);
}
