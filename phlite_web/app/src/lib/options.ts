import type { AlgoOptions } from './types';
import { engine } from './engine.svelte';
import { prefs } from './prefs.svelte';

/**
 * A pipeline's defaults, mirroring its Python package's presets (and `phlite_web`'s `Defaults`): those of
 * `Algorithm.serial()`, or of `Algorithm.parallel()` if the algorithm is parallel clearing.
 */
export interface PipelineDefaults {
  columns: NonNullable<AlgoOptions['columns']>;
  cache: boolean;
  storeR: boolean;
  /** Whether pivot columns are possible (the matrices know the rank of every row). */
  pivot: boolean;
}

export const RIPS: PipelineDefaults = { columns: 'heap', cache: false, storeR: false, pivot: false };
export const GRPPH: PipelineDefaults = { columns: 'merged', cache: true, storeR: false, pivot: false };
export const COMPLEX: PipelineDefaults = { columns: 'pivot', cache: false, storeR: true, pivot: true };

/** The default algorithm: serial clearing, or parallel clearing if chosen (and there are threads). */
export function defaultAlgorithm(): AlgoOptions['algorithm'] {
  return prefs.parallel && !engine.single ? 'parallel' : 'clearing';
}

/** Every option, set to the defaults for `algorithm`. */
export function defaultsFor(d: PipelineDefaults, algorithm = defaultAlgorithm()): Required<AlgoOptions> {
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

/** Whether `options` are the pipeline's defaults (for the "defaults" badge). */
export function isDefault(d: PipelineDefaults, o: Required<AlgoOptions>): boolean {
  const def = defaultsFor(d);
  return (Object.keys(def) as (keyof AlgoOptions)[]).every((k) => def[k] === o[k]);
}

/** Whether `options` are the preset for `algorithm`, apart from the method and representatives. */
export function isPreset(d: PipelineDefaults, o: Required<AlgoOptions>, algorithm: AlgoOptions['algorithm']): boolean {
  const def = defaultsFor(d, algorithm);
  return (Object.keys(def) as (keyof AlgoOptions)[]).every((k) => k === 'method' || k === 'representatives' || def[k] === o[k]);
}
