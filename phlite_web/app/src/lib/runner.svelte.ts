// A pipeline's computation state: run a job (the latest one wins), with the result, the selected bar and the
// filtration value being looked at.
import { engine, Cancelled } from './engine.svelte';
import type { Job, RunResult } from './types';

export class Runner {
  result = $state.raw<RunResult | null>(null);
  /** The job `result` was computed for (to tell whether it is still the data's). */
  resultJob = $state.raw<Job | null>(null);
  error = $state<string | null>(null);
  running = $state(false);
  selected = $state<number | null>(null);
  t = $state<number | null>(null);
  /** Recompute whenever the data or options change. */
  auto = $state(true);
  /**
   * Recompute live, as the data is edited: rather than waiting for a pause and cancelling a computation still running,
   * let it finish and then compute the latest data.
   */
  live = $state(false);
  #generation = 0;
  #timer: ReturnType<typeof setTimeout> | undefined;
  /** In live mode, the job to run when the one running finishes (`undefined`: none). */
  #queued: Job | null | undefined = undefined;

  async run(job: Job | null) {
    clearTimeout(this.#timer);
    this.#queued = undefined;
    const generation = ++this.#generation;
    if (!job) {
      this.result = null;
      this.resultJob = null;
      this.error = null;
      this.selected = null;
      return;
    }
    this.running = true;
    this.error = null;
    // The selected bar belongs to the old barcode: drawing it on new data would be nonsense
    this.selected = null;
    try {
      const result = await engine.run(job);
      if (generation !== this.#generation) return;
      this.result = result;
      this.resultJob = job;
    } catch (e) {
      if (generation !== this.#generation) return;
      this.error = e instanceof Cancelled ? 'Cancelled.' : (e as Error).message;
      this.result = null;
      this.resultJob = null;
    } finally {
      if (generation === this.#generation) {
        this.running = false;
        const next = this.#queued;
        if (next !== undefined) this.run(next);
      }
    }
  }

  /**
   * Run `job` after a short pause (so that dragging a slider doesn't start a computation per pixel), or, live, as soon
   * as the computation running (if any) is done.
   */
  schedule(job: Job | null, delay = 200) {
    clearTimeout(this.#timer);
    if (this.live) {
      if (this.running) this.#queued = job;
      else this.run(job);
      return;
    }
    this.#timer = setTimeout(() => this.run(job), delay);
  }
}
