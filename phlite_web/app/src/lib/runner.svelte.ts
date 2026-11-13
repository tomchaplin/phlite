// A pipeline's computation state: run a job (the latest one wins), with the result, the selected bar and the
// filtration value being looked at.
import { engine, Cancelled } from './engine.svelte';
import type { Job, RunResult } from './types';

export class Runner {
  result = $state<RunResult | null>(null);
  error = $state<string | null>(null);
  running = $state(false);
  selected = $state<number | null>(null);
  t = $state<number | null>(null);
  /** Recompute whenever the data or options change. */
  auto = $state(true);
  #generation = 0;
  #timer: ReturnType<typeof setTimeout> | undefined;

  async run(job: Job | null) {
    clearTimeout(this.#timer);
    const generation = ++this.#generation;
    if (!job) {
      this.result = null;
      this.error = null;
      this.selected = null;
      return;
    }
    this.running = true;
    this.error = null;
    try {
      const result = await engine.run(job);
      if (generation !== this.#generation) return;
      this.result = result;
      if (this.selected !== null && this.selected >= result.barcode.bars.length) this.selected = null;
      // A new barcode's bars are different bars
      this.selected = null;
    } catch (e) {
      if (generation !== this.#generation) return;
      this.error = e instanceof Cancelled ? 'Cancelled.' : (e as Error).message;
      this.result = null;
    } finally {
      if (generation === this.#generation) this.running = false;
    }
  }

  /** Run `job` after a short pause (so that dragging a slider doesn't start a computation per pixel). */
  schedule(job: Job | null, delay = 200) {
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => this.run(job), delay);
  }
}
