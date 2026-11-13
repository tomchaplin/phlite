// The page's handle on the worker: start it with some number of threads, run jobs one at a time, and cancel a job by
// throwing the worker away (a running WebAssembly call can't be interrupted) and starting a new one.
import type { Job, RunResult } from './types';

export class Cancelled extends Error {
  constructor() {
    super('Cancelled');
  }
}

class Engine {
  status = $state<'loading' | 'ready' | 'busy' | 'error'>('loading');
  error = $state<string | null>(null);
  /** Threads in the pool once started. */
  threads = $state(0);
  /** Threads asked for. */
  requested = $state(Math.min(navigator.hardwareConcurrency || 4, 16));
  readonly maxThreads = navigator.hardwareConcurrency || 4;

  #worker: Worker | null = null;
  #ready: Promise<void> = Promise.resolve();
  #pending: { id: number; resolve: (r: RunResult) => void; reject: (e: Error) => void } | null = null;
  #nextId = 0;

  start(threads = this.requested) {
    this.#worker?.terminate();
    this.#rejectPending(new Cancelled());
    this.requested = threads;
    this.status = 'loading';
    this.error = null;
    if (!self.crossOriginIsolated) {
      this.status = 'error';
      this.error =
        'This page is not cross-origin isolated, so it cannot share memory between threads. It must be served with ' +
        'the headers Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp.';
      this.#ready = Promise.reject(new Error(this.error));
      this.#ready.catch(() => {});
      return;
    }
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.#worker = worker;
    this.#ready = new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          this.threads = msg.threads;
          this.status = 'ready';
          resolve();
        } else if (msg.type === 'init-error') {
          this.status = 'error';
          this.error = `Could not start phlite: ${msg.message}`;
          reject(new Error(this.error));
        } else if (this.#pending && msg.id === this.#pending.id) {
          const pending = this.#pending;
          this.#pending = null;
          this.status = 'ready';
          if (msg.type === 'result') pending.resolve({ barcode: msg.barcode, ms: msg.ms });
          else {
            pending.reject(new Error(msg.message));
            if (msg.fatal) this.start(this.requested);
          }
        }
      };
      worker.onerror = (event) => {
        this.status = 'error';
        this.error = `The worker failed: ${event.message}`;
        reject(new Error(this.error));
      };
    });
    this.#ready.catch(() => {});
    worker.postMessage({ type: 'init', threads });
  }

  /** Run `job`, cancelling any job still running. */
  async run(job: Job): Promise<RunResult> {
    if (this.#pending) this.cancel();
    await this.#ready;
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending = { id, resolve, reject };
      this.status = 'busy';
      this.#worker!.postMessage({ type: 'run', id, job });
    });
  }

  cancel() {
    if (this.#pending) this.start(this.requested);
  }

  #rejectPending(error: Error) {
    const pending = this.#pending;
    this.#pending = null;
    pending?.reject(error);
  }
}

export const engine = new Engine();
