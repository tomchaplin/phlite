// The page's handle on the worker: start it with some number of threads, run jobs one at a time, and cancel a job by
// throwing the worker away (a running WebAssembly call can't be interrupted) and starting a new one.
//
// Threads need shared memory, which browsers only allow on cross-origin isolated pages. Without it, or if the threaded
// module fails to start (say a phone refusing its memory), the worker loads the build without threads instead.
//
// A crash (a panic or running out of memory) kills the module: the worker hears about it from the module itself, on
// a BroadcastChannel (which works even when the crash is on a thread of the pool, while the worker sits waiting for
// that thread forever), and starts a fresh worker.
import type { Job, RunResult } from './types';

export class Cancelled extends Error {
  constructor() {
    super('Cancelled');
  }
}

/** phlite crashed: the job failed and the worker was restarted. */
export class Crashed extends Error {}

type CrashReport = { kind: 'panic'; message: string } | { kind: 'oom'; bytes: number };

function describe(report: CrashReport | string): string {
  if (typeof report === 'string') return `phlite crashed (${report}). It has been restarted.`;
  if (report.kind === 'oom') {
    const mb = report.bytes / 2 ** 20;
    const size = mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.max(1, Math.round(mb))} MB`;
    return (
      `phlite ran out of memory (it asked for ${size} more, and a page gets at most 4 GB, often less). ` +
      'Try less data, a lower dimension or a threshold. It has been restarted.'
    );
  }
  return (
    `phlite crashed, which is a bug: please report it at https://github.com/tomchaplin/phlite/issues. ` +
    `It has been restarted. ${report.message}`
  );
}

const randomName = () => `phlite-crash-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;

class Engine {
  status = $state<'loading' | 'ready' | 'busy' | 'error'>('loading');
  error = $state<string | null>(null);
  /** Threads in the pool once started (1 without threads). */
  threads = $state(0);
  /** Threads asked for. */
  requested = $state(Math.min(navigator.hardwareConcurrency || 4, 16));
  readonly maxThreads = navigator.hardwareConcurrency || 4;
  /** Whether the module has threads, and if not, why not. */
  single = $state(!self.crossOriginIsolated);
  singleReason = $state<string | null>(
    self.crossOriginIsolated
      ? null
      : "this page isn't cross-origin isolated (the COOP and COEP headers), so it can't share memory between threads",
  );

  #worker: Worker | null = null;
  #channel: BroadcastChannel | null = null;
  #ready: Promise<void> = Promise.resolve();
  #pending: { id: number; resolve: (r: RunResult) => void; reject: (e: Error) => void } | null = null;
  #nextId = 0;

  start(threads = this.requested) {
    this.#worker?.terminate();
    this.#channel?.close();
    this.#rejectPending(new Cancelled());
    this.requested = threads;
    this.status = 'loading';
    this.error = null;
    const single = this.single;
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    const channel = new BroadcastChannel(randomName());
    this.#worker = worker;
    this.#channel = channel;
    channel.onmessage = (event: MessageEvent<CrashReport>) => this.#crashed(worker, event.data);
    this.#ready = new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        if (worker !== this.#worker) return;
        const msg = event.data;
        if (msg.type === 'ready') {
          this.threads = msg.threads;
          this.status = 'ready';
          resolve();
        } else if (msg.type === 'init-error' && !single) {
          // Try again without threads
          this.single = true;
          this.singleReason = `the module with threads failed to start (${msg.message})`;
          this.start(threads);
          this.#ready.then(resolve, reject);
        } else if (msg.type === 'init-error') {
          this.status = 'error';
          this.error = `Could not start phlite: ${msg.message}`;
          reject(new Error(this.error));
        } else if (this.#pending && msg.id === this.#pending.id) {
          if (msg.type === 'result') {
            this.#settle().resolve({ barcode: msg.barcode, ms: msg.ms });
          } else if (!msg.fatal) {
            this.#settle().reject(new Error(msg.message));
          } else {
            // The module trapped: its crash report (which says why) is on its way, unless it couldn't send one
            setTimeout(() => this.#crashed(worker, msg.message), 300);
          }
        }
      };
      worker.onerror = (event) => {
        if (worker !== this.#worker) return;
        if (this.#pending) return this.#crashed(worker, event.message || 'the worker failed');
        this.status = 'error';
        this.error = `The worker failed: ${event.message}`;
        reject(new Error(this.error));
      };
    });
    this.#ready.catch(() => {});
    worker.postMessage({ type: 'init', threads, single, channel: channel.name });
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

  /** `worker` crashed: fail its job and start again (once, however many reports arrive). */
  #crashed(worker: Worker, report: CrashReport | string) {
    if (worker !== this.#worker) return;
    const pending = this.#pending;
    this.#pending = null;
    this.start(this.requested);
    pending?.reject(new Crashed(describe(report)));
  }

  #settle() {
    const pending = this.#pending!;
    this.#pending = null;
    this.status = 'ready';
    return pending;
  }

  #rejectPending(error: Error) {
    const pending = this.#pending;
    this.#pending = null;
    pending?.reject(error);
  }
}

export const engine = new Engine();
