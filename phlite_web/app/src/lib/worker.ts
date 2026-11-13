// The Web Worker that owns the WebAssembly module. With threads, the module's pool is more Web Workers, started by
// wasm-bindgen-rayon; a computation blocks this worker (never the page) until it is done. Without (the page can't
// share memory, or the threaded module wouldn't start), it loads the build without threads instead.
import type { Job } from './types';

// What both builds export (their glue differs)
type Module = Pick<
  typeof import('phlite-web-single'),
  'rips' | 'alpha' | 'grpph' | 'image' | 'complex' | 'n_threads' | 'set_crash_hooks' | 'crash_for_testing'
>;
type Message = { type: 'init'; threads: number; single: boolean; channel: string } | { type: 'run'; id: number; job: Job };

let phlite: Module;

self.onmessage = async (event: MessageEvent<Message>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    try {
      if (msg.single) {
        const single = await import('phlite-web-single');
        await single.default();
        phlite = single;
      } else {
        const threaded = await import('phlite-web');
        await threaded.default();
        await threaded.initThreadPool(msg.threads);
        phlite = threaded;
      }
      phlite.set_crash_hooks(msg.channel);
      postMessage({ type: 'ready', threads: phlite.n_threads() });
    } catch (e) {
      postMessage({ type: 'init-error', message: String(e) });
    }
    return;
  }
  const { id, job } = msg;
  const start = performance.now();
  try {
    let barcode;
    switch (job.pipeline) {
      case 'rips':
        barcode = phlite.rips(job.coords, job.pointDim, job.maxDim, job.threshold ?? undefined, job.options);
        break;
      case 'alpha':
        barcode = phlite.alpha(job.coords, job.options);
        break;
      case 'grpph':
        barcode = phlite.grpph(job.n, job.sources, job.targets, job.weights, job.options);
        break;
      case 'image':
        barcode = phlite.image(job.pixels, job.width, job.height, job.superlevel, job.options);
        break;
      case 'complex':
        barcode = phlite.complex(
          job.dimensions, job.indptr, job.indices, job.coefficients, job.values, job.modulus,
          job.maxDim ?? undefined, job.options,
        );
        break;
      case 'crash':
        phlite.crash_for_testing(job.how);
        break;
    }
    postMessage({ type: 'result', id, barcode, ms: performance.now() - start });
  } catch (e) {
    // A panic or a failed allocation traps, which leaves the module unusable: the page restarts the worker
    const fatal = e instanceof WebAssembly.RuntimeError;
    postMessage({ type: 'error', id, message: e instanceof Error ? e.message : String(e), fatal });
  }
};
