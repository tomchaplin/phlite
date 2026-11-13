// The Web Worker that owns the WebAssembly module and its thread pool. The pool's threads are more Web Workers,
// started by wasm-bindgen-rayon; a computation blocks this worker (never the page) until it is done.
import init, { initThreadPool, set_panic_hook, n_threads, rips, alpha, grpph, image, complex } from 'phlite-web';
import type { Job } from './types';

type Message = { type: 'init'; threads: number } | { type: 'run'; id: number; job: Job };

self.onmessage = async (event: MessageEvent<Message>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    try {
      await init();
      set_panic_hook();
      await initThreadPool(msg.threads);
      postMessage({ type: 'ready', threads: n_threads() });
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
        barcode = rips(job.coords, job.pointDim, job.maxDim, job.threshold ?? undefined, job.options);
        break;
      case 'alpha':
        barcode = alpha(job.coords, job.options);
        break;
      case 'grpph':
        barcode = grpph(job.n, job.sources, job.targets, job.weights, job.options);
        break;
      case 'image':
        barcode = image(job.pixels, job.width, job.height, job.superlevel, job.options);
        break;
      case 'complex':
        barcode = complex(
          job.dimensions, job.indptr, job.indices, job.coefficients, job.values, job.modulus,
          job.maxDim ?? undefined, job.options,
        );
        break;
    }
    postMessage({ type: 'result', id, barcode, ms: performance.now() - start });
  } catch (e) {
    // A panic leaves the module unusable (it traps), so the page restarts the worker
    const fatal = e instanceof WebAssembly.RuntimeError;
    postMessage({ type: 'error', id, message: e instanceof Error ? e.message : String(e), fatal });
  }
};
