// The Web Worker that owns the WebAssembly module. With threads, the module's pool is more Web Workers, started by
// wasm-bindgen-rayon; a computation blocks this worker (never the page) until it is done. Without (the page can't
// share memory, or the threaded module wouldn't start), it loads the build without threads instead.
import type { Job } from './types';

// What both builds export (their glue differs)
type Module = Pick<
  typeof import('phlite-web-single'),
  'rips' | 'rips_six_pack' | 'cloud_zigzag' | 'moving_zigzag' | 'network_zigzag' | 'timed_zigzag' | 'video_zigzag' | 'magnitude' | 'mpss' | 'binary_filtration' | 'delaunay' | 'grpph' | 'network' | 'graph' | 'image' | 'complex' | 'n_threads' | 'set_crash_hooks' | 'crash_for_testing'
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
        barcode = phlite.rips(
          job.coords, job.pointDim, job.maxDim, job.threshold ?? undefined, job.metric, job.dtmK ?? undefined, job.options,
        );
        break;
      case 'rips-six-pack':
        barcode = phlite.rips_six_pack(
          job.coords, job.pointDim, job.subset, job.maxDim, job.threshold ?? undefined, job.metric, job.options,
        );
        break;
      case 'delaunay':
        barcode = phlite.delaunay(job.coords, job.pointDim, job.cech, job.options);
        break;
      case 'cloud-zigzag':
        barcode = phlite.cloud_zigzag(
          job.coords, job.pointDim, job.window, job.step, job.threshold, job.maxDim, job.metric, job.complex, job.intersection,
        );
        break;
      case 'moving-zigzag':
        barcode = phlite.moving_zigzag(
          job.coords, job.pointDim, job.nFrames, job.threshold, job.maxDim, job.metric, job.complex, job.intersection,
        );
        break;
      case 'timed-zigzag':
        barcode = phlite.timed_zigzag(
          job.dimensions, job.indptr, job.indices, job.lifeIndptr, job.starts, job.ends, job.maxDim ?? undefined, job.intersection,
        );
        break;
      case 'network-zigzag':
        barcode = phlite.network_zigzag(
          job.n, job.sources, job.targets, job.weights, job.directed, job.width, job.step, job.maxDim, job.intersection,
        );
        break;
      case 'video-zigzag':
        barcode = phlite.video_zigzag(
          job.pixels, job.width, job.height, job.nFrames, job.threshold, job.superlevel, job.construction, job.periodic, job.intersection,
        );
        break;
      case 'mpss':
        barcode = phlite.mpss(job.n, job.sources, job.targets, job.lMax, job.modulus, job.rMax, 4000);
        break;
      case 'magnitude':
        barcode = phlite.magnitude(job.n, job.sources, job.targets, job.lMax, job.integral);
        break;
      case 'grpph':
        barcode = phlite.grpph(job.n, job.sources, job.targets, job.weights, job.grounded, job.options);
        break;
      case 'network':
        barcode = phlite.network(job.kind, job.shortestPath, job.n, job.sources, job.targets, job.weights, job.maxDim, job.options);
        break;
      case 'graph':
        barcode = phlite.graph(job.kind, job.n, job.sources, job.targets, job.weights, job.maxDim, job.options);
        break;
      case 'image':
        {
          const pixels = job.binary ? phlite.binary_filtration(job.pixels, job.width, job.height, job.binary) : job.pixels;
          barcode = phlite.image(pixels, job.width, job.height, job.superlevel, job.construction, job.periodic, job.options);
        }
        break;
      case 'complex':
        barcode = phlite.complex(
          job.dimensions, job.indptr, job.indices, job.coefficients, job.values, job.modulus,
          job.maxDim ?? undefined, job.extended, job.options,
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
