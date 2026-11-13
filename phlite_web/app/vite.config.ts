import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Shared memory (for the WebAssembly thread pool) needs the page to be cross-origin isolated.
// The docs site sets the same headers for /playground/ (docs/landing/_headers).
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // Relative, so that the build works under /playground/
  base: './',
  plugins: [svelte()],
  // The WebAssembly package is linked from ../pkg, outside the app, which the dev server refuses by default
  server: { headers: isolation, fs: { allow: ['..'] } },
  preview: { headers: isolation },
  worker: { format: 'es' },
  // Pre-bundling would break the thread pool's `new URL(..., import.meta.url)`
  optimizeDeps: { exclude: ['phlite-web', 'phlite-web-single'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1000 },
});
