import { rng } from '../../lib/random';

/** A greyscale image, row by row from the top, values in [0, 1]. */
export interface Img {
  width: number;
  height: number;
  pixels: Float64Array;
}

function make(width: number, height: number, f: (x: number, y: number) => number): Img {
  const pixels = new Float64Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) pixels[y * width + x] = f(x, y);
  return { width, height, pixels };
}

/** Rescale to [0, 1]. */
export function normalise(img: Img): Img {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of img.pixels) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = hi - lo || 1;
  return { ...img, pixels: img.pixels.map((v) => (v - lo) / span) };
}

/** A sum of Gaussian bumps. */
function bumps(size: number, seed: number, count: number, width: number): Img {
  const r = rng(seed);
  const centres = Array.from({ length: count }, () => [r.uniform(0, size), r.uniform(0, size), r.uniform(0.3, 1), width * r.uniform(0.6, 1.4)]);
  return normalise(
    make(size, size, (x, y) => centres.reduce((s, [cx, cy, h, w]) => s + h * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w)), 0)),
  );
}

/** Smooth noise: random values on a coarse grid, smoothly interpolated, over a few octaves. */
export function smoothNoise(size: number, scale: number, seed: number): Img {
  const r = rng(seed);
  const octaves = [1, 0.5, 0.25];
  const grids = octaves.map((_, o) => {
    const cells = Math.max(2, Math.round(size / (scale / 2 ** o))) + 2;
    return { cells, values: Array.from({ length: cells * cells }, () => r.next()) };
  });
  const smooth = (t: number) => t * t * (3 - 2 * t);
  return normalise(
    make(size, size, (x, y) =>
      grids.reduce((s, { cells, values }, o) => {
        const gx = (x / size) * (cells - 2);
        const gy = (y / size) * (cells - 2);
        const ix = Math.floor(gx);
        const iy = Math.floor(gy);
        const fx = smooth(gx - ix);
        const fy = smooth(gy - iy);
        const v = (i: number, j: number) => values[j * cells + i];
        const top = v(ix, iy) * (1 - fx) + v(ix + 1, iy) * fx;
        const bottom = v(ix, iy + 1) * (1 - fx) + v(ix + 1, iy + 1) * fx;
        return s + octaves[o] * (top * (1 - fy) + bottom * fy);
      }, 0),
    ),
  );
}

/** Text drawn on a canvas and blurred a little: dark letters on a light background. */
function text(word: string, width: number, height: number): Img {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);
  ctx.filter = 'blur(0.8px)';
  ctx.fillStyle = '#000';
  ctx.font = `bold ${Math.round(height * 0.62)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(word, width / 2, height * 0.52);
  return fromImageData(ctx.getImageData(0, 0, width, height));
}

export function fromImageData(data: ImageData): Img {
  const { width, height } = data;
  const pixels = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const [r, g, b] = [data.data[4 * i], data.data[4 * i + 1], data.data[4 * i + 2]];
    pixels[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  }
  return { width, height, pixels };
}

export interface ImageExample {
  id: string;
  name: string;
  about: string;
  superlevel: boolean;
  make: () => Img;
}

export const EXAMPLES: ImageExample[] = [
  {
    id: 'rings',
    name: 'Rings',
    about: 'Bright rings on a dark background. Sublevel sets: the dark background loops around each ring (H₁) until the ring fills in, and the dark disc inside each ring is its own component (H₀) until then. In superlevel sets the rings themselves are the loops.',
    superlevel: false,
    make: () =>
      make(64, 64, (x, y) => {
        const d1 = Math.hypot(x - 22, y - 24);
        const d2 = Math.hypot(x - 46, y - 42);
        return Math.max(0.9 * Math.exp(-((d1 - 13) ** 2) / 6), 0.6 * Math.exp(-((d2 - 9) ** 2) / 5));
      }),
  },
  {
    id: 'blobs',
    name: 'Blobs',
    about: 'Gaussian bumps of different heights. Superlevel sets: each peak is born at its height and merges with a higher one at the saddle between them.',
    superlevel: true,
    make: () => bumps(64, 3, 9, 6),
  },
  {
    id: 'text',
    name: 'Text',
    about: 'The word "phlite", dark on light. Sublevel sets: each letter is a component (H₀); the holes in p and e are loops (H₁).',
    superlevel: false,
    make: () => text('phlite', 120, 48),
  },
  {
    id: 'noise',
    name: 'Smooth noise',
    about: 'Random smooth terrain: many bars of every length, from its minima, saddles and maxima.',
    superlevel: false,
    make: () => smoothNoise(64, 16, 5),
  },
  {
    id: 'cells',
    name: 'Cells',
    about: 'A Voronoi-like texture: bright walls around dark cells, so every cell is an H₁ bar.',
    superlevel: false,
    make: () => {
      const r = rng(11);
      const sites = Array.from({ length: 14 }, () => [r.uniform(0, 72), r.uniform(0, 72)]);
      return normalise(
        make(72, 72, (x, y) => {
          const d = sites.map(([sx, sy]) => Math.hypot(x - sx, y - sy)).sort((a, b) => a - b);
          return Math.exp(-((d[1] - d[0]) ** 2) / 8) + 0.05 * r.next();
        }),
      );
    },
  },
];

/** Load an image file as greyscale, scaled down so that its longer side is at most `maxSide` pixels. */
export async function loadImage(file: File, maxSide: number): Promise<Img> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, width, height);
  return fromImageData(ctx.getImageData(0, 0, width, height));
}

/** A 3×3 box blur, for softening a painting. */
export function blur(img: Img): Img {
  const { width, height, pixels } = img;
  const out = new Float64Array(pixels.length);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let s = 0;
      let n = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
            s += pixels[yy * width + xx];
            n++;
          }
        }
      out[y * width + x] = s / n;
    }
  return { width, height, pixels: out };
}
