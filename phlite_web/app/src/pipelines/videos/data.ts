// Videos for the zigzag pipeline: synthetic clips, and frames taken from a video file or the camera, as greyscale
// images (values in [0, 1], 0 dark), all of one size.
import type { Img } from '../images/data';

export interface Video {
  width: number;
  height: number;
  frames: Img[];
}

export interface VideoExample {
  id: string;
  name: string;
  about: string;
  threshold: number;
  superlevel: boolean;
  make: () => Video;
}

function clip(width: number, height: number, n: number, f: (x: number, y: number, t: number) => number): Video {
  const frames: Img[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    const pixels = new Float64Array(width * height);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) pixels[y * width + x] = Math.min(1, Math.max(0, f(x, y, t)));
    frames.push({ width, height, pixels });
  }
  return { width, height, frames };
}

const blob = (x: number, y: number, cx: number, cy: number, r: number) => Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r));

/** A small seeded generator (mulberry32). */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const EXAMPLES: VideoExample[] = [
  {
    id: 'merge',
    name: 'Blobs that merge and part',
    about:
      'Two dark blobs drift together, merge and drift apart again. Below the threshold there are two components, then one (an H₀ bar ends where they meet), then two again (a new bar is born where they part).',
    threshold: 0.5,
    superlevel: false,
    make: () =>
      clip(48, 32, 40, (x, y, t) => {
        const d = 14 * Math.abs(Math.cos(Math.PI * t));
        return 1 - blob(x, y, 24 - d, 16, 4.5) - blob(x, y, 24 + d, 16, 4.5);
      }),
  },
  {
    id: 'ring',
    name: 'A ring closes and opens',
    about:
      'A dark arc grows into a full ring, which then breaks open again: an H₁ bar lives exactly while the ring is closed.',
    threshold: 0.5,
    superlevel: false,
    make: () =>
      clip(40, 40, 36, (x, y, t) => {
        const extent = 2 * Math.PI * (0.55 + 0.6 * Math.sin(Math.PI * t));
        const r = Math.hypot(x - 19.5, y - 19.5);
        let angle = Math.atan2(y - 19.5, x - 19.5);
        if (angle < 0) angle += 2 * Math.PI;
        return Math.abs(r - 12) < 2.6 && angle <= extent ? 0 : 1;
      }),
  },
  {
    id: 'bubbles',
    name: 'Drifting bubbles',
    about:
      'Bright bubbles appear, drift and fade (superlevel sets, bright first): each bubble is an H₀ bar from when it rises above the threshold to when it fades below it or merges with another.',
    threshold: 0.45,
    superlevel: true,
    make: () => {
      const r = rng(7);
      const bubbles = Array.from({ length: 7 }, () => ({
        x: r() * 56, y: r() * 40, vx: (r() - 0.5) * 30, vy: (r() - 0.5) * 20, start: r() * 0.6, life: 0.3 + r() * 0.4, size: 3 + r() * 3,
      }));
      return clip(56, 40, 48, (x, y, t) =>
        bubbles.reduce((s, b) => {
          const age = (t - b.start) / b.life;
          if (age < 0 || age > 1) return s;
          return s + Math.sin(Math.PI * age) * blob(x, y, b.x + b.vx * t, b.y + b.vy * t, b.size);
        }, 0),
      );
    },
  },
];

/** Greyscale (brightness, in [0, 1]) of what `source` shows now, scaled to `width` × `height`. */
function grab(source: CanvasImageSource, width: number, height: number, ctx: CanvasRenderingContext2D): Img {
  ctx.drawImage(source, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  const pixels = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) pixels[i] = (0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]) / 255;
  return { width, height, pixels };
}

function scaled(w: number, h: number, maxSide: number): [number, number] {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

function canvas(width: number, height: number): CanvasRenderingContext2D {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return c.getContext('2d', { willReadFrequently: true })!;
}

/** Seek `video` to `time` and wait until the frame there is ready (or a second, if no event comes). */
function seek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener('seeked', done);
      resolve();
    };
    const timer = setTimeout(done, 1000);
    video.addEventListener('seeked', done);
    video.currentTime = time;
  });
}

/** `nFrames` frames evenly spread through a video file, in greyscale, with the longest side at most `maxSide`. */
export async function decodeVideo(file: Blob, nFrames: number, maxSide: number): Promise<Video> {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  const url = URL.createObjectURL(file);
  video.src = url;
  try {
    await new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error("Your browser can't decode this video"));
    });
    // Some files (often WebM from recorders) don't say how long they are: seeking far past the end makes the browser
    // find out
    if (!Number.isFinite(video.duration) || video.duration === 0) {
      await seek(video, 1e101);
    }
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    if (!(duration > 0)) throw new Error("Couldn't find how long this video is");
    const [width, height] = scaled(video.videoWidth, video.videoHeight, maxSide);
    const ctx = canvas(width, height);
    const frames: Img[] = [];
    for (let i = 0; i < nFrames; i++) {
      await seek(video, ((i + 0.5) / nFrames) * duration);
      frames.push(grab(video, width, height, ctx));
    }
    return { width, height, frames };
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** `nFrames` frames from the camera, `intervalMs` apart, in greyscale, with the longest side at most `maxSide`. */
export async function recordCamera(nFrames: number, intervalMs: number, maxSide: number): Promise<Video> {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    const [width, height] = scaled(video.videoWidth, video.videoHeight, maxSide);
    const ctx = canvas(width, height);
    const frames: Img[] = [];
    for (let i = 0; i < nFrames; i++) {
      frames.push(grab(video, width, height, ctx));
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    return { width, height, frames };
  } finally {
    stream.getTracks().forEach((t) => t.stop());
  }
}

/** The frames one after another, for the WebAssembly module. */
export function concat(video: Video): Float64Array {
  const out = new Float64Array(video.width * video.height * video.frames.length);
  video.frames.forEach((f, i) => out.set(f.pixels, i * video.width * video.height));
  return out;
}
