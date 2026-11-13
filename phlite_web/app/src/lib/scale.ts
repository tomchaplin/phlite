import type { Bar } from './types';

/** About `count` round tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** The range of filtration values to show for `bars` (and any `extra` values, such as the data's), padded a little. */
export function barDomain(bars: Bar[], extra: number[] = []): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  const see = (v: number | null) => {
    if (v === null || !Number.isFinite(v)) return;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  };
  for (const b of bars) {
    see(b.birth);
    see(b.death);
  }
  extra.forEach(see);
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi === lo) hi = lo + 1;
  return [lo, hi + (hi - lo) * 0.08];
}

/** Whether `bar` is alive at `t` (with `reversed`, for superlevel filtrations, values decrease). */
export function alive(bar: Bar, t: number, reversed = false): boolean {
  if (reversed) return bar.birth >= t && (bar.death === null || bar.death < t);
  return bar.birth <= t && (bar.death === null || bar.death > t);
}

export const persistence = (bar: Bar) => (bar.death === null ? Infinity : Math.abs(bar.death - bar.birth));

export const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
export const H = (d: number) => `H${String(d).split('').map((c) => SUB[+c]).join('')}`;
