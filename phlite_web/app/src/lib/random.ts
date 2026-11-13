/** A small seeded random number generator (mulberry32), so that examples and random data are reproducible. */
export function rng(seed: number) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(next(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  return {
    next,
    normal,
    uniform: (lo: number, hi: number) => lo + (hi - lo) * next(),
    int: (n: number) => Math.floor(next() * n),
  };
}
export type Rng = ReturnType<typeof rng>;

export const randomSeed = () => Math.floor(Math.random() * 1e6);
