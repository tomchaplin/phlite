// Colours for canvases, SVG attributes and three.js, read from the CSS variables in app.css (Catppuccin Latte in light
// mode, Macchiato in dark), so that everything drawn matches the page. Reading a colour makes the caller depend on the
// colour scheme: when the system switches between light and dark, every view that used one redraws.

let scheme = $state(0);
const cache = new Map<string, string>();

if (typeof window !== 'undefined' && window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    cache.clear();
    scheme++;
  });
}

/** The value of the CSS colour variable `name` (e.g. `'--accent'`) on the page. */
export function cssColour(name: string): string {
  // Depend on the scheme, so callers recompute when it changes
  void scheme;
  let value = cache.get(name);
  if (value === undefined) {
    value = typeof document === 'undefined' ? '' : getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    value ||= '#888888';
    cache.set(name, value);
  }
  return value;
}

/** The colour of homology dimension `d`: blue, peach, green, red, mauve (then mauve again). */
export const dimColour = (d: number) => cssColour(`--dim${Math.min(d, 4)}`);

/** A `#rrggbb` colour as `[r, g, b]`. */
export function rgb(colour: string): [number, number, number] {
  const hex = colour.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
}
