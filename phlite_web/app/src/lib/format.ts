/** A number for display: a few significant figures, ∞ for infinity. */
export function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined) return '∞';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(digits - 1);
  return String(Number(x.toPrecision(digits)));
}

export function duration(ms: number): string {
  return ms < 1000 ? `${ms.toFixed(ms < 10 ? 1 : 0)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** Download `text` as a file. */
export function download(name: string, text: string, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
