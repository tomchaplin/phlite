/** How consecutive complexes of a zigzag are joined, as the choices of a `Segmented` control. */
export type Join = 'union' | 'intersection';
export const JOINS: { id: Join; label: string; title: string }[] = [
  { id: 'union', label: 'Unions', title: 'K₀ → K₀ ∪ K₁ ← K₁ → …: a feature lasts while it overlaps itself' },
  { id: 'intersection', label: 'Intersections', title: 'K₀ ← K₀ ∩ K₁ → K₁ ← …: a feature lasts while what it shares with the next carries it' },
];
