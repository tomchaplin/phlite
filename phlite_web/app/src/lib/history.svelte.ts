/**
 * Undo and redo for a dataset that is replaced (not mutated) on every edit.
 *
 * A gesture (a drag, a stroke of the brush) makes many edits but is undone as one: the view calls `begin` with the
 * data as it was when the pointer went down, and the page calls `edited` on every edit, so the snapshot is kept only
 * if something changed. A one-off change (a button, a new example) is recorded with `commit`.
 */
export class History<T> {
  past = $state.raw<T[]>([]);
  future = $state.raw<T[]>([]);
  private pending: T | null = null;

  constructor(private limit = 100) {}

  /** The start of a gesture, with the data as it is now. */
  begin(current: T) {
    this.pending = current;
  }

  /** Something was edited (in a gesture, if one has begun). */
  edited() {
    if (this.pending === null) return;
    this.push(this.pending);
    this.pending = null;
  }

  /** Record `current` before a one-off change. */
  commit(current: T) {
    this.pending = null;
    this.push(current);
  }

  private push(previous: T) {
    this.past = [...this.past.slice(1 - this.limit), previous];
    this.future = [];
  }

  /** The data to go back to, if any (`current` goes on the redo stack). */
  undo(current: T): T | null {
    if (this.past.length === 0) return null;
    const previous = this.past[this.past.length - 1];
    this.past = this.past.slice(0, -1);
    this.future = [...this.future, current];
    return previous;
  }

  redo(current: T): T | null {
    if (this.future.length === 0) return null;
    const next = this.future[this.future.length - 1];
    this.future = this.future.slice(0, -1);
    this.past = [...this.past, current];
    return next;
  }
}
