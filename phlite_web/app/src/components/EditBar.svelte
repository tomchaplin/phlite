<script lang="ts" generics="T extends string">
  import type { Snippet } from 'svelte';

  /**
   * The editing toolbar under a dataset's view: a choice of tool (or none, to just look), undo and redo, and whatever
   * else the pipeline adds. Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y) undo and redo, Escape puts the tool down and Delete
   * calls `ondelete`, while this toolbar is on screen and no text field has the focus.
   */
  let {
    tool = $bindable(),
    tools,
    canUndo,
    canRedo,
    onundo,
    onredo,
    ondelete,
    children,
  }: {
    tool: T | null;
    tools: [T, string, string][];
    canUndo: boolean;
    canRedo: boolean;
    onundo: () => void;
    onredo: () => void;
    ondelete?: () => void;
    children?: Snippet;
  } = $props();

  let bar: HTMLElement;
  function key(e: KeyboardEvent) {
    // Only on the pipeline being shown (the others are kept alive, hidden)
    if (!bar || bar.offsetParent === null) return;
    const target = e.target as HTMLElement | null;
    if (target && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName))) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) onredo();
      else onundo();
    } else if (mod && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      onredo();
    } else if ((e.key === 'Delete' || e.key === 'Backspace') && ondelete) {
      e.preventDefault();
      ondelete();
    } else if (e.key === 'Escape') tool = null;
  }
</script>

<svelte:window onkeydown={key} />

<div class="editbar" bind:this={bar}>
  <div class="tools" role="toolbar" aria-label="Edit">
    <button class:on={tool === null} title="Just look: select bars, move the slider" onclick={() => (tool = null)}>View</button>
    {#each tools as [id, name, help]}
      <button class:on={tool === id} title={help} onclick={() => (tool = tool === id ? null : id)}>{name}</button>
    {/each}
    <span class="sep"></span>
    <button disabled={!canUndo} title="Undo (Ctrl+Z)" aria-label="Undo" onclick={onundo}>↶</button>
    <button disabled={!canRedo} title="Redo (Ctrl+Shift+Z)" aria-label="Redo" onclick={onredo}>↷</button>
  </div>
  {#if children}<div class="extra">{@render children()}</div>{/if}
</div>

<style>
  .editbar { display: flex; flex-direction: column; gap: 0.35rem; margin-top: 0.6rem; padding-top: 0.55rem; border-top: 1px dashed var(--line); }
  .sep { flex: 1; }
  .extra { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.8rem; font-size: 0.85rem; }
</style>
