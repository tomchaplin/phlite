<script lang="ts">
  export type Source = 'examples' | 'random' | 'upload' | 'draw';
  let { source = $bindable(), drawLabel = 'Draw' }: { source: Source; drawLabel?: string } = $props();
  const tabs: [Source, string][] = $derived([
    ['examples', 'Examples'],
    ['random', 'Random'],
    ['upload', 'Upload'],
    ['draw', drawLabel],
  ]);
</script>

<div class="tabs" role="tablist">
  {#each tabs as [id, name]}
    <button role="tab" aria-selected={source === id} class:active={source === id} onclick={() => (source = id)}>{name}</button>
  {/each}
</div>

<style>
  .tabs { display: flex; gap: 0.25rem; border-bottom: 1px solid var(--line); margin-bottom: 0.6rem; }
  button { border: none; background: none; padding: 0.35rem 0.7rem; border-bottom: 2px solid transparent; border-radius: 0; color: var(--muted); }
  button.active { color: var(--fg); border-bottom-color: var(--accent); font-weight: 600; }
</style>
