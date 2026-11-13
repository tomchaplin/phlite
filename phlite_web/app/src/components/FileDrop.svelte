<script lang="ts">
  let { accept = '', onfile, children }: { accept?: string; onfile: (file: File) => void; children?: import('svelte').Snippet } = $props();
  let over = $state(false);
  let input: HTMLInputElement;
</script>

<div
  class="drop"
  class:over
  role="button"
  tabindex="0"
  onclick={() => input.click()}
  onkeydown={(e) => e.key === 'Enter' && input.click()}
  ondragover={(e) => { e.preventDefault(); over = true; }}
  ondragleave={() => (over = false)}
  ondrop={(e) => { e.preventDefault(); over = false; const f = e.dataTransfer?.files[0]; if (f) onfile(f); }}
>
  <strong>Drop a file here</strong> or click to choose one
  {#if children}<div class="help">{@render children()}</div>{/if}
  <input bind:this={input} type="file" {accept} hidden onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) onfile(f); e.currentTarget.value = ''; }} />
</div>

<style>
  .drop { border: 2px dashed var(--line); border-radius: 8px; padding: 1rem; text-align: center; cursor: pointer; color: var(--muted); }
  .drop.over, .drop:hover { border-color: var(--accent); color: var(--fg); }
  .help { margin-top: 0.5rem; font-size: 0.82rem; text-align: left; }
  .help :global(code) { font-size: 0.8rem; }
  .help :global(pre) { margin: 0.3rem 0; font-size: 0.78rem; background: var(--panel2); padding: 0.4rem; border-radius: 4px; overflow-x: auto; }
</style>
