<script lang="ts">
  // A short note with the rest of the explanation folded away.
  import type { Snippet } from 'svelte';
  let { lead, children }: { lead: Snippet; children?: Snippet } = $props();
</script>

<div class="note">
  {@render lead()}
  {#if children}
    <details>
      <summary>More</summary>
      <div class="rest">{@render children()}</div>
    </details>
  {/if}
</div>

<style>
  details { display: inline; }
  summary { display: inline; cursor: pointer; color: var(--accent); list-style: none; margin-left: 0.25rem; }
  summary::-webkit-details-marker { display: none; }
  details[open] summary { display: none; }
  .rest { display: block; margin-top: 0.3rem; }
</style>
