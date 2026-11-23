<script lang="ts" generics="T extends string">
  // A row of mutually exclusive choices, joined into one control.
  let {
    value = $bindable(),
    options,
    onchange,
    label,
  }: {
    value: T;
    options: { id: T; label: string; title?: string; disabled?: boolean }[];
    onchange?: (value: T) => void;
    /** For screen readers. */
    label?: string;
  } = $props();
</script>

<div class="segmented" role="radiogroup" aria-label={label}>
  {#each options as o}
    <button
      role="radio"
      aria-checked={value === o.id}
      class:on={value === o.id}
      title={o.title}
      disabled={o.disabled}
      onclick={() => {
        value = o.id;
        onchange?.(o.id);
      }}>{o.label}</button
    >
  {/each}
</div>

<style>
  .segmented { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
  button { border: none; border-radius: 0; border-right: 1px solid var(--line); background: var(--panel); padding: 0.3rem 0.7rem; font-size: 0.88rem; }
  button:last-child { border-right: none; }
  button.on { background: color-mix(in srgb, var(--accent) 20%, var(--panel)); color: var(--fg); font-weight: 600; }
  button:disabled { opacity: 0.45; }
</style>
