<script lang="ts">
  import { engine } from './lib/engine.svelte';
  import Points from './pipelines/points/Points.svelte';
  import Images from './pipelines/images/Images.svelte';
  import Networks from './pipelines/networks/Networks.svelte';
  import Complexes from './pipelines/complexes/Complexes.svelte';

  type Tab = 'points' | 'images' | 'networks' | 'complexes';
  const tabs: [Tab, string, string][] = [
    ['points', 'Point clouds', 'Vietoris–Rips'],
    ['images', 'Images', 'cubical, sub/superlevel'],
    ['networks', 'Networks', 'GrPPH of weighted digraphs'],
    ['complexes', 'Chain complexes', 'explicit boundary matrices'],
  ];
  const fromHash = () => (tabs.find(([id]) => `#${id}` === location.hash)?.[0] ?? 'points') as Tab;
  let tab = $state<Tab>(fromHash());
  // Mount a pipeline the first time it's shown, then keep it (and its data) alive
  let visited = $state(new Set<Tab>([fromHash()]));
  function show(t: Tab) {
    tab = t;
    visited = new Set([...visited, t]);
    history.replaceState(null, '', `#${t}`);
  }
  // Follow the address bar too (edited links, back and forward)
  $effect(() => {
    const follow = () => show(fromHash());
    addEventListener('hashchange', follow);
    return () => removeEventListener('hashchange', follow);
  });
</script>

<header>
  <a class="brand" href="../">phlite</a>
  <span class="title">playground</span>
  <nav>
    <a href="../book/">Book</a>
    <a href="../api/">API</a>
    <a href="https://github.com/tomchaplin/phlite">GitHub</a>
  </nav>
  <span class="status" class:err={engine.status === 'error'}>
    {#if engine.status === 'loading'}starting phlite…
    {:else if engine.status === 'error'}phlite failed to start
    {:else}phlite · WebAssembly · {engine.threads} thread{engine.threads === 1 ? '' : 's'}{engine.status === 'busy' ? ' · computing…' : ''}
    {/if}
  </span>
</header>

{#if engine.error}
  <div class="banner error">{engine.error}</div>
{/if}

<nav class="pipelines">
  {#each tabs as [id, name, sub]}
    <button class:active={tab === id} onclick={() => show(id)}>
      <strong>{name}</strong><span>{sub}</span>
    </button>
  {/each}
</nav>

<main>
  {#each tabs as [id]}
    {#if visited.has(id)}
      <section hidden={tab !== id}>
        {#if id === 'points'}<Points />
        {:else if id === 'images'}<Images />
        {:else if id === 'networks'}<Networks />
        {:else}<Complexes />{/if}
      </section>
    {/if}
  {/each}
</main>

<style>
  header { display: flex; align-items: baseline; gap: 0.75rem; padding: 0.7rem 1.2rem; border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
  .brand { font-weight: 800; font-size: 1.25rem; color: var(--fg); text-decoration: none; }
  .title { color: var(--muted); }
  header nav { display: flex; gap: 0.9rem; margin-left: 1rem; }
  header nav a { text-decoration: none; }
  .status { margin-left: auto; font-size: 0.85rem; color: var(--muted); }
  .status.err { color: var(--danger); }
  .banner { margin: 0.8rem 1.2rem 0; padding: 0.7rem 0.9rem; border: 1px solid var(--danger); border-radius: 8px; }
  .pipelines { display: flex; gap: 0.5rem; padding: 0.8rem 1.2rem 0; flex-wrap: wrap; }
  .pipelines button { display: flex; flex-direction: column; align-items: flex-start; padding: 0.45rem 0.9rem; background: var(--panel); }
  .pipelines button span { font-size: 0.78rem; color: var(--muted); }
  .pipelines button.active { border-color: var(--accent); box-shadow: inset 0 -3px 0 var(--accent); }
  main { padding: 0.8rem 1.2rem 2rem; }
  @media (max-width: 600px) {
    header, main { padding-left: 16px; padding-right: 16px; }
    .pipelines { padding-left: 16px; padding-right: 16px; }
  }
</style>
