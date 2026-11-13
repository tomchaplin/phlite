import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { engine } from './lib/engine.svelte';

engine.start();
export default mount(App, { target: document.getElementById('app')! });

// For tests and the console: `await __phlite.run({ pipeline: 'rips', ... })`
(window as unknown as { __phlite: typeof engine }).__phlite = engine;
