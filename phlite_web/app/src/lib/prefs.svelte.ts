// Preferences that apply to every pipeline, remembered in this browser.

const KEY = 'phlite-playground-parallel';

function load(): boolean {
  try {
    return localStorage.getItem(KEY) !== 'false';
  } catch {
    return true;
  }
}

class Prefs {
  #parallel = $state(load());

  /** Whether the pipelines default to parallel clearing (the default, given threads), or serial clearing. */
  get parallel() {
    return this.#parallel;
  }

  set parallel(value: boolean) {
    this.#parallel = value;
    try {
      localStorage.setItem(KEY, String(value));
    } catch {
      // Not remembered, then
    }
  }
}

export const prefs = new Prefs();
