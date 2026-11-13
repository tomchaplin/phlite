// Render ```mermaid code blocks with mermaid (loaded from a CDN, pinned), in the book's light or dark theme.
(function () {
  const blocks = document.querySelectorAll("code.language-mermaid");
  if (blocks.length === 0) return;
  const dark = ["coal", "navy", "ayu"].some((t) => document.documentElement.classList.contains(t));
  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js";
  script.onload = function () {
    blocks.forEach(function (code) {
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      code.parentElement.replaceWith(div);
    });
    window.mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral" });
    window.mermaid.run({ querySelector: ".mermaid" });
  };
  document.head.appendChild(script);
})();
