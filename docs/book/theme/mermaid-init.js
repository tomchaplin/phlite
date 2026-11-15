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
    // The book's Catppuccin palette (theme/catppuccin.css), as mermaid's base theme's variables
    const style = getComputedStyle(document.documentElement);
    const c = (name) => style.getPropertyValue("--ctp-" + name).trim();
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      darkMode: dark,
      themeVariables: {
        background: c("base"),
        primaryColor: c("surface0"),
        primaryTextColor: c("text"),
        primaryBorderColor: c("blue"),
        secondaryColor: c("mantle"),
        secondaryTextColor: c("text"),
        secondaryBorderColor: c("surface2"),
        tertiaryColor: c("crust"),
        tertiaryTextColor: c("text"),
        tertiaryBorderColor: c("surface2"),
        lineColor: c("overlay2"),
        textColor: c("text"),
        mainBkg: c("surface0"),
        nodeBorder: c("blue"),
        clusterBkg: c("mantle"),
        clusterBorder: c("surface2"),
        edgeLabelBackground: c("base"),
      },
    });
    window.mermaid.run({ querySelector: ".mermaid" });
  };
  document.head.appendChild(script);
})();
