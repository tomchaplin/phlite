"""Make favicon.svg: the paper plane from phlite_logo.svg, flattened (so that the lines keep one width) and with
thicker lines than the logo's, to read at 16 pixels. Run from the repository root: python3 docs/landing/make_favicon.py
"""

import re

svg = open("phlite_logo.svg").read()
# The plane's six triangles: each a path in a transformed group, filled grey and outlined in black
groups = re.findall(
    r'<g transform="matrix\(([^)]*)\)">\s*<path d="([^"]*)" style="fill:(rgb\([^)]*\));stroke:black', svg
)
polygons = []
for matrix, path, fill in groups:
    a, b, c, d, e, f = map(float, matrix.split(","))
    points = [tuple(map(float, p.split(","))) for p in re.findall(r"[\d.]+,[\d.]+", path)][:-1]
    # The group's matrix, then the enclosing translations
    polygons.append(
        ([(a * x + c * y + e + 484.477 - 531.903, b * x + d * y + f - 251.024 - 269.235) for x, y in points], fill)
    )

xs = [x for p, _ in polygons for x, _ in p]
ys = [y for p, _ in polygons for _, y in p]
side = max(max(xs) - min(xs), max(ys) - min(ys))
size = side * 1.12
left = (min(xs) + max(xs)) / 2 - size / 2
top = (min(ys) + max(ys)) / 2 - size / 2
shade = {"rgb(235,235,235)": "d", "rgb(247,247,247)": "l"}
paths = "".join(
    f'<path class="{shade[fill]}" d="M{" L".join(f"{x - left:.1f},{y - top:.1f}" for x, y in p)}Z"/>'
    for p, fill in polygons
)
# Catppuccin, as the landing page's plane: Latte in light mode, Macchiato in dark
style = (
    f"path{{stroke:#4c4f69;stroke-width:{size * 0.035:.1f};stroke-linejoin:round}}.l{{fill:#eff1f5}}.d{{fill:#ccd0da}}"
    "@media (prefers-color-scheme:dark){path{stroke:#cad3f5}.l{fill:#363a4f}.d{fill:#494d64}}"
)
out = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size:.0f} {size:.0f}"><style>{style}</style>{paths}</svg>\n'
open("docs/landing/favicon.svg", "w").write(out)
print(f"wrote docs/landing/favicon.svg ({len(polygons)} triangles)")
