"""Draw the book's figures (SVG) into docs/book/src/figures/: the data plots with Altair (and vl-convert to save them),
the diagrams of matrices and data structures with matplotlib. `just docs` runs it in python-tests' `latest`
environment, which has all of these in its `docs` group:

    python docs/figures/make_figures.py

Most figures use one running example, a small filtered simplicial complex over Z/2 (see `CELLS`).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import altair as alt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "book", "src", "figures")

# A palette that reads on white (figures sit on a white card in every book theme)
INK = "#222222"
MUTED = "#9a9a9a"
PALE = "#eeeeee"
BLUE = "#3b6fb6"
ORANGE = "#e07b39"
GREEN = "#3a9d5d"
RED = "#c8453a"
PURPLE = "#7d5ba6"
DIM_COLOURS = [BLUE, ORANGE, GREEN]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
        "axes.edgecolor": INK,
        "text.color": INK,
    }
)

# ====== The running example ==================================================
# (name, dimension, faces (by index), filtration value), listed in filtration order
CELLS = [
    ("a", 0, [], 0),
    ("b", 0, [], 0),
    ("c", 0, [], 1),
    ("ab", 1, [0, 1], 1),
    ("bc", 1, [1, 2], 1),
    ("ac", 1, [0, 2], 2),
    ("d", 0, [], 2),
    ("cd", 1, [2, 6], 3),
    ("bd", 1, [1, 6], 3),
    ("abc", 2, [3, 4, 5], 4),
]
N = len(CELLS)
POS = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (0.5, 0.85), "d": (1.55, 0.95)}


def reduce_standard(columns):
    """The standard algorithm over Z/2: (R, V, pairs), with columns as sets of indices."""
    R = [set(c) for c in columns]
    V = [{j} for j in range(len(columns))]
    owner = {}
    pairs = []
    for j in range(len(columns)):
        while R[j] and max(R[j]) in owner:
            k = owner[max(R[j])]
            R[j] ^= R[k]
            V[j] ^= V[k]
        if R[j]:
            owner[max(R[j])] = j
            pairs.append((max(R[j]), j))
    return R, V, pairs


BOUNDARY = [set(c[2]) for c in CELLS]
R, V, PAIRS = reduce_standard(BOUNDARY)
PAIRED = {x for p in PAIRS for x in p}
ESSENTIAL = [i for i in range(N) if i not in PAIRED]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def bars():
    """(dimension, birth value, death value, birth cell, death cell) of every bar of positive length."""
    out = []
    for b, d in PAIRS:
        if CELLS[b][3] != CELLS[d][3]:
            out.append((CELLS[b][1], CELLS[b][3], CELLS[d][3], CELLS[b][0], CELLS[d][0]))
    for e in ESSENTIAL:
        out.append((CELLS[e][1], CELLS[e][3], None, CELLS[e][0], None))
    return sorted(out, key=lambda x: (x[0], x[1]))


# The data plots (the filtration and its barcode) are drawn with Altair; the diagrams below with matplotlib.

def save_altair(chart, name):
    os.makedirs(OUT, exist_ok=True)
    chart.save(os.path.join(OUT, name))
    print("wrote", name)


def fig_filtration():
    """The running example's subcomplex at each time, new cells in orange. The triangle is a geoshape (Altair has no
    polygon mark), so everything is drawn in the same identity projection, with longitude and latitude as x and y."""
    panels = []
    for t in range(5):
        present = [c for c in CELLS if c[3] <= t]
        colour = lambda value: ORANGE if value == t else None
        triangles = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"new": value == t},
                    "geometry": {"type": "Polygon", "coordinates": [[list(POS[v]) for v in name] + [list(POS[name[0]])]]},
                }
                for name, dim, _, value in present
                if dim == 2
            ],
        }
        edges = pl.DataFrame(
            [
                {"x": POS[n[0]][0], "y": POS[n[0]][1], "x2": POS[n[1]][0], "y2": POS[n[1]][1], "new": value == t}
                for n, dim, _, value in present
                if dim == 1
            ],
            schema={"x": pl.Float64, "y": pl.Float64, "x2": pl.Float64, "y2": pl.Float64, "new": pl.Boolean},
        )
        vertices = pl.DataFrame(
            [{"x": POS[n][0], "y": POS[n][1], "name": n, "new": value == t} for n, dim, _, value in present if dim == 0]
        )
        # Fixed colours (layers encoding the same field would otherwise share one colour scale)
        new_or = lambda field, old: alt.condition(f"datum.{field}", alt.value(ORANGE), alt.value(old))
        # A frame of invisible points fixes every panel's extent
        frame = alt.Chart(pl.DataFrame({"x": [-0.25, 1.85], "y": [-0.25, 1.15]})).mark_point(opacity=0).encode(
            longitude="x:Q", latitude="y:Q"
        )
        layers = [frame]
        if triangles["features"]:
            layers.append(
                alt.Chart(alt.Data(values=triangles["features"])).mark_geoshape(opacity=0.35, stroke=None).encode(
                    color=new_or("properties.new", GREEN)
                )
            )
        if len(edges):
            layers.append(
                alt.Chart(edges).mark_rule(strokeWidth=2.5).encode(
                    longitude="x:Q", latitude="y:Q", longitude2="x2:Q", latitude2="y2:Q", color=new_or("new", INK)
                )
            )
        layers.append(
            alt.Chart(vertices).mark_circle(size=120, opacity=1).encode(
                longitude="x:Q",
                latitude="y:Q",
                color=new_or("new", BLUE),
            )
        )
        layers.append(
            alt.Chart(vertices).mark_text(dx=-9, dy=-9, fontSize=12, color=INK).encode(
                longitude="x:Q", latitude="y:Q", text="name:N"
            )
        )
        panels.append(
            alt.layer(*layers)
            .project(type="identity", reflectY=True)
            .properties(width=150, height=95, title=f"t = {t}")
        )
    chart = (
        alt.hconcat(*panels)
        .properties(title="A filtration: cells enter over time (new cells in orange)")
        .configure_view(stroke=None)
        .configure_title(fontSize=13, anchor="middle")
    )
    save_altair(chart, "filtration.svg")


def fig_barcode():
    rows = []
    for y, (dim, birth, death, birth_cell, death_cell) in enumerate(bars()):
        rows.append(
            {
                "bar": y,
                "dimension": f"H{dim}",
                "birth": birth,
                "end": 5.0 if death is None else death,
                "essential": death is None,
                "label": f"H{dim}: born at {birth_cell}" + (f", dies at {death_cell}" if death_cell else ", never dies"),
            }
        )
    frame = pl.DataFrame(rows)
    colour = alt.Color("dimension:N", scale=alt.Scale(domain=["H0", "H1"], range=[BLUE, ORANGE]), legend=None)
    y = alt.Y("bar:O", axis=None, sort="descending")
    x = alt.X("birth:Q", title="filtration value", scale=alt.Scale(domain=[-0.2, 9.4]), axis=alt.Axis(values=[0, 1, 2, 3, 4]))
    bars_ = alt.Chart(frame).mark_rule(strokeWidth=7).encode(x=x, x2="end:Q", y=y, color=colour)
    arrows = alt.Chart(frame.filter(pl.col("essential"))).mark_point(
        shape="triangle-right", filled=True, size=120, opacity=1
    ).encode(x="end:Q", y=y, color=colour)
    labels = alt.Chart(frame).transform_calculate(at="datum.end + 0.35").mark_text(align="left", fontSize=11).encode(
        x="at:Q", y=y, text="label:N"
    )
    chart = (
        (bars_ + arrows + labels)
        .properties(width=420, height=140, title="Its barcode (H0 blue, H1 orange)")
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )
    save_altair(chart, "barcode.svg")


def draw_matrix(ax, columns, title, row_labels=None, col_labels=None, highlight_pivots=True,
                cleared=(), shade_cols=(), shade_rows=(), mark=None, entry_colour=INK):
    n_rows = len(row_labels) if row_labels is not None else N
    n_cols = len(columns)
    row_labels = row_labels or [c[0] for c in CELLS]
    col_labels = col_labels or [c[0] for c in CELLS]
    for j in shade_cols:
        ax.add_patch(Rectangle((j, 0), 1, n_rows, facecolor="#dfe9f7", edgecolor="none"))
    for i in shade_rows:
        ax.add_patch(Rectangle((0, i), n_cols, 1, facecolor="#fbe6d8", edgecolor="none"))
    for j in cleared:
        ax.add_patch(Rectangle((j, 0), 1, n_rows, facecolor=PALE, edgecolor="none", hatch="///"))
    for j, col in enumerate(columns):
        pivot = max(col) if col else None
        for i in col:
            is_pivot = highlight_pivots and i == pivot
            if is_pivot:
                ax.add_patch(Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, facecolor=ORANGE, edgecolor="none"))
            ax.text(j + 0.5, i + 0.52, "1", ha="center", va="center",
                    color="white" if is_pivot else entry_colour, fontsize=10, fontweight="bold" if is_pivot else None)
    if mark:
        for (i, j), colour in mark.items():
            ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor="none", edgecolor=colour, lw=2.5))
    for k in range(n_cols + 1):
        ax.plot([k, k], [0, n_rows], color="#d0d0d0", lw=0.6)
    for k in range(n_rows + 1):
        ax.plot([0, n_cols], [k, k], color="#d0d0d0", lw=0.6)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks([k + 0.5 for k in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks([k + 0.5 for k in range(n_rows)])
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    ax.set_title(title, pad=24)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_reduction():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    draw_matrix(axes[0], BOUNDARY, "D (boundary matrix)", highlight_pivots=False)
    draw_matrix(axes[1], R, "R = D V (reduced; pivots in orange)")
    draw_matrix(axes[2], V, "V (records the additions)", highlight_pivots=False)
    save(fig, "reduction.svg")


def fig_reduction_step():
    """Reducing column ac: its pivot c is owned by bc, then b by ab, until nothing is left."""
    steps = [({0, 2}, "ac = {a, c}", "pivot c is owned by bc: add bc"),
             ({0, 1}, "{a, b}", "pivot b is owned by ab: add ab"),
             (set(), "0", "zero: ac is born and never dies here")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    rows = ["a", "b", "c"]
    for ax, (col, label, note) in zip(axes, steps):
        for i in range(3):
            filled = i in col
            is_pivot = col and i == max(col)
            ax.add_patch(Rectangle((0, i), 1, 1, facecolor=ORANGE if is_pivot else ("#dfe9f7" if filled else "white"),
                                   edgecolor="#bbbbbb"))
            ax.text(0.5, i + 0.5, "1" if filled else "", ha="center", va="center",
                    color="white" if is_pivot else INK, fontweight="bold")
            ax.text(-0.25, i + 0.5, rows[i], ha="right", va="center")
        ax.set_xlim(-0.8, 1.2)
        ax.set_ylim(3.2, -0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, -0.35, label, ha="center", fontsize=11)
        ax.text(0.5, 3.6, note, ha="center", fontsize=9.5, wrap=True)
    for k in range(2):
        fig.add_artist(FancyArrowPatch((0.29 + 0.33 * k, 0.5), (0.37 + 0.33 * k, 0.5), transform=fig.transFigure,
                                       arrowstyle="->", mutation_scale=18, color=INK))
    save(fig, "reduction_step.svg")


def fig_clearing():
    """Homology with clearing: reduce dimension 2 first; its pair (ac, abc) clears column ac."""
    cols = [set(c) for c in BOUNDARY]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    draw_matrix(axes[0], cols, "1. Reduce dimension 2: abc has pivot ac", shade_cols=[9],
                mark={(5, 9): RED}, highlight_pivots=False)
    cleared_R = [set(c) for c in R]
    cleared_R[5] = set()
    draw_matrix(axes[1], cleared_R, "2. So column ac is a cycle: clear it, unreduced",
                cleared=[5], shade_cols=[3, 4, 7, 8])
    save(fig, "clearing.svg")


def antitranspose(columns, n):
    """Columns of the anti-transpose: entry (i, j) moves to (n-1-j, n-1-i)."""
    out = [set() for _ in range(n)]
    for j, col in enumerate(columns):
        for i in col:
            out[n - 1 - i].add(n - 1 - j)
    return out


def fig_antitranspose():
    names = [c[0] for c in CELLS]
    rev = names[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    draw_matrix(axes[0], BOUNDARY, "D: boundary, cells in filtration order", highlight_pivots=False)
    at = antitranspose(BOUNDARY, N)
    draw_matrix(axes[1], at, "D⊥: coboundary, cells in reverse order",
                row_labels=rev, col_labels=rev, highlight_pivots=False, entry_colour=PURPLE)
    save(fig, "antitranspose.svg")


def fig_apparent():
    """Every pair of the example is apparent: the pivot r of column c, with c the first column in row r."""
    marks = {}
    for r, c in PAIRS:
        first = min(j for j in range(N) if r in BOUNDARY[j])
        marks[(r, c)] = GREEN if first == c else RED
    fig, ax = plt.subplots(figsize=(5.4, 5))
    draw_matrix(ax, BOUNDARY, "Apparent pairs (boxed): the pivot is the\nfirst entry in its row", mark=marks)
    save(fig, "apparent.svg")


def fig_store_r():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)

    def column(x, y, h, label, colour, fill=0.3):
        ax.add_patch(Rectangle((x, y), 0.45, h, facecolor=colour, alpha=fill, edgecolor=colour))
        ax.text(x + 0.22, y - 0.25, label, ha="center", fontsize=10)

    ax.text(2.4, 3.15, "Store V (default): recompute R_j = D V_j", ha="center", fontsize=11, fontweight="bold")
    for k, x in enumerate([0.6, 1.4, 2.2, 3.0, 3.8]):
        column(x, 0.6, 1.6 + 0.2 * (k % 2), f"D_{k + 1}", BLUE)
        if k:
            ax.text(x - 0.2, 1.4, "+", ha="center", fontsize=14)
    ax.text(2.4, 2.65, "V_j = e_1 + e_2 + e_3 + e_4 + e_5: add five columns of D", ha="center", fontsize=9.5)
    ax.text(7.6, 3.15, "Store R (store_r): add R_j once", ha="center", fontsize=11, fontweight="bold")
    column(7.35, 0.6, 1.8, "R_j", ORANGE, 0.45)
    ax.text(7.6, 2.65, "one column, kept from when j was reduced", ha="center", fontsize=9.5)
    ax.plot([5.1, 5.1], [0.3, 3.2], color="#cccccc", lw=1)
    save(fig, "store_r.svg")


def fig_involution():
    """Cohomology pairs, then a homology reduction restricted to the deaths and essential births."""
    keep = sorted({d for _, d in PAIRS} | set(ESSENTIAL))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    shade = keep
    draw_matrix(ax, BOUNDARY, "The involution reduces only the shaded columns\n(deaths and essential births)",
                shade_cols=shade, highlight_pivots=False)
    save(fig, "involution.svg")


def fig_heap():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.2)
    # A binary-ish heap of entries
    nodes = {0: (4.5, 2.6, "9"), 1: (3.0, 1.7, "7"), 2: (6.0, 1.7, "8"), 3: (2.2, 0.8, "7"), 4: (3.8, 0.8, "3"),
             5: (5.4, 0.8, "5"), 6: (6.6, 0.8, "2")}
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=MUTED, zorder=0)
    for k, (x, y, label) in nodes.items():
        colour = ORANGE if k == 0 else (RED if label == "7" else BLUE)
        ax.scatter([x], [y], s=650, color=colour, zorder=1)
        ax.text(x, y, label, ha="center", va="center", color="white", fontweight="bold")
    ax.text(0.2, 2.6, "every term of every\ncolumn added is pushed", fontsize=10, va="center")
    ax.text(7.2, 2.6, "pop the top (the pivot);\nequal rows (7, red) are\nsummed as they surface", fontsize=10, va="center")
    save(fig, "heap.svg")


def fig_merged():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    runs = [[9, 6, 4, 1], [8, 6, 2], [7, 5, 3, 2, 0]]
    cursors = [1, 0, 0]
    for r, (run, cur) in enumerate(zip(runs, cursors)):
        y = 2.9 - r * 1.0
        ax.text(0.15, y + 0.2, f"run {r + 1}", fontsize=10, va="center")
        for k, v in enumerate(run):
            colour = MUTED if k < cur else (ORANGE if k == cur else BLUE)
            ax.add_patch(Rectangle((1.2 + 0.6 * k, y), 0.5, 0.42, facecolor=colour, alpha=0.85 if k >= cur else 0.4))
            ax.text(1.45 + 0.6 * k, y + 0.21, str(v), ha="center", va="center", color="white", fontweight="bold")
        ax.annotate("", xy=(1.45 + 0.6 * cur, y + 0.47), xytext=(1.45 + 0.6 * cur, y + 0.78),
                    arrowprops=dict(arrowstyle="->", color=INK))
    hx, hy = 6.8, 2.2
    ax.text(hx + 0.8, 3.35, "heap of cursors (one per run)", ha="center", fontsize=10)
    for k, (x, y, v) in enumerate([(hx + 0.8, 2.6, "8"), (hx + 0.2, 1.6, "7"), (hx + 1.4, 1.6, "6")]):
        ax.scatter([x], [y], s=600, color=ORANGE if k == 0 else BLUE, zorder=2)
        ax.text(x, y, v, ha="center", va="center", color="white", fontweight="bold", zorder=3)
    ax.plot([hx + 0.8, hx + 0.2], [2.6, 1.6], color=MUTED, zorder=1)
    ax.plot([hx + 0.8, hx + 1.4], [2.6, 1.6], color=MUTED, zorder=1)
    ax.text(hx + 0.8, 0.7, "pivot = top cursor;\nterms below it are never read", ha="center", fontsize=10)
    save(fig, "merged.svg")


def fig_sorted():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.8)
    sorted_vec = [0, 1, 3, 4, 6, 9]
    pending = [7, 5]
    ax.text(0.1, 2.25, "sorted (ascending)", fontsize=10)
    for k, v in enumerate(sorted_vec):
        moved = v > 4
        ax.add_patch(Rectangle((2.2 + 0.6 * k, 2.0), 0.5, 0.45, facecolor=ORANGE if moved else BLUE, alpha=0.85))
        ax.text(2.45 + 0.6 * k, 2.22, str(v), ha="center", va="center", color="white", fontweight="bold")
    ax.text(0.1, 1.25, "pending (unsorted)", fontsize=10)
    for k, v in enumerate(pending):
        ax.add_patch(Rectangle((2.2 + 0.6 * k, 1.0), 0.5, 0.45, facecolor=GREEN, alpha=0.85))
        ax.text(2.45 + 0.6 * k, 1.22, str(v), ha="center", va="center", color="white", fontweight="bold")
    ax.text(6.3, 1.9, "on the next pivot query, sort the pending terms\nand merge them in from the smallest (5):\n"
            "terms below it (blue) stay put, the rest are\nrewritten (orange). The pivot is the last term.",
            fontsize=9.5, va="center")
    save(fig, "sorted.svg")


def fig_bit_tree():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.text(0.1, 3.2, "level 1: one bit per word below", fontsize=10)
    ax.text(0.1, 1.95, "level 0: one bit per rank (64 per word)", fontsize=10)
    top = [1, 0, 1, 1, 0, 0, 0, 1]
    for k, b in enumerate(top):
        ax.add_patch(Rectangle((1.0 + 1.0 * k, 2.6), 0.9, 0.35, facecolor=ORANGE if b else PALE, edgecolor="#bbbbbb"))
        ax.text(1.45 + 1.0 * k, 2.77, str(b), ha="center", va="center", fontsize=9)
    import random

    rng = random.Random(3)
    for k, b in enumerate(top):
        for m in range(8):
            bit = b and rng.random() < 0.35
            ax.add_patch(Rectangle((1.0 + 1.0 * k + 0.11 * m, 1.35), 0.1, 0.35,
                                   facecolor=BLUE if bit else PALE, edgecolor="#cccccc", lw=0.4))
        ax.plot([1.45 + 1.0 * k, 1.45 + 1.0 * k], [2.6, 1.72], color="#cccccc", lw=0.8)
    ax.text(5.0, 0.6, "add an entry: flip its bit (equal entries cancel at once).  pivot: the highest set bit,\n"
            "found by a bit-scan at each level, top down. A slot array beside the bits holds each entry.",
            ha="center", fontsize=9.5)
    save(fig, "bit_tree.svg")


def fig_bookkeeping():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    titles = ["hashed (sparse keys, e.g. simplices)", "slotted (dense keys, big values)", "inline (dense keys, small values)"]
    for k, title in enumerate(titles):
        x = 0.2 + 3.3 * k
        ax.text(x, 3.45, title, fontsize=10, fontweight="bold")
    # hashed
    for m in range(6):
        ax.add_patch(Rectangle((0.2 + 0.45 * m, 2.6), 0.4, 0.4, facecolor=BLUE if m in (1, 4) else PALE))
    ax.text(0.2, 2.2, "hash the key, probe a table", fontsize=9)
    # slotted
    for m in range(6):
        ax.add_patch(Rectangle((3.5 + 0.45 * m, 2.6), 0.4, 0.4, facecolor=BLUE if m in (0, 3) else PALE))
    for m in range(2):
        ax.add_patch(Rectangle((3.5 + 0.9 * m, 1.6), 0.8, 0.4, facecolor=ORANGE))
    ax.annotate("", xy=(3.9, 2.02), xytext=(3.7, 2.58), arrowprops=dict(arrowstyle="->", color=INK))
    ax.annotate("", xy=(4.8, 2.02), xytext=(5.05, 2.58), arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(3.5, 1.2, "slots[key] -> entries: two accesses", fontsize=9)
    # inline
    for m in range(6):
        ax.add_patch(Rectangle((6.8 + 0.45 * m, 2.6), 0.4, 0.4, facecolor=ORANGE if m in (0, 3) else PALE))
    ax.text(6.8, 2.2, "values[key]: one access", fontsize=9)
    ax.text(5.0, 0.4, "phlite picks the storage from the key type (DENSE_INDEX) and the value's size, at compile time",
            ha="center", fontsize=9.5)
    save(fig, "bookkeeping.svg")


def fig_ranks():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    values = [3, 0, 2, 0, 1, 2]
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ax.text(0.1, 2.4, "input cells\n(value)", fontsize=10, va="center")
    ax.text(0.1, 0.6, "by rank", fontsize=10, va="center")
    for i, v in enumerate(values):
        ax.add_patch(Rectangle((2.0 + 1.2 * i, 2.2), 0.9, 0.45, facecolor=BLUE, alpha=0.85))
        ax.text(2.45 + 1.2 * i, 2.42, f"{i} ({v})", ha="center", va="center", color="white", fontsize=9.5)
    for r, i in enumerate(order):
        ax.add_patch(Rectangle((2.0 + 1.2 * r, 0.4), 0.9, 0.45, facecolor=ORANGE, alpha=0.85))
        ax.text(2.45 + 1.2 * r, 0.62, f"r{r} = {i}", ha="center", va="center", color="white", fontsize=9.5)
        ax.annotate("", xy=(2.45 + 1.2 * r, 0.88), xytext=(2.45 + 1.2 * i, 2.18),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    save(fig, "ranks.svg")


def fig_cubical():
    """The tutorial's cubical complex: a 3x3 image, cells at the points of a 5x5 grid."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    n = 5
    for i in range(n):
        for j in range(n):
            dim = (i % 2) + (j % 2)
            x, y = j, n - 1 - i
            if dim == 2:
                ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, facecolor=GREEN, alpha=0.3))
            elif dim == 1:
                if i % 2:
                    ax.plot([x, x], [y - 0.45, y + 0.45], color=INK, lw=2.5)
                else:
                    ax.plot([x - 0.45, x + 0.45], [y, y], color=INK, lw=2.5)
            else:
                ax.scatter([x], [y], s=120, color=BLUE, zorder=3)
            ax.text(x + 0.12, y + 0.12, str(i * n + j), fontsize=8, color=PURPLE, zorder=4)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Cells of a 3x3 image, numbered row by row\n(vertex: both coordinates even; square: both odd)",
                 fontsize=10)
    save(fig, "cubical.svg")


if __name__ == "__main__":
    fig_filtration()
    fig_barcode()
    fig_reduction()
    fig_reduction_step()
    fig_clearing()
    fig_antitranspose()
    fig_apparent()
    fig_store_r()
    fig_involution()
    fig_heap()
    fig_merged()
    fig_sorted()
    fig_bit_tree()
    fig_bookkeeping()
    fig_ranks()
    fig_cubical()
