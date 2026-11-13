import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # A filtration on your own simplices with `phlite`

        The Vietoris–Rips complex of \(n\) points has \(\binom{n}{3}\) triangles. In the plane the **Delaunay
        triangulation** has only about \(2n\), and filtering it by edge length still finds the loops of a point cloud
        (this is close to the alpha complex). We build it with scipy, hand the simplices to `phlite` as tuples, and
        draw the representative loops on the points.
        """
    )
    return (mo,)


@app.cell
def _():
    import itertools

    import altair as alt
    import numpy as np
    import phlite
    import polars as pl
    from scipy.spatial import Delaunay

    return Delaunay, alt, itertools, np, phlite, pl


@app.cell
def _(np, pl):
    # Points on two noisy circles of different sizes
    rng = np.random.default_rng(1)

    def circle(centre, radius, n):
        angles = rng.uniform(0, 2 * np.pi, n)
        return np.c_[np.cos(angles), np.sin(angles)] * radius + centre + rng.normal(scale=0.08, size=(n, 2))

    points = np.vstack([circle((0, 0), 1.0, 150), circle((2.6, 0.3), 0.6, 80)])
    point_frame = pl.DataFrame({"x": points[:, 0], "y": points[:, 1]})
    return point_frame, points


@app.cell
def _(Delaunay, itertools, np, pl, points):
    triangles = [tuple(sorted(t)) for t in Delaunay(points).simplices.tolist()]
    edges = sorted({e for t in triangles for e in itertools.combinations(t, 2)})
    simplices = [(v,) for v in range(len(points))] + edges + triangles
    edge_frame = pl.DataFrame(
        {
            "x": points[[a for a, _ in edges], 0],
            "y": points[[a for a, _ in edges], 1],
            "x2": points[[b for _, b in edges], 0],
            "y2": points[[b for _, b in edges], 1],
            "length": [float(np.linalg.norm(points[a] - points[b])) for a, b in edges],
        }
    )
    pl.DataFrame({"points": [len(points)], "edges": [len(edges)], "triangles": [len(triangles)]})
    return edge_frame, edges, simplices


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The chain complex and its filtration

        Simplices are tuples of vertices; their boundary is the usual alternating sum, which we give as a mapping from
        faces to coefficients (here over \(\mathbb{Z}/3\), to show that coefficients other than \(\pm 1\) work too).
        Edges get their length; vertices appear at 0, and triangles with their longest edge (`extend=True`).
        """
    )
    return


@app.cell
def _(edge_frame, edges, phlite, pl, points, simplices):
    def boundary(simplex):
        if len(simplex) == 1:
            return {}
        return {simplex[:i] + simplex[i + 1 :]: (-1) ** i for i in range(len(simplex))}

    chain = phlite.ChainComplex.from_cells(simplices, dimension=lambda s: len(s) - 1, boundary=boundary, modulus=3)
    values = {(v,): 0.0 for v in range(len(points))}
    values.update(zip(edges, edge_frame["length"].to_list()))
    barcode = chain.filtration(values, extend=True).persistence(1, representatives=True)
    bars = pl.DataFrame(
        [{"dimension": f"H{b.dimension}", "birth": b.birth_value, "death": b.death_value} for b in barcode.bars]
    )
    barcode
    return bars, barcode


@app.cell
def _(alt, bars, pl):
    _finite = bars.filter(pl.col("death").is_finite())
    _top = _finite["death"].max()
    _diagonal = alt.Chart(pl.DataFrame({"t": [0.0, _top]})).mark_line(color="grey", strokeWidth=1).encode(
        x="t:Q", y="t:Q"
    )
    (
        _diagonal
        + alt.Chart(_finite)
        .mark_circle(size=40)
        .encode(
            x=alt.X("birth:Q", title="birth (edge length)"),
            y=alt.Y("death:Q", title="death"),
            color=alt.Color("dimension:N", title=None),
            tooltip=["dimension", "birth", "death"],
        )
    ).properties(width=320, height=320, title="persistence diagram")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Two H1 points stand far from the diagonal: the two circles. Here are the edges of the complex at a scale where
        both loops are present, with their representatives (lists of `(edge, coefficient)`) drawn on top.
        """
    )
    return


@app.cell
def _(alt, barcode, edge_frame, pl, point_frame, points):
    loops = sorted((b for b in barcode.bars if b.dimension == 1), key=lambda b: b.persistence, reverse=True)[:2]
    scale = max(b.birth_value for b in loops) + 0.01
    _segments = pl.DataFrame(
        [
            {
                "loop": f"[{_b.birth_value:.2f}, {_b.death_value:.2f})",
                "x": points[_e[0], 0],
                "y": points[_e[0], 1],
                "x2": points[_e[1], 0],
                "y2": points[_e[1], 1],
            }
            for _b in loops
            for _e, _coefficient in _b.representative
        ]
    )
    _axis = dict(axis=None, scale=alt.Scale(zero=False))
    _complex = alt.Chart(edge_frame.filter(pl.col("length") <= scale)).mark_rule(color="#bbbbbb").encode(
        x=alt.X("x:Q", **_axis), y=alt.Y("y:Q", **_axis), x2="x2:Q", y2="y2:Q"
    )
    _loops = alt.Chart(_segments).mark_rule(strokeWidth=3).encode(
        x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q", color=alt.Color("loop:N", title="H1 bar")
    )
    _points = alt.Chart(point_frame).mark_circle(size=12, color="black").encode(x="x:Q", y="y:Q")
    (_complex + _loops + _points).properties(
        width=560, height=300, title=f"edges of length at most {scale:.2f}, and the two longest H1 representatives"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watching the filtration

        At scale \(s\) the complex has every edge of length at most \(s\), and every triangle whose edges are all
        there. Components merge as edges appear; each circle closes into a loop, which is filled in once its triangles
        span the middle. On the right, the bars alive at \(s\) are highlighted.
        """
    )
    return


@app.cell
def _(bars, edge_frame, mo, np, pl, points, triangles):
    import base64
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection, PolyCollection

    def animate(draw, times, figsize=(10, 3.8), fps=6):
        """An animated GIF of `draw(axes, t)` over `times`, as an image to display."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]})
        animation = FuncAnimation(fig, lambda t: draw(axes, t), frames=times)
        with tempfile.NamedTemporaryFile(suffix=".gif") as file:
            animation.save(file.name, writer=PillowWriter(fps=fps), dpi=72)
            plt.close(fig)
            data = base64.b64encode(open(file.name, "rb").read()).decode()
        return mo.Html(f'<img src="data:image/gif;base64,{data}" style="max-width: 100%" />')

    def draw_barcode(ax, frame, t, colours, top):
        """Bars as horizontal lines, those alive at `t` in colour, the rest faded, and a line at `t`."""
        ax.clear()
        for y, (dimension, birth, death) in enumerate(frame.iter_rows()):
            end = death if np.isfinite(death) else top
            alive = birth <= t < death
            ax.plot([birth, end], [y, y], color=colours[dimension], lw=2.2, alpha=1.0 if alive else 0.15)
        ax.axvline(t, color="black", lw=1)
        ax.set_yticks([])
        ax.set_xlim(0, top)
        ax.set_title(f"barcode, s = {t:.2f}")

    _lengths = edge_frame["length"].to_numpy()
    _segments = np.stack([edge_frame.select("x", "y").to_numpy(), edge_frame.select("x2", "y2").to_numpy()], axis=1)
    _triangle_values = np.array(
        [max(np.linalg.norm(points[a] - points[b]) for a, b in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]) for t in triangles]
    )
    _shown = bars.filter((pl.col("death") - pl.col("birth")) > 0.05).sort("dimension", "birth")
    _top = float(_shown.filter(pl.col("death").is_finite())["death"].max()) * 1.1

    def _draw(axes, t):
        ax = axes[0]
        ax.clear()
        ax.add_collection(
            PolyCollection([points[list(tri)] for tri, v in zip(triangles, _triangle_values) if v <= t],
                           facecolor="tab:green", alpha=0.25)
        )
        ax.add_collection(LineCollection(_segments[_lengths <= t], color="#555555", lw=0.6))
        ax.scatter(points[:, 0], points[:, 1], s=5, color="black", zorder=3)
        ax.set_xlim(points[:, 0].min() - 0.2, points[:, 0].max() + 0.2)
        ax.set_ylim(points[:, 1].min() - 0.2, points[:, 1].max() + 0.2)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("the complex at scale s")
        draw_barcode(axes[1], _shown.select("dimension", "birth", "death"), t,
                     {"H0": "tab:blue", "H1": "tab:orange"}, _top)

    animate(_draw, np.linspace(0, _top, 44))
    return


if __name__ == "__main__":
    app.run()
