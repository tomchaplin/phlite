import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Vietoris–Rips persistence with `phlite_rips`

        A point cloud sampled from a noisy figure eight has two loops. We compute its Vietoris–Rips barcode up to
        dimension 1, with representatives, draw the loops on the points, and watch the filtration grow.
        """
    )
    return (mo,)


@app.cell
def _():
    import time

    import altair as alt
    import numpy as np
    import polars as pl
    from phlite_rips import Algorithm, rips

    return Algorithm, alt, np, pl, rips, time


@app.cell
def _(np, pl):
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 2 * np.pi, 300)
    # A lemniscate (figure eight), with noise
    points = np.c_[np.cos(t), np.sin(t) * np.cos(t)] / (1 + np.sin(t) ** 2)[:, None] * 2
    points += rng.normal(scale=0.04, size=points.shape)
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    point_frame = pl.DataFrame({"x": points[:, 0], "y": points[:, 1]})
    return distances, point_frame, points


@app.cell
def _(distances, pl, rips, time):
    start = time.perf_counter()
    barcode = rips(distances, max_dim=1, representatives=True)
    elapsed = time.perf_counter() - start
    bars = pl.DataFrame(
        [
            {"dimension": f"H{dim}", "birth": birth, "death": death}
            for dim in (0, 1)
            for birth, death in barcode.pairings[dim]
        ]
    ).with_columns(persistence=pl.col("death") - pl.col("birth"))
    barcode
    return bars, barcode, elapsed


@app.cell
def _(barcode, elapsed, mo):
    mo.md(
        f"""
        {len(barcode.pairings[0])} finite H0 bars and {len(barcode.pairings[1])} finite H1 bars (bars of length zero
        are omitted), computed with representatives in {elapsed * 1000:.0f} ms.
        """
    )
    return


@app.cell
def _(alt, bars, pl):
    _top = bars["death"].max()
    _diagonal = alt.Chart(pl.DataFrame({"t": [0.0, _top]})).mark_line(color="grey", strokeWidth=1).encode(
        x="t:Q", y="t:Q"
    )
    _diagram = (
        _diagonal
        + alt.Chart(bars)
        .mark_circle(size=40)
        .encode(
            x=alt.X("birth:Q"),
            y=alt.Y("death:Q"),
            color=alt.Color("dimension:N", title=None),
            tooltip=["dimension", "birth", "death"],
        )
    ).properties(width=320, height=320, title="persistence diagram")

    def barcode_panel(dimension):
        frame = bars.filter(pl.col("dimension") == dimension).sort("birth", "death").with_row_index("bar")
        return (
            alt.Chart(frame)
            .mark_rule(strokeWidth=1.5)
            .encode(
                x=alt.X("birth:Q", title="distance", scale=alt.Scale(domain=[0, _top])),
                x2="death:Q",
                y=alt.Y("bar:O", axis=None),
                color=alt.Color("dimension:N", title=None),
                tooltip=["dimension", "birth", "death"],
            )
            .properties(width=300, height=140, title=f"{dimension} barcode ({len(frame)} bars)")
        )

    _diagram | (barcode_panel("H0") & barcode_panel("H1"))
    return


@app.cell
def _(alt, barcode, np, pl, point_frame, points):
    _order = np.argsort([b - d for b, d in barcode.pairings[1]])[:2]
    _segments = pl.DataFrame(
        [
            {
                "loop": f"[{barcode.pairings[1][_i][0]:.2f}, {barcode.pairings[1][_i][1]:.2f})",
                "x": points[_a, 0],
                "y": points[_a, 1],
                "x2": points[_b, 0],
                "y2": points[_b, 1],
            }
            for _i in _order
            for _a, _b in barcode.pairing_representatives[1][_i]
        ]
    )
    _axis = dict(axis=None, scale=alt.Scale(zero=False))
    _points = alt.Chart(point_frame).mark_circle(size=12, color="black").encode(
        x=alt.X("x:Q", **_axis), y=alt.Y("y:Q", **_axis)
    )
    _loops = alt.Chart(_segments).mark_rule(strokeWidth=3).encode(
        x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q", color=alt.Color("loop:N", title="H1 bar")
    )
    (_points + _loops).properties(width=560, height=280, title="representatives of the two longest H1 bars")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watching the filtration

        At radius \(r\) the Vietoris–Rips complex has an edge between every two points at distance at most \(r\) (and
        a simplex on every set of points that are pairwise that close; only the edges are drawn). Components merge
        quickly; each lobe of the figure eight closes into a loop, which is filled in once the points across it are
        within \(r\) of each other. On the right, the bars alive at \(r\) are highlighted.
        """
    )
    return


@app.cell
def _(bars, distances, mo, np, pl, points):
    import base64
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

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
            alive = birth <= t < death
            ax.plot([birth, death], [y, y], color=colours[dimension], lw=2.2, alpha=1.0 if alive else 0.15)
        ax.axvline(t, color="black", lw=1)
        ax.set_yticks([])
        ax.set_xlim(0, top)
        ax.set_title(f"barcode, r = {t:.2f}")

    _i, _j = np.triu_indices(len(points), 1)
    _lengths = distances[_i, _j]
    _segments = np.stack([points[_i], points[_j]], axis=1)
    _shown = bars.filter(pl.col("persistence") > 0.03).sort("dimension", "birth")
    _top = float(_shown["death"].max()) * 1.1

    def _draw(axes, t):
        ax = axes[0]
        ax.clear()
        ax.add_collection(LineCollection(_segments[_lengths <= t], color="#555555", lw=0.4, alpha=0.6))
        ax.scatter(points[:, 0], points[:, 1], s=5, color="black", zorder=3)
        ax.set_xlim(points[:, 0].min() - 0.1, points[:, 0].max() + 0.1)
        ax.set_ylim(points[:, 1].min() - 0.1, points[:, 1].max() + 0.1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("edges of the Rips complex at radius r")
        draw_barcode(axes[1], _shown.select("dimension", "birth", "death"), t,
                     {"H0": "tab:blue", "H1": "tab:orange"}, _top)

    animate(_draw, np.linspace(0, _top, 44))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Choosing an algorithm

        The default (`Algorithm.serial()`: clearing, heap columns, uncached) favours memory. Every configuration gives
        the same barcode; here are a few, timed:
        """
    )
    return


@app.cell
def _(Algorithm, distances, pl, rips, time):
    _configurations = {
        "serial (default)": dict(algorithm=Algorithm.serial()),
        "serial, merged columns, cached": dict(algorithm=Algorithm.serial().with_columns("merged").with_cache(True)),
        "parallel": dict(algorithm=Algorithm.parallel()),
        "homology": dict(method="homology"),
    }
    _rows = []
    for _name, _options in _configurations.items():
        _start = time.perf_counter()
        rips(distances, max_dim=1, **_options)
        _rows.append({"configuration": _name, "time (ms)": round((time.perf_counter() - _start) * 1000, 1)})
    pl.DataFrame(_rows)
    return


if __name__ == "__main__":
    app.run()
