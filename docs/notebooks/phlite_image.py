import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Persistent homology of an image with `phlite`

        We build the cubical complex of a greyscale image ourselves, as Python cells, attach the **lower-star
        filtration** (a cell appears with its brightest pixel), and read off the barcode and a **representative loop**
        for each ring in the image.
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

    return alt, itertools, np, phlite, pl


@app.cell
def _(mo, np):
    # An image with three dark rings, plus a little noise
    size = 64
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    rings = [((18, 20), 10), ((44, 22), 7), ((32, 46), 13)]
    image = np.full((size, size), np.inf)
    for (cy, cx), radius in rings:
        distance = np.hypot(yy - cy, xx - cx)
        image = np.minimum(image, np.abs(distance - radius))
    image = np.minimum(image, 8) + rng.normal(scale=0.4, size=image.shape)
    mo.md(f"A {size} × {size} image: dark (low) pixels on three rings.")
    return image, size


@app.cell
def _(alt, image, np, pl, size):
    _row, _col = np.divmod(np.arange(size * size), size)
    pixels = pl.DataFrame({"row": _row, "col": _col, "value": image.ravel()})

    def picture(pixels):
        """The image as a heatmap, in pixel coordinates (row down, column across)."""
        return (
            alt.Chart(pixels)
            .transform_calculate(x="datum.col - 0.5", x2="datum.col + 0.5", y="datum.row - 0.5", y2="datum.row + 0.5")
            .mark_rect()
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.5, size - 0.5], nice=False)),
                x2="x2:Q",
                y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.5, size - 0.5], nice=False, reverse=True)),
                y2="y2:Q",
                color=alt.Color("value:Q", scale=alt.Scale(scheme="greys", reverse=True), legend=None),
            )
            .properties(width=320, height=320)
        )

    picture(pixels).properties(title="the image")
    return picture, pixels


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The cubical complex, as Python cells

        Pixels are vertices; cells sit on a grid twice as fine, at `(r, c)`: a vertex where both coordinates are even,
        an edge where one is odd, a square where both are. A cell's faces are one step either way along each odd
        coordinate. We describe cells as plain tuples and tell `phlite` how to get their dimension and boundary.

        Cells with equal values enter in the order they are listed, so every face must be listed before its cofaces:
        we list vertices, then edges, then squares.
        """
    )
    return


@app.cell
def _(itertools, phlite, size):
    side = 2 * size - 1

    def dimension(cell):
        return cell[0] % 2 + cell[1] % 2

    def boundary(cell):
        r, c = cell
        faces = []
        if r % 2:
            faces += [(r - 1, c), (r + 1, c)]
        if c % 2:
            faces += [(r, c - 1), (r, c + 1)]
        return faces

    cells = sorted(itertools.product(range(side), range(side)), key=dimension)
    chain = phlite.ChainComplex.from_cells(cells, dimension=dimension, boundary=boundary)
    chain
    return (chain,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The lower-star filtration

        We only give values to the vertices (pixels); `extend=True` gives every other cell the largest value of its
        faces, which is exactly the lower-star filtration.
        """
    )
    return


@app.cell
def _(chain, image, pl, size):
    pixel_values = {(2 * i, 2 * j): float(image[i, j]) for i in range(size) for j in range(size)}
    barcode = chain.filtration(pixel_values, extend=True).persistence(representatives=True)
    bars = pl.DataFrame(
        [
            {"dimension": f"H{b.dimension}", "birth": b.birth_value, "death": b.death_value}
            for b in barcode.bars
        ]
    ).with_columns(persistence=pl.col("death") - pl.col("birth"))
    barcode
    return bars, barcode


@app.cell
def _(alt, bars, pl):
    _finite = bars.filter(pl.col("death").is_finite())
    _diagonal = alt.Chart(pl.DataFrame({"t": [_finite["birth"].min(), _finite["death"].max()]})).mark_line(
        color="grey", strokeWidth=1
    ).encode(x="t:Q", y="t:Q")
    _diagram = _diagonal + alt.Chart(_finite).mark_circle(size=30).encode(
        x=alt.X("birth:Q", title="birth"),
        y=alt.Y("death:Q", title="death"),
        color=alt.Color("dimension:N", title=None),
        tooltip=["dimension", "birth", "death"],
    ).properties(width=360, height=360, title="persistence diagram")

    def barcode_panel(dimension):
        """The bars of one dimension longer than 0.25, sorted by birth (essential bars run off the right)."""
        frame = (
            bars.filter((pl.col("dimension") == dimension) & (pl.col("persistence") > 0.25))
            .with_columns(end=pl.when(pl.col("death").is_finite()).then(pl.col("death")).otherwise(9.5))
            .sort("birth", "death")
            .with_row_index("bar")
        )
        return (
            alt.Chart(frame)
            .mark_rule(strokeWidth=1.2)
            .encode(
                x=alt.X("birth:Q", title="filtration value", scale=alt.Scale(domain=[-1.5, 9.5])),
                x2="end:Q",
                y=alt.Y("bar:O", axis=None),
                color=alt.Color("dimension:N", title=None),
                tooltip=["dimension", "birth", "death"],
            )
            .properties(width=300, height=230, title=f"{dimension} barcode ({len(frame)} bars longer than 0.25)")
        )

    _diagram | (barcode_panel("H0") & barcode_panel("H1"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Three H1 bars stand out: the three rings, each born when its ring of dark pixels closes up and dying when its
        brighter centre fills in. Their **representatives** are loops of edges, which we can draw on the image (an edge
        between grid points `(r, c)` sits between the pixels at half those coordinates).
        """
    )
    return


@app.cell
def _(alt, barcode, picture, pixels, pl):
    loops = sorted((b for b in barcode.bars if b.dimension == 1), key=lambda b: b.persistence, reverse=True)[:3]
    _segments = []
    for _bar in loops:
        _label = f"[{_bar.birth_value:.2f}, {_bar.death_value:.2f})"
        for (_r, _c), _coefficient in _bar.representative:
            # The edge's two pixels
            if _r % 2:
                (_y0, _x0), (_y1, _x1) = ((_r - 1) / 2, _c / 2), ((_r + 1) / 2, _c / 2)
            else:
                (_y0, _x0), (_y1, _x1) = (_r / 2, (_c - 1) / 2), (_r / 2, (_c + 1) / 2)
            _segments.append({"loop": _label, "x": _x0, "y": _y0, "x2": _x1, "y2": _y1})
    _edges = alt.Chart(pl.DataFrame(_segments)).mark_rule(strokeWidth=2.5).encode(
        x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q", color=alt.Color("loop:N", title="H1 bar")
    )
    (picture(pixels) + _edges).properties(title="representatives of the three longest H1 bars")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watching the filtration

        The sublevel set at time \(t\) is every pixel darker than \(t\) (with the edges and squares between them). As
        \(t\) grows, components (H0) appear in the dark pixels and merge, and loops (H1) close up around each ring and
        are filled in when the brighter centre arrives. On the right, the bars alive at \(t\) are highlighted.
        """
    )
    return


@app.cell
def _(bars, image, mo, np, pl):
    import base64
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def animate(draw, times, figsize=(9, 3.8), fps=6):
        """An animated GIF of `draw(axes, t)` over `times`, as an image to display."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.2]})
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
        ax.set_xlim(frame["birth"].min() - 0.3, top)
        ax.set_title(f"barcode, t = {t:.2f}")

    _shown = (
        bars.filter(pl.col("persistence") > 0.6)
        .sort("dimension", "birth")
        .select("dimension", "birth", "death")
    )
    _colours = {"H0": "tab:blue", "H1": "tab:orange"}

    def _draw(axes, t):
        axes[0].clear()
        axes[0].imshow(np.where(image <= t, image, np.nan), cmap="gray", vmin=image.min(), vmax=image.max())
        axes[0].set_facecolor("white")
        axes[0].set_axis_off()
        axes[0].set_title("pixels below t")
        draw_barcode(axes[1], _shown, t, _colours, 9.5)

    animate(_draw, np.linspace(image.min(), 9.0, 48))
    return


if __name__ == "__main__":
    app.run()
