import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Path homology of a digraph with `phlite_grpph`

        Grounded persistent path homology (GrPPH) tells apart cycles that an undirected method would treat alike:
        a long directed cycle lives longer than a short one, a square of two parallel paths dies quickly, and a
        bidirected edge is a small loop of its own. We build a digraph with a few of each, compute its barcode with
        representatives, and highlight them.
        """
    )
    return (mo,)


@app.cell
def _():
    import math

    import altair as alt
    import networkx as nx
    import polars as pl
    from phlite_grpph import grpph

    return alt, grpph, math, nx, pl


@app.cell
def _(nx):
    G = nx.DiGraph()
    # A directed cycle of length 6
    cycle = [f"c{i}" for i in range(6)]
    G.add_weighted_edges_from((cycle[i], cycle[(i + 1) % 6], 1.0) for i in range(6))
    # A directed 4-cycle, attached at c0
    square = ["c0", "s1", "s2", "s3"]
    G.add_weighted_edges_from((square[i], square[(i + 1) % 4], 1.0) for i in range(4))
    # Two paths from c3 to t2: a square, but not a directed cycle
    G.add_weighted_edges_from([("c3", "t1", 1.0), ("t1", "t2", 1.0), ("c3", "t3", 1.0), ("t3", "t2", 1.0)])
    # A bidirected edge
    G.add_weighted_edges_from([("t2", "b", 0.5), ("b", "t2", 0.5)])
    G
    return (G,)


@app.cell
def _(G, grpph, pl):
    barcode = grpph(G, representatives=True)
    pl.DataFrame(
        [{"dimension": "H0", "birth": b, "death": None} for b in barcode.essential[0]]
        + [{"dimension": "H1", "birth": b, "death": d} for b, d in barcode.pairings[1]]
    )
    return (barcode,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Four H1 bars, all born at 0 (in the grounded filtration the graph's own edges are there from the start), with
        very different lifetimes: the bidirected edge dies at 0.5, its weight; the two paths from `c3` to `t2`, which
        are not a directed cycle, at 1; the directed 4-cycle at 2 and the directed 6-cycle at 3. A directed cycle of
        \(n\) unit edges lives until \(n / 2\). Each representative is a list of edges:
        """
    )
    return


@app.cell
def _(G, alt, barcode, math, nx, pl):
    position = nx.kamada_kawai_layout(G.to_undirected())

    def edge_rows(edges, bend):
        """Each edge as a segment, shortened at both ends, with an arrowhead three quarters of the way along.
        Edges in both directions are offset sideways (`bend`) so that they don't overlap."""
        rows = []
        for a, b in edges:
            (x0, y0), (x1, y1) = position[a], position[b]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            ox, oy = -dy / length * bend, dx / length * bend
            if not G.has_edge(b, a):
                ox = oy = 0.0
            sx, sy = x0 + 0.12 * dx + ox, y0 + 0.12 * dy + oy
            ex, ey = x0 + 0.88 * dx + ox, y0 + 0.88 * dy + oy
            rows.append(
                {
                    "x": sx, "y": sy, "x2": ex, "y2": ey,
                    "hx": x0 + 0.75 * dx + ox, "hy": y0 + 0.75 * dy + oy,
                    # Vega's angle is clockwise from up; the triangle points up
                    "angle": (90 - math.degrees(math.atan2(dy, dx))) % 360,
                }
            )
        return pl.DataFrame(rows)

    nodes = pl.DataFrame([{"node": n, "x": x, "y": y} for n, (x, y) in position.items()])
    _axis = dict(axis=None, scale=alt.Scale(zero=False))

    def draw(edges, colour, title):
        frame = edge_rows(edges, 0.03)
        lines = alt.Chart(frame).mark_rule(color=colour, strokeWidth=2.5 if colour != "#bbbbbb" else 1.2).encode(
            x=alt.X("x:Q", **_axis), y=alt.Y("y:Q", **_axis), x2="x2:Q", y2="y2:Q"
        )
        heads = alt.Chart(frame).mark_point(shape="triangle", filled=True, size=60, color=colour).encode(
            x="hx:Q", y="hy:Q", angle=alt.Angle("angle:Q", scale=None)
        )
        return lines + heads

    _node_marks = alt.Chart(nodes).mark_circle(size=260, color="#dfe9f7", opacity=1).encode(x="x:Q", y="y:Q")
    _labels = alt.Chart(nodes).mark_text(fontSize=9).encode(x="x:Q", y="y:Q", text="node:N")
    _panels = []
    for (_birth, _death), _rep, _colour in zip(
        barcode.pairings[1], barcode.pairing_representatives[1], ["#c8453a", "#e07b39", "#3a9d5d", "#7d5ba6"]
    ):
        _panels.append(
            (draw(G.edges, "#bbbbbb", "") + draw([tuple(e) for e in _rep], _colour, "") + _node_marks + _labels)
            .properties(width=230, height=230, title=f"H1 bar [{_birth}, {_death})")
        )
    alt.hconcat(*_panels)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watching the filtration

        In the grounded filtration the graph's own edges are there from the start, and at time \(t\) a new edge
        \(u \to v\) appears (dashed) once the shortest path from \(u\) to \(v\) has length at most \(t\). Cycles
        are filled in by longer paths through the digraph. The bidirected edge's loop and the two parallel paths die
        (at 0.5 and 1) before any new edge appears: paths along the graph's own edges are enough to fill them. The
        directed 4- and 6-cycles need the new edges, and die at 2 and 3. On the right, the bars alive at \(t\) are
        highlighted.
        """
    )
    return


@app.cell
def _(G, barcode, mo, nx):
    import base64
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    def animate(draw, times, figsize=(10, 4), fps=4):
        """An animated GIF of `draw(axes, t)` over `times`, as an image to display."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]})
        animation = FuncAnimation(fig, lambda t: draw(axes, t), frames=times)
        with tempfile.NamedTemporaryFile(suffix=".gif") as file:
            animation.save(file.name, writer=PillowWriter(fps=fps), dpi=72)
            plt.close(fig)
            data = base64.b64encode(open(file.name, "rb").read()).decode()
        return mo.Html(f'<img src="data:image/gif;base64,{data}" style="max-width: 100%" />')

    _layout = nx.kamada_kawai_layout(G.to_undirected())
    _distance = dict(nx.all_pairs_dijkstra_path_length(G))
    _shortcuts = [(u, v, d) for u, row in _distance.items() for v, d in row.items() if u != v and not G.has_edge(u, v)]
    _bars = [(0, b, float("inf")) for b in barcode.essential[0]] + [(1, b, d) for b, d in barcode.pairings[1]]
    _colours = ["tab:blue", "tab:orange"]

    def _draw(axes, t):
        ax = axes[0]
        ax.clear()
        _new = [(u, v) for u, v, d in _shortcuts if d <= t]
        nx.draw_networkx_edges(G, _layout, edgelist=_new, ax=ax, edge_color="tab:green", style="dashed",
                               alpha=0.5, arrowsize=8, connectionstyle="arc3,rad=0.15")
        nx.draw_networkx(G, _layout, ax=ax, node_size=220, font_size=7, node_color="#dfe9f7",
                         edge_color="#444444", arrowsize=10, connectionstyle="arc3,rad=0.1")
        ax.set_axis_off()
        ax.set_title(f"the graph and its shortcuts of length at most t ({len(_new)})")
        ax = axes[1]
        ax.clear()
        for y, (dimension, birth, death) in enumerate(_bars):
            end = death if death != float("inf") else 3.5
            ax.plot([birth, end], [y, y], color=_colours[dimension], lw=4,
                    alpha=1.0 if birth <= t < death else 0.15)
        ax.axvline(t, color="black", lw=1)
        ax.set_xlim(-0.1, 3.5)
        ax.set_ylim(-0.6, len(_bars) - 0.4)
        ax.set_yticks(range(len(_bars)))
        ax.set_yticklabels([f"H{d}" for d, _, _ in _bars])
        ax.set_title(f"barcode, t = {t:.2f}")

    animate(_draw, np.linspace(0, 3.4, 35))
    return


if __name__ == "__main__":
    app.run()
