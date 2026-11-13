"""`phlite.Algorithm` and the registry of pipelines' defaults (`phlite.defaults()`), against each pipeline's docstring
and the book's table of defaults, so that neither can go stale."""

import re
from pathlib import Path

import numpy as np
import pytest

import phlite
from phlite import Algorithm

BOOK = Path(__file__).resolve().parents[2] / "docs" / "book" / "src" / "python" / "algorithms.md"


def pipeline_function(name):
    """The Python function of the pipeline `name` (`'cloud.rips'` is `phlite.cloud.rips`)."""
    if name == "complex":
        return phlite.FilteredComplex.persistence
    module, function = name.split(".")
    return getattr(getattr(phlite, module), function)


def python(value):
    return repr(value)


def test_registry_lists_the_pipelines():
    defaults = phlite.defaults()
    assert {"complex", "cloud.rips", "network.grpph"} <= set(defaults)
    for name, presets in defaults.items():
        assert set(presets) == {"serial", "parallel"}
        assert presets["serial"] == Algorithm.serial().resolve(name)
        assert presets["parallel"] == Algorithm.parallel().resolve(name)
        # Resolved presets set every option
        for algorithm in presets.values():
            assert algorithm.columns is not None and algorithm.cache is not None and algorithm.store_r is not None
            assert "<pipeline default>" not in repr(algorithm)
            assert algorithm.resolve(name) == algorithm


@pytest.mark.parametrize("name", list(phlite.defaults()))
def test_docstrings_state_the_defaults(name):
    """Each pipeline's docstring says `Pipeline defaults (`'name'`): columns='...', cache=..., store_r=...`, as its
    serial preset resolves."""
    serial = phlite.defaults()[name]["serial"]
    doc = " ".join(pipeline_function(name).__doc__.split())
    expected = (
        f"Pipeline defaults (`'{name}'`): columns={python(serial.columns)}, cache={python(serial.cache)}, "
        f"store_r={python(serial.store_r)}."
    )
    assert expected in doc


def book_table():
    """{pipeline: row} from the table between the `defaults` markers in the book."""
    text = BOOK.read_text()
    table = text.split("<!-- defaults:start -->")[1].split("<!-- defaults:end -->")[0]
    rows = {}
    for line in table.strip().splitlines()[2:]:
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def test_book_table_matches_the_registry():
    """The book's table: pipeline, function, then serial clearing's columns, cache and store R, and parallel
    clearing's columns and cache."""
    table = book_table()
    defaults = phlite.defaults()
    assert set(table) == set(defaults), "add a row to the table in docs/book/src/python/algorithms.md"
    for name, (function, columns, cache, store_r, parallel_columns, parallel_cache) in table.items():
        serial, parallel = defaults[name]["serial"], defaults[name]["parallel"]
        expected_function = "FilteredComplex.persistence" if name == "complex" else f"phlite.{name}"
        assert function == expected_function
        assert [columns, cache, store_r] == [python(serial.columns), python(serial.cache), python(serial.store_r)]
        assert [parallel_columns, parallel_cache] == [python(parallel.columns), python(parallel.cache)]


def test_unset_options_are_shown():
    assert repr(Algorithm.serial()) == (
        "Algorithm('clearing', columns=<pipeline default>, cache=<pipeline default>, store_r=<pipeline default>, "
        "skip_zero_apparent_pairs=True, dimension_order=None)"
    )
    algorithm = Algorithm.parallel(4).with_columns("sorted").with_cache(False)
    assert repr(algorithm) == (
        "Algorithm('parallel-clearing', columns='sorted', cache=False, skip_zero_apparent_pairs=True, "
        "dimension_order=None, storage='locking', threads=4)"
    )
    assert algorithm.with_columns(None).columns is None
    assert Algorithm.serial().columns is None and Algorithm.serial().with_cache(None) == Algorithm.serial()


def test_resolve():
    assert Algorithm.serial().resolve("cloud.rips").columns == "heap"
    assert Algorithm.serial().resolve("complex").columns == "pivot"
    # Only serial clearing uses pivot columns
    assert Algorithm.parallel().resolve("complex").columns == "merged"
    # Options set win
    assert Algorithm.serial().with_columns("sorted").resolve("complex").columns == "sorted"
    with pytest.raises(ValueError, match="Unknown pipeline 'nope'.*'cloud.rips'"):
        Algorithm.serial().resolve("nope")
    with pytest.raises(ValueError, match="row"):
        Algorithm.serial().with_columns("pivot").resolve("cloud.rips")


def test_options_must_suit_the_algorithm():
    for bad in [
        lambda: Algorithm.parallel().with_columns("pivot"),
        lambda: Algorithm.parallel().with_store_r(),
        lambda: Algorithm.serial().with_threads(2),
        lambda: Algorithm.standard().with_skip_zero_apparent_pairs(True),
        lambda: Algorithm.serial().with_storage("lock-free"),
        lambda: Algorithm.serial().with_columns("nope"),
    ]:
        with pytest.raises(ValueError):
            bad()


def test_barcodes_record_their_algorithm():
    rng = np.random.default_rng(0)
    points = rng.random((20, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    barcode = phlite.cloud.rips(distances)
    assert barcode.algorithm == Algorithm.serial().resolve("cloud.rips")
    assert repr(barcode.algorithm) in repr(barcode)
    parallel = Algorithm.parallel(2).with_cache(True)
    assert phlite.cloud.rips(distances, algorithm=parallel).algorithm == parallel.resolve("cloud.rips")

    grpph = phlite.network.grpph(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    assert grpph.algorithm == Algorithm.serial().resolve("network.grpph")

    chain = phlite.ChainComplex.from_pairs([(0, []), (0, []), (1, [0, 1])])
    barcode = chain.filtration([0.0, 1.0, 2.0]).persistence()
    assert barcode.algorithm == Algorithm.serial().resolve("complex")
    assert repr(barcode.algorithm) in repr(barcode)


def test_pipelines_refuse_what_they_cannot_run():
    with pytest.raises(ValueError, match="row"):
        phlite.cloud.rips([[0.0, 1.0], [1.0, 0.0]], algorithm=Algorithm.serial().with_columns("pivot"))
    with pytest.raises(TypeError):
        phlite.cloud.rips([[0.0, 1.0], [1.0, 0.0]], algorithm="clearing")


def test_submodules_import():
    import phlite.cloud
    import phlite.image
    import phlite.network
    from phlite.cloud import rips
    from phlite.network import grpph

    assert rips is phlite.cloud.rips and grpph is phlite.network.grpph
    assert rips.__module__ == "phlite.cloud" and grpph.__module__ == "phlite.network"
    assert Algorithm.__module__ == "phlite"
    assert {"cloud", "network", "image", "Algorithm", "ValueBarcode", "defaults"} <= set(phlite.__all__)
    assert isinstance(phlite.__version__, str)
