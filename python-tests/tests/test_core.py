"""`phlite.core`: `Algorithm`, the namespace package `phlite`, and the registry of pipelines' defaults
(`phlite.core.defaults()`), against each pipeline's docstring and the book's table of defaults, so that neither can go
stale."""

import importlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pytest

import phlite
import phlite.core
from phlite.core import Algorithm, Barcode, defaults

BOOK = Path(__file__).resolve().parents[2] / "docs" / "book" / "src" / "python" / "algorithms.md"


def pipeline_function(name):
    """The Python function of the pipeline `name`: `'clouds.rips'` is `phlite.clouds.rips`, and `'complexes'` is
    `phlite.complexes.FilteredComplex.persistence`."""
    if name == "complexes":
        return importlib.import_module("phlite.complexes").FilteredComplex.persistence
    area, function = name.split(".")
    return getattr(importlib.import_module(f"phlite.{area}"), function)


def test_namespace_package():
    """`phlite` is a namespace package: no `__init__.py` from any package, one portion per installed package."""
    assert getattr(phlite, "__file__", None) is None
    for name in ["clouds", "complexes", "images", "networks"]:
        package = importlib.import_module(f"phlite.{name}")
        assert package.__doc__ and package.__file__.endswith(f"phlite/{name}/__init__.py")
    assert phlite.core.__version__ == importlib.metadata.version("phlite-core")
    for name in ["phlite-complexes", "phlite-clouds", "phlite-networks", "phlite-images"]:
        assert importlib.metadata.version(name) == phlite.core.__version__


def test_registry_lists_the_pipelines():
    registry = defaults()
    assert {"complexes", "clouds.rips", "networks.grpph"} <= set(registry)
    for name, presets in registry.items():
        assert set(presets) == {"serial", "parallel"}
        assert presets["serial"] == Algorithm.serial().resolve(name)
        assert presets["parallel"] == Algorithm.parallel().resolve(name)
        # Resolved presets set every option
        for algorithm in presets.values():
            assert algorithm.columns is not None and algorithm.cache is not None and algorithm.store_r is not None
            assert "<pipeline default>" not in repr(algorithm)
            assert algorithm.resolve(name) == algorithm


@pytest.mark.parametrize("name", list(defaults()))
def test_docstrings_state_the_defaults(name):
    """Each pipeline's docstring says `Pipeline defaults (`'name'`): columns='...', cache=..., store_r=...`, as its
    serial preset resolves."""
    serial = defaults()[name]["serial"]
    doc = " ".join(pipeline_function(name).__doc__.split())
    union_find = "" if serial.union_find is None else f", union_find={serial.union_find!r}"
    expected = (
        f"Pipeline defaults (`'{name}'`): columns={serial.columns!r}, cache={serial.cache!r}, "
        f"store_r={serial.store_r!r}{union_find}."
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
    registry = defaults()
    assert set(table) == set(registry), "add a row to the table in docs/book/src/python/algorithms.md"
    for name, (function, columns, cache, store_r, parallel_columns, parallel_cache, union_find) in table.items():
        serial, parallel = registry[name]["serial"], registry[name]["parallel"]
        expected_function = (
            "phlite.complexes.FilteredComplex.persistence" if name == "complexes" else f"phlite.{name}"
        )
        assert function == expected_function
        assert [columns, cache, store_r] == [repr(serial.columns), repr(serial.cache), repr(serial.store_r)]
        assert [parallel_columns, parallel_cache] == [repr(parallel.columns), repr(parallel.cache)]
        assert union_find == ("—" if serial.union_find is None else repr(serial.union_find))
        assert parallel.union_find == serial.union_find


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


def test_parallel_setup():
    """Unset, the setup follows the algorithm; set, either way round, and threads size a serial algorithm's parallel
    setup."""
    assert Algorithm.serial().parallel_setup is None
    assert "parallel_setup" not in repr(Algorithm.serial().resolve("clouds.rips"))
    mixed = Algorithm.serial().with_parallel_setup().with_threads(2)
    assert mixed.parallel_setup is True and mixed.threads == 2
    assert repr(mixed).endswith("threads=2, parallel_setup=True)")
    assert repr(Algorithm.parallel().with_parallel_setup(False)).endswith("parallel_setup=False)")
    with pytest.raises(ValueError, match="threads"):
        Algorithm.serial().with_threads(2)
    assert mixed.resolve("clouds.rips").parallel_setup is True
    from phlite.clouds import rips

    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    expected = rips(points=points)
    for algorithm in [mixed, Algorithm.parallel().with_parallel_setup(False)]:
        barcode = rips(points=points, algorithm=algorithm)
        assert (barcode.pairings, barcode.essential) == (expected.pairings, expected.essential)
        assert barcode.algorithm.parallel_setup == algorithm.parallel_setup


def test_union_find_only_where_supported():
    assert Algorithm.serial().resolve("clouds.rips").union_find is True
    assert Algorithm.serial().with_union_find(False).resolve("clouds.rips").union_find is False
    assert Algorithm.serial().resolve("networks.grpph").union_find is None
    assert "union_find" not in repr(Algorithm.serial().resolve("networks.grpph"))
    assert repr(Algorithm.serial().with_union_find()).endswith("union_find=True)")
    with pytest.raises(ValueError, match="union-find"):
        Algorithm.serial().with_union_find().resolve("networks.grpph")
    # Survives the trip between packages
    barcode = __import__("phlite.clouds", fromlist=["rips"]).rips(
        [[0.0, 1.0], [1.0, 0.0]], algorithm=Algorithm.serial().with_union_find(False)
    )
    assert barcode.algorithm.union_find is False


def test_resolve():
    assert Algorithm.serial().resolve("clouds.rips").columns == "heap"
    assert Algorithm.serial().resolve("complexes").columns == "pivot"
    # Only serial clearing uses pivot columns
    assert Algorithm.parallel().resolve("complexes").columns == "merged"
    # Options set win
    assert Algorithm.serial().with_columns("sorted").resolve("complexes").columns == "sorted"
    with pytest.raises(ValueError, match="Unknown pipeline 'clouds.nope'.*'clouds.rips'"):
        Algorithm.serial().resolve("clouds.nope")
    with pytest.raises(ValueError, match="phlite.nope is not installed"):
        Algorithm.serial().resolve("nope.thing")
    with pytest.raises(ValueError, match="row"):
        Algorithm.serial().with_columns("pivot").resolve("clouds.rips")


def test_spec_round_trip():
    """The private plain-data form every package reads an `Algorithm` through."""
    for algorithm in [
        Algorithm.serial(),
        Algorithm.parallel(3).with_storage("lock-free").with_cache(True),
        Algorithm.standard().with_columns("sorted"),
        Algorithm.clearing().with_dimension_order([2, 1, 0]).with_store_r(False),
    ]:
        spec = algorithm._spec()
        assert spec["protocol"] == 1
        assert Algorithm._from_spec(spec) == algorithm


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
    from phlite.clouds import rips
    from phlite.complexes import ChainComplex
    from phlite.networks import grpph

    rng = np.random.default_rng(0)
    points = rng.random((20, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    barcode = rips(distances)
    assert isinstance(barcode, Barcode)
    assert barcode.algorithm == Algorithm.serial().resolve("clouds.rips")
    assert repr(barcode.algorithm) in repr(barcode)
    parallel = Algorithm.parallel(2).with_cache(True)
    assert rips(distances, algorithm=parallel).algorithm == parallel.resolve("clouds.rips")

    assert grpph(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).algorithm == Algorithm.serial().resolve("networks.grpph")

    chain = ChainComplex.from_pairs([(0, []), (0, []), (1, [0, 1])])
    barcode = chain.filtration([0.0, 1.0, 2.0]).persistence()
    assert barcode.algorithm == Algorithm.serial().resolve("complexes")
    assert repr(barcode.algorithm) in repr(barcode)


def test_pipelines_refuse_what_they_cannot_run():
    from phlite.clouds import rips

    with pytest.raises(ValueError, match="row"):
        rips([[0.0, 1.0], [1.0, 0.0]], algorithm=Algorithm.serial().with_columns("pivot"))
    for not_an_algorithm in ["clearing", 3, object()]:
        with pytest.raises(TypeError, match="phlite.core.Algorithm"):
            rips([[0.0, 1.0], [1.0, 0.0]], algorithm=not_an_algorithm)


def test_modules_and_names():
    from phlite.clouds import rips
    from phlite.networks import grpph

    assert rips.__module__ == "phlite.clouds" and grpph.__module__ == "phlite.networks"
    assert Algorithm.__module__ == "phlite.core" and Barcode.__module__ == "phlite.core"
    assert {"Algorithm", "Barcode", "defaults"} == set(phlite.core.__all__)
    assert isinstance(phlite.core.__version__, str)
