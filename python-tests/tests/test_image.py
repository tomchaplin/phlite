"""`phlite.image`: nothing to test yet beyond that it imports. Add each pipeline's fuzz tests (against gudhi and
cripser) as `tests/test_image_<pipeline>.py`, e.g. `test_image_cubical.py`; the justfile picks them up."""

import phlite.image


def test_imports():
    assert phlite.image.__doc__
