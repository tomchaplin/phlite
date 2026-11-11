"""Persistent homology with your own cell objects: a filtered hollow tetrahedron."""

import dataclasses
import itertools

import phlite


@dataclasses.dataclass(frozen=True)
class Simplex:
    vertices: tuple
    # Not part of equality, so faces can be built without knowing when they enter
    entrance_time: float = dataclasses.field(default=None, compare=False)

    @property
    def dimension(self):
        return len(self.vertices) - 1

    def boundary(self):
        vs = self.vertices
        return (
            [Simplex(vs[:i] + vs[i + 1 :]) for i in range(len(vs))]
            if len(vs) > 1
            else []
        )


cells = [
    Simplex(vs, float(len(vs) - 1 + (sum(vs) % 3) / 3))
    for k in range(1, 4)
    for vs in itertools.combinations(range(4), k)
]
barcode = phlite.CellComplex(cells).persistence(representatives=True)
print(barcode)
for bar in barcode.bars:
    print(f"H_{bar.dimension} [{bar.birth_value:.2f}, {bar.death_value:.2f}):")
    print(
        "    born at",
        bar.birth.vertices,
        "killed by",
        bar.death.vertices if bar.death else None,
    )
    print("    cycle:", [(cell.vertices, coeff) for cell, coeff in bar.representative])

# Plain tuples work too, if you say how to get dimensions and boundaries.
# Over Z/3, with an oriented boundary given as a mapping from faces to coefficients
simplices = [vs for k in range(1, 4) for vs in itertools.combinations(range(4), k)]
complex = phlite.CellComplex(
    simplices,
    dimension=lambda s: len(s) - 1,
    boundary=lambda s: (
        {s[:i] + s[i + 1 :]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {}
    ),
    modulus=3,
)
print(complex, complex.persistence(representatives=True).bar((1, 2, 3)))
