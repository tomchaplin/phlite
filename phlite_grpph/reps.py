from phlite_grpph import grpph

essential, pairings = grpph(
    6,
    [
        (0, 1, 10.0),
        (0, 2, 10.0),
        (1, 3, 10.0),
        (2, 3, 10.0),
        (0, 3, 20.0),
        (0, 4,  5.0),
        (4, 5, 10.0),
        (5, 3,  5.0)
    ]
)

print(essential)
print(pairings)
