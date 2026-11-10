from phlite_grpph import grpph
from pprint import pprint
import time


def do_complete(N):
    barcode = grpph(N, [(i, j, 1.0) for i in range(N) for j in range(N) if i != j])
    essential, pairings = barcode.essential, barcode.pairings

    assert len(essential[0]) == 1
    assert len(essential[1]) == 0
    assert len(pairings[0]) == 0
    assert len(pairings[1]) == N * (N - 1) - N + 1

    print("Done complete")


def do_cycle(N, with_reps=False):
    barcode = grpph(
        N, [(i, (i + 1) % N, 1.0) for i in range(N)], representatives=with_reps
    )
    essential, pairings = barcode.essential, barcode.pairings
    if with_reps:
        reps = barcode.pairing_representatives[1]
        assert len(reps) == 1
        assert len(reps[0]) == N

    assert len(essential[0]) == 1
    assert len(essential[1]) == 0
    assert len(pairings[0]) == 0
    assert len(pairings[1]) == 1

    return barcode


N = 200

tic0 = time.time()
do_cycle(N, with_reps=False)
tic1 = time.time()
print(f"Without reps: {tic1 - tic0}")

tic2 = time.time()
do_cycle(N, with_reps=True)
tic3 = time.time()
print(f"With reps   : {tic3 - tic2}")

do_complete(30)
