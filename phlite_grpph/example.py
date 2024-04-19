from phlite_grpph import grpph
from pprint import pprint
import time

def do_complete(N):
    essential, pairings = grpph(
        N,
        [
            (i, j, 1.0)
            for i in range(N)
            for j in range(N)
            if i != j
        ] 
    )

    assert len(essential[0]) == 1
    assert len(essential[1]) == 0
    assert len(pairings[0]) == 0
    assert len(pairings[1]) == N * (N-1) - N + 1

    print("Done complete")

def do_cycle(N):
    essential, pairings = grpph(
        N,
        [
            (i, (i+1)%N, 1.0)
            for i in range(N)
        ] 
    )

    assert len(essential[0]) == 1
    assert len(essential[1]) == 0
    assert len(pairings[0]) == 0
    assert len(pairings[1]) == 1
    print("Done cycle")

#do_complete(100)
N=100
do_cycle(100)
involution_size = N * (N-1) - N + 1
o2_size = N * (N-1) * (N-1) # Eventually every 2paths is allowed
assert N == o2_size/involution_size
print(involution_size)

# benchmarks = []
# N_range = [10,15, 20,25, 30,40, 50, 100,120, 140, 150,180, 200, 220, 250, 280, 300, 350, 400, 500]
# 
# for N in N_range:
#     tic = time.time()
#     do_cycle(N)
#     toc = time.time()
#     benchmarks.append(toc - tic)
# 
# for N,t in zip(N_range, benchmarks):
#     print(f'{N},{t:.3f}')
