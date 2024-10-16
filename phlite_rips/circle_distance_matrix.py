import sys


def circle_matrix(n):
    def dist(i, j):
        return min((j - i) % n, (i - j) % n)

    return dist


N = int(sys.argv[1])
circle_N = circle_matrix(N)
for i in range(N):
    print(*[circle_N(i, j) for j in range(N)], sep=",")
