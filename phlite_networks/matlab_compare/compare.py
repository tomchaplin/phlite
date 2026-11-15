"""Compare phlite_networks::pph against pph-matlab (run under Octave) on random and structured networks.

Usage: PPH_MATLAB_DIR=/path/to/pph-matlab python3 compare.py [seed] [n_random] [extra_networks.txt]
(build the CLI first: cargo build --release -p phlite_networks)
"""
import itertools, math, os, random, subprocess, sys, time
from collections import Counter

CLI = os.environ.get("PHLITE_PPH_CLI", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../target/release/phlite_networks_pph"))
HERE = os.path.dirname(os.path.abspath(__file__))
INF = float("inf")


# ---------- networks: (name, weight matrix, max_dim, t_end) ----------

def cycle(n):
    # pph-matlab's cycle_nets.mat convention: A[i][j] = (j - i) mod n
    return [[(j - i) % n for j in range(n)] for i in range(n)]

def structured():
    nets = []
    for n in range(3, 9):
        nets.append((f"cycle C_{n}", cycle(n), 1, n))
    for n in range(3, 7):
        nets.append((f"cycle C_{n} (dim 2)", cycle(n), 2, n))
    for n in (3, 4, 5):
        nets.append((f"cycle C_{n} (dim 3)", cycle(n), 3, n))
    # Complete digraph, all edges at time 1
    for n in (3, 4, 5, 6):
        nets.append((f"complete K_{n}", [[0 if i == j else 1 for j in range(n)] for i in range(n)], 1, 2))
    for n in (3, 4, 5):
        nets.append((f"complete K_{n} (dim 2)", [[0 if i == j else 1 for j in range(n)] for i in range(n)], 2, 2))
    # Bi-fan a->b, c->b, c->d, a->d (H_1 = 1) and bi-parallel w->x->y, w->z->y (H_1 = 0), other edges at t_end
    def from_edges(n, edges, rest):
        A = [[0 if i == j else rest for j in range(n)] for i in range(n)]
        for a, b, w in edges:
            A[a][b] = w
        return A
    nets.append(("bi-fan", from_edges(4, [(0, 1, 1), (2, 1, 1), (2, 3, 1), (0, 3, 1)], 3), 1, 3))
    nets.append(("bi-parallel", from_edges(4, [(0, 1, 1), (1, 3, 1), (0, 2, 1), (2, 3, 1)], 3), 2, 3))
    nets.append(("bi-parallel, staggered", from_edges(4, [(0, 1, 1), (1, 3, 2), (0, 2, 1), (2, 3, 1)], 4), 2, 4))
    # Directed triangle vs 3-cycle
    nets.append(("directed triangle", from_edges(3, [(0, 1, 1), (1, 2, 1), (0, 2, 2)], 3), 2, 3))
    nets.append(("3-cycle then reverse", from_edges(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1), (1, 0, 2), (2, 1, 2), (0, 2, 2)], 3), 2, 3))
    # Circulant digraphs i -> i+1 at 1, i -> i+2 at 2, ...
    for n in (5, 6, 7):
        A = [[0 if i == j else (j - i) % n for j in range(n)] for i in range(n)]
        A = [[min(x, 3) if x else 0 for x in row] for row in A]
        nets.append((f"circulant Z_{n} capped", A, 2 if n < 7 else 1, 3))
    # Directed grid (a DAG): (r,c) -> (r,c+1), (r+1,c); time = 1, everything else at t_end
    for R, C in ((2, 2), (2, 3), (3, 3)):
        idx = lambda r, c: r * C + c
        edges = [(idx(r, c), idx(r, c + 1), 1) for r in range(R) for c in range(C - 1)]
        edges += [(idx(r, c), idx(r + 1, c), 1) for r in range(R - 1) for c in range(C)]
        nets.append((f"grid DAG {R}x{C}", from_edges(R * C, edges, 3), 2 if R * C <= 6 else 1, 3))
    # Directed octahedron-like 3-sphere candidate: 2-dim "suspension" with sources/sinks
    edges = [(0, a, 1) for a in (1, 2, 3, 4)] + [(a, 5, 1) for a in (1, 2, 3, 4)]
    edges += [(1, 2, 1), (3, 2, 1), (3, 4, 1), (1, 4, 1)]
    nets.append(("suspension of bi-fan", from_edges(6, edges, 2), 2, 2))
    # All 3-node networks with weights in {1, 2, 3}
    pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
    for k, ws in enumerate(itertools.product((1, 2, 3), repeat=6)):
        A = [[0] * 3 for _ in range(3)]
        for (a, b), w in zip(pairs, ws):
            A[a][b] = w
        nets.append((f"3-node #{k}", A, 2, 3))
    return nets

def random_nets(rng, count):
    nets = []
    for k in range(count):
        kind = rng.choice(["int", "real", "symmetric", "metric", "sparse"])
        dim = rng.choice([1, 1, 2, 2, 3])
        n = {1: rng.randint(3, 8), 2: rng.randint(3, 6), 3: rng.randint(3, 5)}[dim]
        t_end = 5
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if kind == "int":
                    A[i][j] = rng.randint(1, 4)
                elif kind == "real":
                    A[i][j] = round(rng.uniform(0.1, 4.9), 6)
                elif kind == "symmetric":
                    A[i][j] = A[j][i] if j < i else rng.randint(1, 4)
                elif kind == "sparse":
                    # missing edges appear at t_end (pph-matlab caps weights at t_end)
                    A[i][j] = rng.randint(1, 3) if rng.random() < 0.4 else t_end
        if kind == "metric":
            # directed (quasi-)metric: shortest paths in a random weighted digraph
            for i in range(n):
                for j in range(n):
                    if i != j:
                        A[i][j] = rng.randint(1, 9) if rng.random() < 0.5 else 100
            for m in range(n):
                for i in range(n):
                    for j in range(n):
                        A[i][j] = min(A[i][j], A[i][m] + A[m][j])
            t_end = max(A[i][j] for i in range(n) for j in range(n)) + 1
        nets.append((f"random {kind} n={n}", A, dim, t_end))
    return nets


# ---------- running both ----------

def write_input(path, nets):
    with open(path, "w") as f:
        for _name, A, dim, t_end in nets:
            f.write(f"{len(A)} {dim} {t_end}\n")
            for row in A:
                f.write(" ".join(repr(float(x)) for x in row) + "\n")

def read_input(path):
    toks = open(path).read().split(); nets = []; i = 0
    while i < len(toks):
        n, dim, t_end = int(toks[i]), int(toks[i + 1]), float(toks[i + 2]); i += 3
        A = [[float(toks[i + r * n + c]) for c in range(n)] for r in range(n)]; i += n * n
        nets.append((f"file network {len(nets)}", A, dim, t_end))
    return nets

def parse_output(text):
    out = []
    for line in text.splitlines():
        if line.startswith("network"):
            out.append([])
        elif line.strip():
            d, b, e = line.split()
            out[-1].append((int(d), float(b), float(e)))
    return out

def run_matlab(infile):
    outfile = infile + ".matlab"
    subprocess.run(["octave", "--no-gui", "--quiet", "--eval", f"run_matlab('{infile}', '{outfile}')"],
                   cwd=HERE, check=True, stderr=subprocess.DEVNULL)
    return parse_output(open(outfile).read())

def run_rust(infile, *args):
    res = subprocess.run([CLI, *args], stdin=open(infile), capture_output=True, text=True, check=True)
    return parse_output(res.stdout)

def canon(bars):
    return Counter((d, round(b, 9), round(e, 9) if e != INF else INF) for d, b, e in bars)

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    rng = random.Random(seed)
    groups = {"structured": structured(), "random": random_nets(rng, n_random)}
    if len(sys.argv) > 3:
        # extra networks from a file in the input format (e.g. found by search_h2.py), limited to the first 80
        groups["higher-dim hits"] = read_input(sys.argv[3])[:80]
    configs = [("--method", m, "--algorithm", a, "--field", f)
               for m in ("homology", "cohomology") for a in ("standard", "clearing", "parallel") for f in ("0",)]
    configs += [("--method", "cohomology", "--algorithm", "clearing", "--field", f) for f in ("2", "3")]
    configs += [("--method", m, "--algorithm", "clearing", "--field", "0", "--representatives") for m in ("homology", "cohomology")]
    total_fail = 0
    for group, nets in groups.items():
        infile = f"/tmp/pph_{group}_{seed}.txt"
        write_input(infile, nets)
        t0 = time.time(); ref = run_matlab(infile); t_matlab = time.time() - t0
        assert len(ref) == len(nets)
        dims = Counter(d for _, _, d, _ in nets)
        per_dim = Counter(d for bars in ref for d in {d for d, _, _ in bars})
        n_bars = Counter(d for bars in ref for d, _, _ in bars)
        print(f"{group}: {len(nets)} networks (max dim {dict(sorted(dims.items()))}), octave {t_matlab:.1f}s")
        print(f"  MATLAB bars by dim: {dict(sorted(n_bars.items()))}; networks with bars by dim: {dict(sorted(per_dim.items()))}")
        for cfg in configs:
            t0 = time.time(); got = run_rust(infile, *cfg); t_rust = time.time() - t0
            fails = [i for i in range(len(nets)) if canon(got[i]) != canon(ref[i])]
            total_fail += len(fails)
            label = " ".join(cfg[1::2]) if "--representatives" not in cfg else " ".join(cfg[1:-1:2]) + " +reps"
            print(f"  {label:32s} mismatches {len(fails):3d}/{len(nets)}  ({t_rust:.2f}s)")
            for i in fails[:3]:
                print("    ", nets[i][0], nets[i][1], "\n      matlab", sorted(ref[i]), "\n      phlite", sorted(got[i]))
    print("TOTAL MISMATCHES", total_fail)

if __name__ == "__main__":
    main()
