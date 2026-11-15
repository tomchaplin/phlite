"""Networks with missing edges (never-allowed paths, infinite allow times), which pph-matlab can't express:
compare phlite_networks::pph against a brute-force computation of persistent Betti numbers from Ω_p(G_t) (pphtest/check.py)."""
import os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brute as check
CLI = os.environ.get("PHLITE_PPH_CLI", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../target/release/phlite_networks_pph"))
INF = float("inf")
rng = random.Random(int(sys.argv[1])); count = int(sys.argv[2])
bad = total = 0
for trial in range(count):
    n = rng.choice([4, 5]); dens = rng.choice([0.3, 0.5, 0.7])
    w = {(a, b): rng.randint(1, 4) for a in range(n) for b in range(n) if a != b and rng.random() < dens}
    T = sorted(set(w.values()) | {0})
    maxdim = 2 if n == 4 else 1
    text = f"{n} {maxdim} 100\n" + "".join(" ".join("0" if a == b else str(w.get((a, b), "inf")) for b in range(n)) + "\n" for a in range(n))
    for q in (2, 3):
        for method in ("homology", "cohomology"):
            out = subprocess.run([CLI, "--no-cap", "--field", str(q), "--method", method], input=text,
                                 capture_output=True, text=True, check=True).stdout.splitlines()[1:]
            bars = [(int(d), float(b), float(e)) for d, b, e in (l.split() for l in out)]
            for p in range(maxdim + 1):
                ref = check.brute(n, w, p, q, T)
                got = {(s, t): sum(1 for d, b, e in bars if d == p and b <= s and e > t) for s in T for t in T if t >= s}
                total += 1
                if got != ref:
                    bad += 1
                    print("MISMATCH", q, method, p, w, ref, got)
print(f"sparse networks: {bad} mismatches of {total} (network, field, method, dim) checks")
