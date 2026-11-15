"""Search for networks with bars in dimension >= 2 (using phlite_networks::pph), so that MATLAB comparisons exercise higher dimensions."""
import os, random, subprocess, sys
CLI = os.environ.get("PHLITE_PPH_CLI", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../target/release/phlite_networks_pph"))
rng = random.Random(int(sys.argv[1])); count = int(sys.argv[2]); out = sys.argv[3]
nets = []
for _ in range(count):
    n = rng.randint(5, 7); dim = 2 if n > 5 else 3
    p = rng.uniform(0.2, 0.6); levels = rng.choice([1, 2, 3])
    A = [[0 if i == j else (rng.randint(1, levels) if rng.random() < p else levels + 1) for j in range(n)] for i in range(n)]
    nets.append((A, dim, levels + 1))
text = "".join(f"{len(A)} {d} {t}\n" + "".join(" ".join(map(str, r)) + "\n" for r in A) for A, d, t in nets)
res = subprocess.run([CLI, "--field", "0"], input=text, capture_output=True, text=True, check=True).stdout
blocks = res.split("network ")[1:]
hits = [i for i, b in enumerate(blocks) if any(l.split()[0] in ("2", "3") for l in b.splitlines()[1:])]
print(f"{len(hits)} of {count} have bars in dim >= 2")
with open(out, "w") as f:
    for i in hits:
        A, d, t = nets[i]
        f.write(f"{len(A)} {d} {t}\n" + "".join(" ".join(map(str, r)) + "\n" for r in A))
