"""Brute-force persistent path homology for tiny networks: Betti numbers of Ω_p(G_t) at every pair of thresholds,
by linear algebra over GF(q), and the single-reduction algorithm in pure Python. Run directly for a self-test."""
import itertools, random, sys
INF = float('inf')

def rank(rows, q):
    rows = [list(r) for r in rows]; r = 0
    if not rows: return 0
    ncol = len(rows[0])
    for c in range(ncol):
        piv = next((i for i in range(r, len(rows)) if rows[i][c] % q), None)
        if piv is None: continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = pow(rows[r][c], q-2, q)
        rows[r] = [x*inv % q for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] % q:
                f = rows[i][c]; rows[i] = [(a-f*b) % q for a, b in zip(rows[i], rows[r])]
        r += 1
    return r

def nullspace(M, ncols, q):
    # M: list of rows (length ncols); return basis of {x: Mx=0}
    rows = [list(r) for r in M]; pivcols = []; r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(rows)) if rows[i][c] % q), None)
        if piv is None: continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = pow(rows[r][c], q-2, q); rows[r] = [x*inv % q for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] % q:
                f = rows[i][c]; rows[i] = [(a-f*b) % q for a, b in zip(rows[i], rows[r])]
        pivcols.append(c); r += 1
    free = [c for c in range(ncols) if c not in pivcols]
    basis = []
    for fc in free:
        v = [0]*ncols; v[fc] = 1
        for i, pc in enumerate(pivcols): v[pc] = (-rows[i][fc]) % q
        basis.append(v)
    return basis

def regular_paths(n, p):
    return [s for s in itertools.product(range(n), repeat=p+1) if all(s[i] != s[i+1] for i in range(p))]

def bd(s, q):
    out = {}
    for i in range(len(s)):
        f = s[:i] + s[i+1:]
        if any(f[k] == f[k+1] for k in range(len(f)-1)): continue
        out[f] = (out.get(f, 0) + (-1)**i) % q
    return {k: v for k, v in out.items() if v}

def ptime(s, w):
    if len(s) == 1: return 0
    return max(w.get((s[i], s[i+1]), INF) for i in range(len(s)-1))

def brute(n, w, p, q, T):
    Pp, Pp1, Pm = regular_paths(n, p), regular_paths(n, p+1), regular_paths(n, p-1) if p >= 1 else []
    idx = {s: i for i, s in enumerate(Pp)}
    res = {}
    def cycles(s):
        A = [x for x in Pp if ptime(x, w) <= s]
        if p == 0: M = []
        else:
            rows = Pm; M = [[bd(x, q).get(r, 0) for x in A] for r in rows]
        Z = nullspace(M, len(A), q) if M else [[1 if i == j else 0 for i in range(len(A))] for j in range(len(A))]
        return [[0]*len(Pp) if False else embed(z, A) for z in Z]
    def embed(z, A):
        v = [0]*len(Pp)
        for c, x in zip(z, A): v[idx[x]] = c
        return v
    def bounds(t):
        A = [x for x in Pp1 if ptime(x, w) <= t]
        bad = [r for r in Pp if ptime(r, w) > t]
        M = [[bd(x, q).get(r, 0) for x in A] for r in bad]
        Om = nullspace(M, len(A), q) if M else [[1 if i == j else 0 for i in range(len(A))] for j in range(len(A))]
        out = []
        for c in Om:
            v = [0]*len(Pp)
            for coef, x in zip(c, A):
                for f, e in bd(x, q).items(): v[idx[f]] = (v[idx[f]] + coef*e) % q
            out.append(v)
        return out
    for s in T:
        Z = cycles(s); dz = rank(Z, q)
        for t in T:
            if t < s: continue
            B = bounds(t)
            res[(s, t)] = rank(Z + B, q) - rank(B, q) if B else dz
    return res

def reduce(cols, q):
    # cols: list of dict row->coef, rows are sortable keys; returns list of low or None
    lows = {}; out = []
    for c in cols:
        c = dict(c)
        while c:
            l = max(c)
            if l not in lows: break
            o = lows[l]; f = c[l] * pow(o[l], q-2, q) % q
            for k, v in o.items():
                c[k] = (c.get(k, 0) - f*v) % q
                if c[k] == 0: del c[k]
        if c: lows[max(c)] = c; out.append(max(c))
        else: out.append(None)
    return out

def fast(n, w, p, q, T):
    def key(s): return (ptime(s, w), s)
    def mat(d):
        C = sorted([x for x in regular_paths(n, d) if ptime(x, w) < INF], key=key)
        cols = [{key(f): e for f, e in bd(x, q).items()} for x in C]
        return C, reduce(cols, q)
    Cp, lp = mat(p) if p >= 1 else (sorted([(i,) for i in range(n)], key=key), [None]*n)
    Cq, lq = mat(p+1)
    positive = [x for x, l in zip(Cp, lp) if l is None]
    bars = []; killed = set()
    for x, l in zip(Cq, lq):
        if l is None or l[0] == INF: continue
        killed.add(l[1]); bars.append((l[0], max(ptime(x, w), l[0])))
    bars += [(ptime(x, w), INF) for x in positive if x not in killed]
    return {(s, t): sum(1 for b, d in bars if b <= s and d > t) for s in T for t in T if t >= s}

if __name__ == "__main__":
    random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    bad = 0
    for trial in range(60):
        n = random.choice([4, 5]); dens = random.choice([0.4, 0.6, 0.9])
        w = {(a, b): random.randint(1, 4) for a in range(n) for b in range(n) if a != b and random.random() < dens}
        T = sorted(set(w.values()) | {0})
        for p in [0, 1, 2]:
            if p == 2 and n == 5: continue
            for q in [2, 3]:
                a, b = brute(n, w, p, q, T), fast(n, w, p, q, T)
                if a != b: bad += 1; print("MISMATCH", n, w, p, q, a, b); break
    print("mismatches:", bad)
