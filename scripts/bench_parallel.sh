#!/usr/bin/env bash
# Benchmark serial vs parallel clearing reduction for phlite_grpph and phlite_rips.
#
# Usage: scripts/bench_parallel.sh [REPEATS] [THREADS...]
#   e.g. scripts/bench_parallel.sh 5 1 2 4 8
# Override problem sizes via env vars:
#   LOCK_FREE=1 also builds and benchmarks the lock-free backend (cargo feature `lock-free`)
#   GRPPH_N (default 100) GRPPH_P (0.1) RIPS_N (160) RIPS_DIM (2) SEED (5)
# Modes: serial; par:N = locking backend on N threads; lf:N = lock-free backend on N threads.
# Output: one line per run, then a per-config summary (min / median of reduce time).
set -euo pipefail
cd "$(dirname "$0")/.."

REPEATS="${1:-5}"; shift || true
THREADS=("$@"); [ ${#THREADS[@]} -eq 0 ] && THREADS=(1 2 4 8)
GRPPH_N="${GRPPH_N:-100}"; GRPPH_P="${GRPPH_P:-0.1}"
RIPS_N="${RIPS_N:-160}"; RIPS_DIM="${RIPS_DIM:-2}"; SEED="${SEED:-5}"

FEATURES=(); [ "${LOCK_FREE:-0}" = 1 ] && FEATURES=(--features lock-free)
cargo build --release --examples -p phlite_grpph -p phlite_rips "${FEATURES[@]}"
echo "# cores: $(nproc)   repeats: $REPEATS   threads: ${THREADS[*]}"

modes=(serial); for t in "${THREADS[@]}"; do modes+=("par:$t"); [ "${LOCK_FREE:-0}" = 1 ] && modes+=("lf:$t"); done
out=$(mktemp); trap 'rm -f "$out"' EXIT

for bench in grpph rips; do
  if [ $bench = grpph ]; then args=("$GRPPH_N" "$GRPPH_P" "$SEED"); else args=("$RIPS_N" "$RIPS_DIM" "$SEED"); fi
  for m in "${modes[@]}"; do
    for ((r=0; r<REPEATS; r++)); do
      line=$(target/release/examples/bench_${bench}_parallel "${args[@]}" "$m")
      echo "$bench ${line}"
      t=$(sed -E 's/.*reduce ([0-9.]+)s.*/\1/' <<<"$line")
      echo "$bench $m $t" >> "$out"
    done
  done
done

echo; echo "# summary (reduce seconds)"
python3 - "$out" <<'PY'
import sys, statistics, collections
d = collections.OrderedDict()
for l in open(sys.argv[1]):
    b, m, t = l.split(); d.setdefault((b, m), []).append(float(t))
print(f"{'bench':6} {'mode':7} {'min':>8} {'median':>8}")
for (b, m), v in d.items():
    print(f"{b:6} {m:7} {min(v):8.3f} {statistics.median(v):8.3f}")
PY
