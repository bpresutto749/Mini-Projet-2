#!/usr/bin/env bash
CONFIGS=${CONFIGS:-"lighthouse:team4 abyss:team4 abyss:team3 cornfield_crossing:team3"}
JOBS=${JOBS:-6}
mkdir -p logs
for c in $CONFIGS; do
  for seed in $(seq 0 9); do
    for algo in "rlpd" "rlpd --offline-ratio 0" "dagger"; do
      echo "--track ${c%:*} --expert ${c#*:} --algo $algo --seed $seed"
    done
  done
done | xargs -P "$JOBS" -I{} sh -c "uv run src/train.py {} > \"logs/\$(echo '{}' | tr -d ' -').log\" 2>&1"
uv run src/analysis.py --last 2
