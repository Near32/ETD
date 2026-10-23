#!/bin/bash
# Every 120s: main RSS/swap, summed env-worker PSS/swap, system memory. Arg: main PID.
M=$1
while kill -0 $M 2>/dev/null; do
  m=$(awk '/^Rss:/{r=$2} /^Swap:/{s=$2} END{printf "main_rss=%dMB main_swap=%dMB", r/1024, s/1024}' /proc/$M/smaps_rollup)
  w=0; ws=0; n=0
  for c in $(for k in $(pgrep -P $M); do pgrep -P $k; done); do
    read a b <<<"$(awk '/^Pss:/{p=$2} /^Swap:/{s=$2} END{print p, s}' /proc/$c/smaps_rollup 2>/dev/null)"
    w=$((w+${a:-0})); ws=$((ws+${b:-0})); n=$((n+1))
  done
  echo "$(date '+%F %T') $m envs=$n env_pss=$((w/1024))MB env_swap=$((ws/1024))MB $(free -m | awk '/^Mem:/{printf "avail=%dMB", $7}')"
  sleep 120
done
