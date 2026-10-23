#!/bin/bash
# Log system memory and training-process RSS once a minute while the run is alive.
PAT="ppo_etd/train.py --exp_name $1"
sleep 30
while pgrep -f "$PAT" >/dev/null; do
  read -r used avail <<<"$(free -m | awk '/^Mem:/{print $3, $7}')"
  swap=$(free -m | awk '/^Swap:/{print $3}')
  rss=$(ps -eo rss,comm | awk '$2=="pt_main_thread"||$2=="python"{s+=$1} END{print int(s/1024)}')
  echo "$(date '+%F %T') mem_used=${used}MB avail=${avail}MB swap_used=${swap}MB train_rss_sum=${rss}MB"
  sleep 60
done
echo "$(date '+%F %T') training gone"
