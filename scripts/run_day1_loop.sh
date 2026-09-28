#!/usr/bin/env bash
#
# Keep the day-1 corpus growing in the background, all day.
#
# Safe to re-run at any time: the loop works out for itself which seeds still
# need training (including any seed an earlier interrupt left half-done), and
# `run_corpus` skips a model whose spec digest already matches. Nothing is ever
# repeated, so a crash or a killed terminal costs you nothing but the model that
# was in flight.
#
# Start it (launchd owns the job, so it survives this terminal closing):
#   launchctl submit -l com.cviaf.day1loop \
#     -o "$PWD/logs/day1_loop.log" -e "$PWD/logs/day1_loop.err" \
#     -- /bin/bash "$PWD/scripts/run_day1_loop.sh"
#
# Watch it:
#   tail -f logs/day1_loop.log
#   cat runs/day1/loop_status.json      # heartbeat: cycles, models, last update
#
# Stop it:
#   launchctl remove com.cviaf.day1loop
#
# Why not `nohup ... &`: measured on this machine, a nohup'd job was reaped within
# seconds of the launching shell going away -- the log stops mid-cycle with no
# traceback, because the whole process group dies. macOS ships no `setsid`. If you
# are keeping a real terminal open for the duration, nohup is fine:
#   nohup ./scripts/run_day1_loop.sh > logs/day1_loop.log 2>&1 &
#
# Override the wall-clock limit (default 8 h) or the per-cycle pause:
#   LOOP_HOURS=12 LOOP_INTERVAL=10 launchctl submit -l com.cviaf.day1loop \
#     -o "$PWD/logs/day1_loop.log" -- /bin/bash "$PWD/scripts/run_day1_loop.sh"
#
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

# -u is not optional: an unflushed stdout buffer is exactly why a killed
# background run leaves an empty log and you cannot tell whether it did anything.
exec .venv/bin/python -u -m cviaf.lab loop \
  --plan configs/corpus_day1.json \
  --seeds-per-cycle "${LOOP_SEEDS_PER_CYCLE:-4}" \
  --interval-minutes "${LOOP_INTERVAL:-20}" \
  --hours "${LOOP_HOURS:-8}"
