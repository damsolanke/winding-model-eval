#!/bin/bash
# Run evaluation jobs one after another (they share the GPU).
# Usage: scripts/run_queue.sh QUEUE_FILE [PID_TO_WAIT_FOR]
# Each queue line: CONFIG [extra windeval.evaluate args...]
# Env: PYTHON (default: python on PATH), WINDEVAL_DATA (default ~/kaggle-data/vesuvius); logs in $WINDEVAL_DATA/logs.
QUEUE="$1"; WAIT_PID="$2"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
DATA="${WINDEVAL_DATA:-$HOME/kaggle-data/vesuvius}"
LOGS="$DATA/logs"; mkdir -p "$LOGS"
cd "$HERE" || exit 1
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
fi
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  tag=$(echo "$line" | tr ' /' '__' | sed 's/configs_//; s/\.json//; s/--//g')
  echo "$(date '+%F %T') START $line" >> "$LOGS/queue.log"
  "$PY" -u -m windeval.evaluate $line > "$LOGS/eval_${tag}.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') END rc=$rc $line" >> "$LOGS/queue.log"
done < "$QUEUE"
echo "$(date '+%F %T') QUEUE DONE $QUEUE" >> "$LOGS/queue.log"
