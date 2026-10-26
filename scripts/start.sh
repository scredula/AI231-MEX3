#!/usr/bin/env bash
# Start the voice agent (voice mode by default; TEXT=1 for text mode).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "Missing .env — run scripts/setup_pi.sh first." >&2
  exit 1
fi

mkdir -p logs
if [ "${TEXT:-0}" = "1" ]; then
  .venv/bin/python -m agent.main 2>&1 | tee -a logs/agent.log
else
  .venv/bin/python -m agent.main --voice 2>&1 | tee -a logs/agent.log
fi
