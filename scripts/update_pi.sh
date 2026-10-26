#!/usr/bin/env bash
# Pull approved updates from GitHub without touching local config or task data.
# Local files preserved: .env, workspace/, data/tasks.json (tracked paths are not
# overwritten because .env/workspace/data are gitignored on both ends).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Stopping agent"
pkill -f "agent.main" 2>/dev/null || true

echo "==> Pulling updates"
git pull --ff-only origin main

echo "==> Refreshing dependencies"
. .venv/bin/activate 2>/dev/null || { echo "venv missing — run scripts/setup_pi.sh"; exit 1; }
pip install -q -r requirements.txt

echo "==> Restarting agent"
nohup scripts/start.sh > logs/agent.log 2>&1 &
echo "restarted (log: logs/agent.log)"
