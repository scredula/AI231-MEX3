#!/usr/bin/env bash
# Stop the voice agent (matches processes running agent.main).
set -euo pipefail
pkill -f "agent.main" 2>/dev/null && echo "stopped" || echo "no running agent found"
