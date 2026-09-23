#!/bin/sh
# Build the three runtime images. Two of them build FROM ANOTHER CHECKOUT, because the runtimes
# live in their own repos — that is the point of the exercise, not an accident of layout.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES_REPO="${HERMES_REPO:-$HOME/projects/hermes-agent}"
DSH_REPO="${DSH_REPO:-$HOME/projects/deepseek-harness}"

# mcp-tools first: the hermes image copies the bubble-watch venv out of it.
echo "==> bubble-watch/mcp-tools   (context: $ROOT)"
docker build -f "$ROOT/docker/Dockerfile.mcp-tools" -t bubble-watch/mcp-tools "$ROOT"

echo "==> bubble-watch/hermes      (context: $HERMES_REPO)"
docker build -f "$ROOT/docker/Dockerfile.hermes" -t bubble-watch/hermes "$HERMES_REPO"

echo "==> bubble-watch/dsh-runner  (context: $DSH_REPO)"
docker build -f "$ROOT/docker/Dockerfile.dsh-runner" -t bubble-watch/dsh-runner "$DSH_REPO"

echo "done. Run with:"
echo "  BUBBLE_WATCH_DEPLOY=containers BUBBLE_WATCH_HOST_ROOT=$ROOT DSH_BIN=$ROOT/bin/dsh-docker \\"
echo "    uv run bubble-watch run --date <YYYY-MM-DD>"
