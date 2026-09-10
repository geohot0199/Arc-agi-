#!/usr/bin/env bash
# One-time setup for the LOCAL verification harness (not needed on Kaggle).
#
# The Kaggle submission only needs the competition's own wheels.  This script
# exists so that `harness/run_local.py` can drive the real arc_agi engine and
# the real ARC-AGI-3-Agents Agent base class against the benchmark fixtures.
#
#   bash harness/setup.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ATLAS_VENV:-$ROOT/.venv}"
AGENTS_DIR="${ATLAS_AGENTS_DIR:-$ROOT/third_party/ARC-AGI-3-Agents}"
AGENTS_REF="${ATLAS_AGENTS_REF:-https://github.com/arcprize/ARC-AGI-3-Agents.git}"

echo "== python venv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip

echo "== arc-agi toolkit + engine (arc-agi>=0.9.1 pulls arcengine)"
# The published wheels declare Requires-Python>=3.12 but are pure python; on an
# older interpreter the flag below is what makes them installable.
# numpy is capped so pip never falls back to a source build (2.3+ has no
# cp311 wheel, and a source build needs python3-dev + Cython).
"$VENV/bin/pip" install --quiet --ignore-requires-python \
  "arc-agi>=0.9.1" "numpy<2.3"

echo "== ARC-AGI-3-Agents framework at $AGENTS_DIR"
if [ ! -d "$AGENTS_DIR/agents" ]; then
  mkdir -p "$(dirname "$AGENTS_DIR")"
  git clone --depth 1 "$AGENTS_REF" "$AGENTS_DIR"
fi

echo
echo "done.  Run the benchmark with:"
echo "  $VENV/bin/python harness/run_local.py"
