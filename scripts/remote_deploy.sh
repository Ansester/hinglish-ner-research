#!/usr/bin/env bash
# Explicit code/data sync only; does not install packages or start training.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 user@host [remote/path]" >&2
  exit 2
fi
REMOTE=$1
REMOTE_PATH=${2:-work/hinglish-ner-research}
if [[ ! "$REMOTE" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$ || ! "$REMOTE_PATH" =~ ^[A-Za-z0-9_./-]+$ || "$REMOTE_PATH" == -* ]]; then
  echo "Use a simple user@host and a path containing letters, digits, _, -, / or ." >&2
  exit 2
fi
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"
ssh "$REMOTE" "mkdir -p -- '$REMOTE_PATH'"
rsync -av --protect-args --exclude=.git --exclude=.venv --exclude=outputs --exclude=__pycache__ \
  "$LOCAL_PATH/" "$REMOTE:$REMOTE_PATH/"
echo "Synced. Run training explicitly on the remote host after checking its environment."
