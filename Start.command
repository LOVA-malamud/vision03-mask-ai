#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Create .venv and install dependencies using README.md first.'
  exit 1
fi
if [ ! -f frontend/dist/index.html ]; then
  (cd frontend && npm ci && npm run build)
fi
echo 'Open http://127.0.0.1:8001 — stop with Ctrl+C'
exec .venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
