#!/usr/bin/env bash
set -euo pipefail

TRAIN="${1:-}"

if [[ ! -d ".venv" ]]; then
  ./scripts/setup.sh
fi

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
fi

. .venv/bin/activate

if [[ "${TRAIN}" == "--train" ]]; then
  python main.py
fi

python app.py
