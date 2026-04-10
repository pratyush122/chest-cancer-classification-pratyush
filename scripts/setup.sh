#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3}"

echo "Using Python interpreter: ${PYTHON_BIN}"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

"${PYTHON_BIN}" -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r requirements-pipeline.txt

echo ""
echo "Setup completed."
echo "Activate environment: source .venv/bin/activate"
echo "Run web app: python app.py"
echo "Run training pipeline: python main.py"
