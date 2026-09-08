#!/usr/bin/env bash
# 启动演示系统（Linux/macOS / Docker 内）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python"; fi
PORT="${MULTIMODE_PORT:-8501}"

echo "启动演示系统 -> http://localhost:${PORT}"
echo "  数据源: ${MULTIMODE_SOURCE:-synthetic}"
exec "$PY" -m streamlit run app.py --server.port "${PORT}" --server.address 0.0.0.0
