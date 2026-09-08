#!/usr/bin/env bash
# 多模态高光谱植物表型系统 - 一键安装引导 (Linux/macOS)
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[ERROR] Python not found. Please install Python 3.9+." >&2
  exit 1
fi

echo "Using: $PY"
export PYTHONIOENCODING=utf-8
"$PY" bootstrap.py "$@"
