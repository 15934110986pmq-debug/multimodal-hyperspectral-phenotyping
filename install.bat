@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Rediscover Python (system python first, then py launcher).
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)

if not defined PY (
  echo [ERROR] Python not found. Please install Python 3.9+ from https://www.python.org/
  echo         and make sure "Add Python to PATH" is checked during install.
  pause
  exit /b 1
)

echo Using: %PY%
set PYTHONIOENCODING=utf-8
%PY% bootstrap.py %*

echo.
pause
