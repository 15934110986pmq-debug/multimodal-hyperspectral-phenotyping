# 启动演示系统（Windows）
# 可用环境变量：MULTIMODE_SOURCE / MULTIMODE_DATA_ROOT / MULTIMODE_MODEL / MULTIMODE_TRAITS / MULTIMODE_PORT
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = if (Test-Path "$Root\.venv\Scripts\python.exe") { "$Root\.venv\Scripts\python.exe" } else { "python" }
$Port = if ($env:MULTIMODE_PORT) { $env:MULTIMODE_PORT } else { 8501 }

Write-Host "启动演示系统 ->  http://localhost:$Port"
Write-Host "  数据源: $($env:MULTIMODE_SOURCE)"
& $Python -m streamlit run app.py --server.port $Port --server.address 0.0.0.0
