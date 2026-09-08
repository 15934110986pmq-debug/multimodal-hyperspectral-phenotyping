# 本机后台部署：把演示系统作为常驻服务运行（隐藏窗口，写日志）。
# 用法：powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
# 环境变量：MULTIMODE_SOURCE/MULTIMODE_DATA_ROOT/MULTIMODE_MODEL/MULTIMODE_TRAITS/MULTIMODE_PORT/MULTIMODE_HOST
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$StateDir = Join-Path $Root "output"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$StateFile = Join-Path $StateDir "deploy_state.json"
if (Test-Path $StateFile) {
    $old = Get-Content $StateFile -Raw | ConvertFrom-Json
    if ($old.pid) {
        $p = Get-Process -Id $old.pid -ErrorAction SilentlyContinue
        if ($p) { Stop-Process -Id $old.pid -Force -ErrorAction SilentlyContinue }
    }
}

$Python = if (Test-Path "$Root\.venv\Scripts\python.exe") { "$Root\.venv\Scripts\python.exe" } else { "python" }
$Port = if ($env:MULTIMODE_PORT) { $env:MULTIMODE_PORT } else { 8501 }
$Host_ = if ($env:MULTIMODE_HOST) { $env:MULTIMODE_HOST } else { "0.0.0.0" }
$Log = Join-Path $StateDir "deploy.log"

$argList = "-m streamlit run app.py --server.port $Port --server.address $Host_ --server.headless true"
$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $Root `
    -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError (Join-Path $StateDir "deploy.err.log") `
    -PassThru

Start-Sleep -Seconds 6
$state = @{ pid = $proc.Id; port = $Port; host = $Host_; source = $env:MULTIMODE_SOURCE; started = (Get-Date).ToString("s") }
$state | ConvertTo-Json | Set-Content $StateFile -Encoding UTF8

try {
    $ok = (Invoke-WebRequest -Uri "http://localhost:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 10).Content
} catch {
    $ok = "ERR: $($_.Exception.Message)"
}
Write-Host ("Started demo system (PID=" + $proc.Id + ") -> http://localhost:" + $Port)
Write-Host ("  Health: " + $ok)
Write-Host ("  Log: " + $Log)
