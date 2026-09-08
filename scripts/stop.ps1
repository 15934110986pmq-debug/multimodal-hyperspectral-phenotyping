$Root = Split-Path -Parent $PSScriptRoot
$StateFile = Join-Path $Root "output\deploy_state.json"
if (Test-Path $StateFile) {
    $s = Get-Content $StateFile -Raw | ConvertFrom-Json
    if ($s.pid) {
        $p = Get-Process -Id $s.pid -ErrorAction SilentlyContinue
        if ($p) { Stop-Process -Id $s.pid -Force; Write-Host ("Stopped PID " + $s.pid) }
        else { Write-Host ("PID " + $s.pid + " not running") }
    }
    Remove-Item $StateFile -ErrorAction SilentlyContinue
} else {
    Write-Host "No deploy state file found"
}
