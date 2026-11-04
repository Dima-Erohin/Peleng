# Онлайн-прогон Peleng одной командой (Windows): видео камер играют как живые, грузовик считается в момент показа.
# Пример:
#   powershell -ExecutionPolicy Bypass -File run_online.ps1 -Data "C:\Programming\project2"
# Параметры: -Data папка с Way1.FinalImage.CineCameraActorN.mp4 и object_transform_*.csv;
#            -Port (8600), -Device (auto | cpu | cuda:0), -Rebuild — пересобрать фронтенд.
# Первый запуск сам создаст backend\.venv, поставит зависимости и соберёт фронтенд (нужен Node.js 18+).
param(
    [Parameter(Mandatory=$true)][string]$Data,
    [int]$Port = 8600,
    [string]$Device = "auto",
    [switch]$Rebuild
)
$ErrorActionPreference = "Continue"
function Check($what) { if ($LASTEXITCODE -ne 0) { Write-Host "ОШИБКА на шаге: $what (код $LASTEXITCODE)"; exit 1 } }
$Root = $PSScriptRoot
$Venv = Join-Path $Root "backend\.venv"
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Data)) { Write-Host "Нет папки с данными: $Data"; exit 1 }

if (-not (Test-Path $Py)) {
    # лучше всего Python 3.12; если его нет — первая установленная из 3.13 / 3.11 / 3.14
    $PyVer = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in "3.12", "3.13", "3.11", "3.14") {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { $PyVer = $v; break }
        }
    }
    if ($PyVer) { Write-Host "== создаю backend\.venv на Python $PyVer =="; & py "-$PyVer" -m venv $Venv }
    else { Write-Host "== создаю backend\.venv на python из PATH =="; python -m venv $Venv }
    Check "venv"
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r (Join-Path $Root "backend\requirements.txt")
    Check "зависимости (backend\requirements.txt)"
}

$Index = Join-Path $Root "backend\web\build\index.html"
if ($Rebuild -or -not (Test-Path $Index)) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Write-Host "Нужен Node.js 18+ (https://nodejs.org) для сборки фронтенда"; exit 1 }
    Push-Location (Join-Path $Root "frontend")
    if (-not (Test-Path "node_modules")) { npm ci --no-audit --no-fund; Check "npm ci" }
    npm run build
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Write-Host "ОШИБКА: сборка фронтенда"; exit 1 }
}

Write-Host "== онлайн-прогон: http://localhost:$Port/ (Ctrl+C — остановить) =="
& $Py -X utf8 (Join-Path $Root "offline\online.py") --data $Data --port $Port --device $Device
