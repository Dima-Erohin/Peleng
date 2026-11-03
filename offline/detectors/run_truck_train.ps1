# Обучение детектора грузовика (YOLO-pose, pivot) одной командой на Windows (CPU).
# Пример:
#   powershell -ExecutionPolicy Bypass -File offline\detectors\run_truck_train.ps1 -Data "D:\project2" -Out "D:\truck_train"
# Параметры: -Data папка с Way1.FinalImage.CineCameraActorN.mp4 и CSV; -Out рабочая папка (датасет, веса);
#            -Epochs (по умолчанию 40), -Imgsz (640), -Batch (8), -Exp (All | A | B).
# Повторный запуск той же командой продолжает с последней эпохи (resume) и не перегенерирует датасет.
param(
    [Parameter(Mandatory=$true)][string]$Data,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$Epochs = 40,
    [int]$Imgsz = 640,
    [int]$Batch = 8,
    [string]$Exp = "All",
    [string]$Weights = "yolo11n-pose.pt"
)
$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Venv = Join-Path $Out "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "== создаю venv (нужен Python 3.12: py -3.12) =="
    py -3.12 -m venv $Venv
    & $Py -m pip install --upgrade pip
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    & $Py -m pip install -r (Join-Path $PSScriptRoot "requirements_truck.txt")
    & $Py -m pip install -r (Join-Path $Repo "backend\requirements.txt")
}
$Ds = Join-Path $Out "truck_ds"
if (-not (Test-Path (Join-Path $Ds "data_All.yaml"))) {
    Write-Host "== генерирую датасет (≈15-30 минут) =="
    & $Py -I (Join-Path $PSScriptRoot "truck_dataset.py") --data $Data --out $Ds --stride 2
}
Write-Host "== обучение ($Exp, $Epochs эпох, imgsz $Imgsz, CPU); чекпойнт каждую эпоху =="
& $Py -I (Join-Path $PSScriptRoot "train_truck_pose.py") --data (Join-Path $Ds "data_$Exp.yaml") --name $Exp `
    --device cpu --workers 2 --batch $Batch --imgsz $Imgsz --epochs $Epochs --weights $Weights --out (Join-Path $Out "runs")
$Best = Join-Path $Out "runs\$Exp\weights\best.pt"
if (Test-Path $Best) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Out "RESULT") | Out-Null
    Copy-Item $Best (Join-Path $Out "RESULT\truck_pose_$Exp.pt") -Force
    Write-Host "ГОТОВО: веса -> $Out\RESULT\truck_pose_$Exp.pt (пришлите этот файл)"
}
