"""Дообучение YOLO (yolo11n по умолчанию) на синтетическом датасете дрона (см. drone_synth.py).

Пример (Mac M-серии, MPS; Linux с GPU — device=0; без GPU — cpu):

    python offline/detectors/drone_synth.py --data /путь/к/project2 --out ./drone_ds --n-train 6000 --n-val 600
    python offline/detectors/train_drone.py --data ./drone_ds/data.yaml --device mps --epochs 40 --out ./runs

Результат: <out>/drone/weights/best.pt — подставляется в TrainedDrone(weights=...).
Веса в репозиторий не кладём.
Время (ориентир): CPU 4 ядра, yolo11n, imgsz 640 — несколько минут на 100 картинок/эпоху → для 6000 картинок
ночь не хватит на много эпох; на M-серии (mps) в разы быстрее; на GPU — минуты.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml из drone_synth.py")
    ap.add_argument("--weights", default="yolo11n.pt", help="стартовые веса (скачаются автоматически)")
    ap.add_argument("--device", default="cpu", help="cpu | mps | 0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0, help="доля train-выборки (для пробных запусков)")
    ap.add_argument("--time", type=float, default=None, help="лимит времени обучения в часах (ultralytics подстроит эпохи)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    from ultralytics import YOLO

    m = YOLO(a.weights)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
            fraction=a.fraction, time=a.time, project=a.out, name="drone", exist_ok=True, patience=15,
            # мелкие объекты: мозаика полезна, масштаб/повороты уже в синтетике — внутренние аугментации умеренные
            mosaic=1.0, scale=0.3, degrees=0.0, fliplr=0.5, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, plots=False, amp=(a.device != "mps"))
    print("лучшие веса:", f"{a.out}/drone/weights/best.pt")


if __name__ == "__main__":
    main()
