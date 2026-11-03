"""Обучение YOLO-pose (1 класс truck, 1 keypoint = pivot) на выборке из truck_dataset.py.

Примеры:
    # диагностика обобщения: train кам.1–4, val кам.5 (в data_A.yaml); B — наоборот (train 2–5, val 1)
    python -I offline/detectors/train_truck_pose.py --data <ds>/data_A.yaml --name A --device cpu --epochs 12
    # финальная модель на всех камерах
    python -I offline/detectors/train_truck_pose.py --data <ds>/data_All.yaml --name All --device mps --epochs 60

Аугментации ultralytics (онлайн): масштаб 0.3–1.7 (scale=0.7), повороты ±10°, перспектива, shear, отражение,
mosaic, HSV; плюс офлайн (в aug/): листва поверх грузовика, размытие, шум, JPEG.
Результат: <out>/<name>/weights/best.pt (метрики pose mAP в results.csv; px-ошибку точки считает eval_truck_pose.py).
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo11n-pose.pt", help="yolo11n-pose.pt | yolo11s-pose.pt (скачаются сами)")
    ap.add_argument("--name", default="truck_pose")
    ap.add_argument("--device", default="cpu", help="cpu | mps | 0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time", type=float, default=None, help="лимит времени, часы")
    ap.add_argument("--fraction", type=float, default=1.0, help="доля train (для пробных запусков)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    from pathlib import Path

    from ultralytics import YOLO

    last = Path(a.out) / a.name / "weights" / "last.pt"
    if last.exists():  # обрыв/перезапуск: продолжаем с последней эпохи (last.pt пишется каждую эпоху)
        print("найден чекпойнт, продолжаю:", last)
        YOLO(str(last)).train(resume=True)
        print("веса:", f"{a.out}/{a.name}/weights/best.pt")
        return
    m = YOLO(a.weights)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers, time=a.time, fraction=a.fraction,
            project=str(Path(a.out).resolve()), name=a.name, exist_ok=True, patience=100, plots=False, save_period=-1,
            scale=0.7, degrees=10.0, translate=0.2, shear=3.0, perspective=0.0008, fliplr=0.5, flipud=0.0,
            mosaic=1.0, close_mosaic=3, mixup=0.1, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, pose=24.0, kobj=2.0, box=7.5)
    print("веса:", f"{a.out}/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
