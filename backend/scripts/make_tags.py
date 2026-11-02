"""Сгенерировать метки ArUco для объектов — печать в точном масштабе (PDF + PNG, 300 dpi, A4).

    python scripts/make_tags.py                    # метки 0..5, сторона 50 мм
    python scripts/make_tags.py --ids 0-11 --size-mm 80

Словарь DICT_5X5_100 (DICT_4X4_50 занят доской калибровки). Печатать в масштабе 100 %
(«Фактический размер», не «Вписать»). Размер метки — сторона чёрного квадрата; после печати измерьте
его линейкой и впишите в config.json -> "objects": {"tag_size_m": ...} (нужно для режима одной камеры).
Вокруг метки должно остаться белое поле — вырезайте по пунктиру.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from peleng.config import DATA_DIR  # noqa: E402
from peleng.objects import TAG_DICT  # noqa: E402

DPI = 300
A4_MM = (210, 297)


def parse_ids(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0-5", help="ID меток: 0-5 или 1,4,7")
    ap.add_argument("--size-mm", type=float, default=50.0, help="сторона чёрного квадрата, мм")
    ap.add_argument("--dict", default=TAG_DICT)
    ap.add_argument("--out", default=str(DATA_DIR))
    args = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    px_per_mm = DPI / 25.4
    side = round(args.size_mm * px_per_mm)
    pad = max(round(6 * px_per_mm), side // 8)  # белое поле вокруг метки (нужно детектору)
    cell = side + 2 * pad
    cap = round(6 * px_per_mm)  # строка подписи
    pw, ph = (round(v * px_per_mm) for v in A4_MM)
    margin = round(10 * px_per_mm)
    cols = max(1, (pw - 2 * margin) // cell)
    rows = max(1, (ph - 2 * margin) // (cell + cap))
    ids = parse_ids(args.ids)

    try:
        font = ImageFont.load_default(size=round(3.5 * px_per_mm))
    except TypeError:  # старый Pillow
        font = ImageFont.load_default()
    pages = []
    for k, tag_id in enumerate(ids):
        if k % (cols * rows) == 0:
            pages.append(Image.new("L", (pw, ph), 255))
        page = pages[-1]
        draw = ImageDraw.Draw(page)
        n = k % (cols * rows)
        x0 = margin + (n % cols) * cell
        y0 = margin + (n // cols) * (cell + cap)
        img = cv2.aruco.generateImageMarker(dictionary, tag_id, side)
        page.paste(Image.fromarray(img), (x0 + pad, y0 + pad))
        # пунктир для вырезания и подпись
        for i in range(0, cell, 24):
            for (a, b) in (((x0 + i, y0), (x0 + min(i + 12, cell), y0)), ((x0 + i, y0 + cell), (x0 + min(i + 12, cell), y0 + cell))):
                draw.line([a, b], fill=170, width=2)
            for (a, b) in (((x0, y0 + i), (x0, y0 + min(i + 12, cell))), ((x0 + cell, y0 + i), (x0 + cell, y0 + min(i + 12, cell)))):
                draw.line([a, b], fill=170, width=2)
        draw.text((x0 + pad, y0 + cell + 4), f"tag:{tag_id}  {args.dict}  {args.size_mm:g} mm", fill=0, font=font)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"tags_{args.dict}_{args.size_mm:g}mm_{ids[0]}-{ids[-1]}"
    pages[0].save(f"{stem}.pdf", resolution=DPI, save_all=True, append_images=pages[1:])
    pages[0].save(f"{stem}.png", dpi=(DPI, DPI))
    print(f"Готово: {stem}.pdf ({len(pages)} стр.) / .png (первая страница)")
    print(f'В config.json: "objects": {{"mode": "aruco", "tag_size_m": {args.size_mm / 1000:g}, "tag_dictionary": "{args.dict}"}}')


if __name__ == "__main__":
    main()
