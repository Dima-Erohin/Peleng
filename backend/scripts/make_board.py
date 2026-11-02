"""Сгенерировать доску ChArUco для печати в точном масштабе (PDF + PNG, 300 dpi).

    python scripts/make_board.py            # A4: клетка 50 мм
    python scripts/make_board.py --a3       # A3: клетка 70 мм (видно дальше — для комнаты)

Печатать в масштабе 100 % («Фактический размер», не «Вписать»). После печати измерьте
клетку линейкой и впишите реальный размер в backend/config.json -> board.square_m / marker_m.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from peleng.calibration import Board  # noqa: E402
from peleng.config import DATA_DIR  # noqa: E402

DPI = 300
PAPER_MM = {"A4": (297, 210), "A3": (420, 297)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3", action="store_true")
    ap.add_argument("--out", default=str(DATA_DIR))
    args = ap.parse_args()

    paper = "A3" if args.a3 else "A4"
    cfg = {"square_m": 0.07, "marker_m": 0.052} if args.a3 else {"square_m": 0.05, "marker_m": 0.037}
    board = Board(cfg)
    px_per_mm = DPI / 25.4
    sq_px = round(board.square_m * 1000 * px_per_mm)
    img = board.board.generateImage((board.squares_x * sq_px, board.squares_y * sq_px), marginSize=0, borderBits=1)
    pw, ph = (round(v * px_per_mm) for v in PAPER_MM[paper])
    page = Image.new("L", (pw, ph), 255)
    page.paste(Image.fromarray(img), ((pw - img.shape[1]) // 2, (ph - img.shape[0]) // 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"charuco_{paper}_{board.squares_x}x{board.squares_y}_{int(board.square_m * 1000)}mm"
    page.save(f"{stem}.pdf", resolution=DPI)
    page.save(f"{stem}.png", dpi=(DPI, DPI))
    print(f"Готово: {stem}.pdf / .png")
    print(f'В config.json: "board": {{"squares_x": {board.squares_x}, "squares_y": {board.squares_y}, '
          f'"square_m": {board.square_m}, "marker_m": {board.marker_m}, "dictionary": "{board.dict_name}"}}')


if __name__ == "__main__":
    main()
