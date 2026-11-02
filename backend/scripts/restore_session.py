"""Восстановить траектории из записи сессии и выгрузить CSV/JSON.

    python scripts/restore_session.py ../data/sessions/session_20261103_120000.jsonl
    python scripts/restore_session.py запись.jsonl --max-gap 1.0 --rate 60 --out результаты/ --raw

Пишет <имя>_restored.csv и <имя>_restored.json (с --raw — ещё <имя>_raw.csv/.json) рядом с записью
или в папку --out, и печатает метрики треков.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import trajectory as tj  # noqa: E402
from peleng.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Восстановление траекторий из записи .jsonl")
    ap.add_argument("session", help="файл записи .jsonl")
    ap.add_argument("--max-gap", type=float, default=None, help="макс. пропуск для сшивки/заполнения, с")
    ap.add_argument("--rate", type=float, default=None, help="частота сетки на выходе, Гц")
    ap.add_argument("--out", default=None, help="папка для файлов (по умолчанию — рядом с записью)")
    ap.add_argument("--raw", action="store_true", help="выгрузить и сырые данные")
    args = ap.parse_args()

    src = Path(args.session)
    if not src.is_file():
        sys.exit(f"нет файла: {src}")
    out_dir = Path(args.out) if args.out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    res = tj.restore_session(src, load_config(), max_gap_s=args.max_gap, rate_hz=args.rate)
    files = []
    for raw in ([False, True] if args.raw else [False]):
        tag = "raw" if raw else "restored"
        p = out_dir / f"{src.stem}_{tag}.csv"
        p.write_text(tj.to_csv(res, raw=raw), encoding="utf-8")
        files.append(p)
        p = out_dir / f"{src.stem}_{tag}.json"
        p.write_text(json.dumps({**tj.to_json(res, raw=raw), "session": src.name}, ensure_ascii=False), encoding="utf-8")
        files.append(p)

    s = tj.summary(res)
    print(f"{src.name}: треков в записи {s['raw_tracks']}, после сшивки и восстановления {s['tracks']}; параметры {s['params']}")
    head = f"{'трек':<14}{'из ID':<14}{'длит., с':>9}{'путь, м':>9}{'сырой, м':>10}{'ср. м/с':>9}{'макс м/с':>9}{'пропуски':>9}{'запол., с':>10}{'СКО,см':>7}"
    print(head)
    print("-" * len(head))
    for it in s["items"]:
        name = f"{it['kind']} {it['id']}" + (f" ({it['cls']})" if it.get("cls") else "")
        print(f"{name:<14}{','.join(map(str, it['merged_ids'])):<14}{it['duration_s']:>9.2f}{it['path_m']:>9.2f}{it['path_raw_m']:>10.2f}"
              f"{it['speed_mean_mps']:>9.2f}{it['speed_max_mps']:>9.2f}{it['gaps_filled']:>9}{it['gaps_filled_s']:>10.2f}{it['std_mean_m'] * 100:>7.1f}")
    for p in files:
        print(f"-> {p}")


if __name__ == "__main__":
    main()
