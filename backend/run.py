"""Запуск сервера Peleng.

    python run.py                 # реальные камеры (config.json + телефоны через /cam)
    python run.py --demo          # без камер: виртуальная комната и люди
    python run.py --replay data/sessions/session_....jsonl
    python run.py --no-https      # только для ноутбука (телефон без HTTPS камеру не отдаст)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from peleng.certs import ensure_cert, local_ips  # noqa: E402
from peleng.config import DATA_DIR, load_config  # noqa: E402
from peleng.server import create_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Peleng — 3D-координаты людей по нескольким камерам")
    ap.add_argument("--config", help="путь к config.json (по умолчанию backend/config.json, если есть)")
    ap.add_argument("--demo", action="store_true", help="демо без камер")
    ap.add_argument("--replay", help="проиграть записанную сессию .jsonl")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-https", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    port = args.port or cfg["server"]["port"]
    use_https = cfg["server"].get("https", True) and not args.no_https
    app = create_app(cfg, demo=args.demo, replay=args.replay)

    scheme = "https" if use_https else "http"
    ssl = {}
    if use_https:
        cert, key = ensure_cert(DATA_DIR / "certs")
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

    print("\n=== Peleng ===")
    print(f"  Вьювер на ноутбуке:  {scheme}://localhost:{port}/")
    for ip in local_ips():
        print(f"  С телефона (камера): {scheme}://{ip}:{port}/cam")
        print(f"  Вьювер по сети:      {scheme}://{ip}:{port}/")
    if use_https:
        print("  Сертификат самоподписанный: в браузере нажмите «Дополнительно → Перейти на сайт».")
    print()
    uvicorn.run(app, host=cfg["server"]["host"], port=port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
