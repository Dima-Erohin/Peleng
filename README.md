# Peleng

3D-координаты и траектории людей в реальном времени по видео с нескольких калиброванных камер (телефоны + ноутбук).

* **[docs/USAGE.md](docs/USAGE.md) — полная инструкция по использованию** (установка, телефоны, калибровка, объекты, траектории, точность, неполадки)
* [docs/MVP.md](docs/MVP.md) — что умеет MVP, как ставить камеры и калибровать, планы
* [docs/data-format.md](docs/data-format.md) — формат WebSocket/REST для фронтенда
* [docs/CONTEXT.md](docs/CONTEXT.md) — контекст проекта и принятые решения
* **[docs/online-run.md](docs/online-run.md) — демо Project 02 одной командой (воспроизведение и онлайн-прогон):** `powershell -ExecutionPolicy Bypass -File run_online.ps1 -Data <папка с видео>` (macOS/Linux: `./run_online.sh <папка>`)
* [docs/trajectory-playback.md](docs/trajectory-playback.md) — 3D-модель грузовика, воспроизведение CSV/SVG, материалы Project 02 и контракт для бэкенда

```
backend/
  run.py              запуск сервера
  peleng/             геометрия, калибровка, YOLO-pose, сопоставление, триангуляция, трекинг, сервер
  web/                собранный интерфейс и локальные ресурсы Three.js
  scripts/            make_board.py — доска ChArUco для печати
  tests/              тесты на синтетических данных
frontend/             интерфейс React (Vite): 3D-вьювер, панель управления и камера телефона
data/                 (не в git) калибровки, сертификат, записи сессий
```

Перед запуском приложения соберите интерфейс: из `frontend/` выполните `npm ci && npm run build`.

## Установка (Windows / macOS / Linux)

Рекомендуется **Python 3.12** (3.10–3.13 тоже работают). На 3.14 torch/ultralytics пока выдают предупреждения — для хакатона лучше 3.12.

```bash
cd backend
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Запуск

Без камер (демо — виртуальная комната и люди):
```bash
python run.py --demo
```

С камерами (сначала `copy config.example.json config.json` и поправить). Только телефоны:
```bash
python run.py
```

Телефоны + вебкамера ноутбука (домашние тесты):
```bash
python run.py --webcam
```

* Ноутбук: `https://localhost:8443/` — 3D-вьювер и кнопки калибровки.
* Телефон в той же сети: `https://<IP ноутбука>:8443/cam` (адрес печатается при запуске) → принять предупреждение о сертификате → ввести ID → «Старт».
* Windows может спросить про брандмауэр — разрешить доступ для частных сетей.

Доска для калибровки:
```bash
python scripts/make_board.py
```

Тесты:
```bash
python -m pytest tests
```

Повтор записанной сессии:
```bash
python run.py --replay ../data/sessions/session_YYYYMMDD_HHMMSS.jsonl
```
