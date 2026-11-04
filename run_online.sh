#!/usr/bin/env bash
# Онлайн-прогон Peleng одной командой (macOS / Linux): видео камер играют как живые, грузовик считается в момент показа.
#   ./run_online.sh /путь/к/project2 [--port 8600] [--device auto|cpu|mps|cuda:0]
# Первый запуск сам создаст backend/.venv, поставит зависимости и соберёт фронтенд (нужен Node.js 18+).
# Пересобрать фронтенд: REBUILD=1 ./run_online.sh ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${1:?укажите папку с mp4 и CSV камер: ./run_online.sh /путь/к/project2}"
shift
[ -d "$DATA" ] || { echo "Нет папки с данными: $DATA"; exit 1; }
VENV="$ROOT/backend/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  PYBIN=""
  for v in python3.12 python3.13 python3.11 python3; do command -v "$v" >/dev/null 2>&1 && { PYBIN="$v"; break; }; done
  [ -n "$PYBIN" ] || { echo "Нужен Python 3.11–3.13"; exit 1; }
  echo "== создаю backend/.venv на $PYBIN =="
  "$PYBIN" -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/backend/requirements.txt"
fi
if [ -n "${REBUILD:-}" ] || [ ! -f "$ROOT/backend/web/build/index.html" ]; then
  command -v npm >/dev/null 2>&1 || { echo "Нужен Node.js 18+ (https://nodejs.org) для сборки фронтенда"; exit 1; }
  (cd "$ROOT/frontend" && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && npm run build)
fi
echo "== онлайн-прогон: http://localhost:8600/ (или указанный --port), Ctrl+C — остановить =="
exec "$PY" -X utf8 "$ROOT/offline/online.py" --data "$DATA" "$@"
