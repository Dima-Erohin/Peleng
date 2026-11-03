# Формат данных (бэкенд ↔ фронтенд)

Единицы — **метры, секунды**. Система координат мира: **Z вверх**, пол — плоскость `z = 0`.
Начало координат — **центр доски ChArUco** при калибровке; ось X идёт вдоль длинной стороны доски.
Three.js по умолчанию Y-up: либо ставить `THREE.Object3D.DEFAULT_UP.set(0,0,1)` (так сделано во вьювере `backend/web/viewer-engine.js`),
либо конвертировать `(x, y, z) → (x, z, -y)`.

## WebSocket `/ws/stream` (сервер → браузер)

Все сообщения — JSON с полем `type`.

### `hello` — сразу после подключения
```json
{
  "type": "hello",
  "mode": "live | demo | replay",
  "server_time": 1730455200.12,
  "detector": "yolo11n-pose.pt (cpu)",
  "recording": null,
  "cameras": [ <camera>, ... ],
  "board": { "squares_x": 5, "squares_y": 4, "square_m": 0.05, "marker_m": 0.037, "height_m": 0.0,
             "size_m": [0.25, 0.2], "corners_world": [[x,y,z] x4] },
  "room": { "size_m": [5, 4, 2.7] | null, "board_offset_m": [0, 0] }
}
```

`<camera>`:
```json
{
  "id": "redmi", "name": "redmi", "kind": "push | opencv | demo",
  "online": true, "fps": 15.0, "size": [1280, 720],
  "calibrated": true,
  "intrinsics": "approx | approx+focal | board | demo",
  "hfov_deg": 68.0,
  "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "R": [[...],[...],[...]], "t": [tx,ty,tz],      // X_cam = R·X_world + t (OpenCV: x вправо, y вниз, z вперёд)
  "center": [x,y,z],                                // положение камеры в мире
  "extrinsics_reproj_px": 0.4
}
```

### `cameras` — изменился состав/калибровка камер
`{ "type": "cameras", "cameras": [ <camera>, ... ] }`

### `frame` — результат обработки (10–30 раз в секунду)
```json
{
  "type": "frame",
  "t": 1730455201.533,          // время кадра (часы сервера), с
  "seq": 4521,
  "persons": [
    {
      "id": 3,                  // стабильный ID трека
      "pos": [1.42, 2.10, 0.95],  // центр таза (сглажен фильтром Калмана)
      "floor": [1.42, 2.10, 0.0], // точка на полу под человеком — для траектории/тепловой карты
      "vel": [0.31, -0.05, 0.0],  // м/с
      "speed": 0.32,              // горизонтальная скорость, м/с
      "mode": "tri | mono",       // tri — ≥2 камер (триангуляция), mono — 1 камера (допущение «ноги на полу»)
      "cams": ["redmi", "laptop"],
      "reproj_err_px": 2.7,       // качество триангуляции (null для mono)
      "predicted": false,         // true — человека сейчас не видно, позиция — прогноз
      "kp3d": [[x,y,z] | null, ... 17]  // скелет COCO-17 (только для tri), иначе null
    }
  ],
  "detections": {               // 2D для панели с видео, в пикселях кадра размера frame_size
    "redmi": [ { "bbox": [x1,y1,x2,y2], "score": 0.91, "kp": [[x,y,conf] x17] } ]
  },
  "frame_size": { "redmi": [640, 360], "laptop": [1280, 720] },  // телефоны при трекинге шлют уменьшенные кадры (pipeline.track_width)
  "stats": { "fps": 12.5, "proc_ms": 80, "detect_ms": 70, "latency_ms": 140, "skew_ms": 25, "cams_used": ["redmi","laptop"] }
  // skew_ms — разброс времени съёмки между камерами пачки (с кадрами по команде обычно единицы–десятки мс)
}
```

**`pos_std_m`** (расширение «Точность», [accuracy.md](accuracy.md)): у элементов `persons` (и `objects`, если есть) с `mode: "tri"` —
`[σx, σy, σz]`, м, оценка ошибки положения (1σ) из геометрии камер, видевших точку, при шуме 2 px (`accuracy.sigma_px`).
Не учитывает ошибки калибровки; у `mono` поля нет.

Порядок точек COCO-17: `nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle`.

### Поле `objects` в `frame` — отслеживаемые предметы (расширение «Объекты», [objects.md](objects.md))
```jsonc
"objects": [
  { "id": 101, "kind": "object", "cls": "sports ball",   // класс COCO или "tag:7" для метки ArUco (DICT_5X5_100)
    "pos": [x,y,z], "floor": [x,y,0], "vel": [vx,vy,vz], "speed": 0.4, "mode": "tri | mono",
    "cams": ["redmi","iphone15"], "reproj_err_px": 1.2, "predicted": false }
],
"object_detections": { "redmi": [ { "cls": "sports ball", "bbox": [x1,y1,x2,y2], "score": 0.8 } ] }  // 2D, только если режим включён
```
Поля как у `persons` (без `kp3d`); `pos` — центр объекта. ID общие с людьми и не пересекаются.
Режим выключен (`off`, по умолчанию) — `"objects": []`. В `stats` добавляется `objects_ms`.
REST: `GET/POST /api/objects/` (режим, классы), `POST /api/objects/reset`.

### Поле `zones` в `frame` — зоны и события (расширение «Зоны и события», [zones.md](zones.md))
Появляется, если на полу нарисована хотя бы одна зона/линия (или в кадре были события):
```jsonc
"zones": {
  "inside": { "z1": [ { "kind": "person", "id": 2, "dwell_s": 3.4 } ] },   // кто сейчас внутри каждой зоны
  "lines":  { "l1": { "a_to_b": 5, "b_to_a": 2 } },                         // счётчики пересечений линий
  "events": [                                                                // новые события за этот кадр
    { "n": 17, "type": "enter | exit | line_cross", "t": 1730455201.5, "kind": "person", "id": 2,
      "zone": "z1", "name": "Дверь", "dwell_s": 3.4, "reason": null, "pos": [x, y] }   // exit: dwell_s, reason: null | "lost" | "reset" | "end"
  ]                                                                          // line_cross: "line", "name", "direction": "A→B | B→A"
}
```
Поле попадает и в запись сессии. REST — `/api/zones/…` (зоны, линии, журнал, тепловая карта, анализ записи).

### `room_points` — «комната точками»
```json
{ "type": "room_points", "points": [[x,y,z], ...], "colors": [[r,g,b], ...], "pairs": [{"a":"redmi","b":"laptop","matches":120,"points":80}] }
```

## Запись сессии (`data/sessions/*.jsonl`)
Одна строка — одно сообщение: первая строка `hello`, дальше `frame`. Тот же формат, что в WebSocket,
поэтому `python run.py --replay файл.jsonl` проигрывает запись во вьювер, а фронтенд может читать файл напрямую.

### Кадры камер рядом с записью (`session_X_frames/`)
Вместе с `session_X.jsonl` сохраняется папка `session_X_frames/` — чтобы при `--replay` во вьювере были превью камер и текстуры на пирамидках:
```
session_X_frames/
  index.jsonl              — по строке на кадр: {"cam": "iphone15", "t": 1730000000.1234, "file": "iphone15/1730000000123.jpg"}
  <камера>/<t в мс>.jpg    — уменьшенные кадры (ширина ≤ 480 px, JPEG quality 70, ≤ 5 к/с на камеру)
```
`t` — время кадра в часах сервера, то же, что `t` у сообщений `frame`; при replay камера показывает кадр, ближайший к `t` текущего сообщения.
Без папки (старые записи) replay работает как раньше, без превью. Настройки: `cfg["recording"]["frames"]` — `enabled`, `max_width`, `fps`, `quality`.

### Восстановление траекторий и экспорт
Запись можно «восстановить» (сшивка обрывков, сглаживание вперёд-назад, заполнение пропусков на сетке 30 Гц)
и выгрузить в CSV `t, kind, id, x, y, z, vx, vy, vz, speed, source, std` или JSON — сырые и восстановленные данные:
блок «Траектории» во вьювере, `GET /api/trajectory/sessions | restore | export`, `scripts/restore_session.py`.
Подробно — [trajectory.md](trajectory.md).

## REST
| Метод | Путь | Что делает |
|---|---|---|
| GET | `/api/state` | то же, что `hello` + статистика |
| POST | `/api/calib/extrinsics` | положение всех камер по доске (`{"cameras": [..]}` — опционально) |
| POST | `/api/calib/{id}/intrinsics/start` | сбор кадров для калибровки оптики (`{"n": 15}`) |
| GET | `/api/calib/{id}/intrinsics` | статус сбора |
| POST | `/api/calib/{id}/hfov` | задать угол обзора вручную `{"hfov_deg": 68}` |
| POST | `/api/calib/{id}/reset` | сбросить положение камеры |
| POST | `/api/room/scan`, `/api/room/clear` | облако точек комнаты |
| POST | `/api/record/start`, `/api/record/stop` | запись сессии |
| POST | `/api/tracks/reset` | сбросить треки |
| GET | `/api/cameras/{id}/mjpeg`, `/snapshot.jpg` | видео камеры с наложенными скелетами |
| GET | `/api/board.png` | картинка доски |
| GET | `/api/demo/truth` | (демо) истинные позиции людей — для проверки точности |
| * | `/api/accuracy/...` | проверки точности (доска, рулетка), отчёт — см. [accuracy.md](accuracy.md) |
| GET | `/api/coverage?cell_m=0.2&height_m=0.9` | карта покрытия пола: сколько камер видят каждую клетку и ожидаемая σ — см. [coverage.md](coverage.md) |
| * | `/api/chain/...` | калибровка цепочкой по парам камер (снимки, расчёт, применить); сообщение `{"type": "chain"}` — см. [chain-calibration.md](chain-calibration.md) |

## Приём кадров с телефона: WebSocket `/ws/push/{cam_id}`
* текст `{"type":"hello","name":...,"width":...,"height":...}`;
* текст `{"type":"ping","tc":<Date.now()>}` → ответ `{"type":"pong","tc":...,"ts":<время сервера, с>}` (синхронизация часов);
* бинарное: `[8 байт float64 LE — время кадра в часах сервера][JPEG]`.

Свой iOS-клиент (например, с глубиной лидара) может слать в этот же эндпоинт.

## JSON комнаты (RoomPlan) — черновик, пока не реализовано
```json
{
  "units": "m", "up": "z", "origin": "board_center",
  "walls":   [ { "id": "w1", "dimensions": [w, h, d], "transform": [16 чисел, по строкам] } ],
  "doors":   [ ... ], "windows": [ ... ],
  "objects": [ { "id": "o1", "category": "table|chair|sofa|storage|bed|...", "dimensions": [w,h,d], "transform": [16] } ],
  "alignment": { "room_points": [[x,y,z] x3], "world_points": [[x,y,z] x3] }
}
```
