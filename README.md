# QUARRY — Mission Control Dashboard

Live dashboard for the QUARRY autonomous spatial-memory robot project
(Cyberwave Builders Program, Cohort 2).

Right now this runs entirely on **mock/simulated data** (`backend/mock_feed.py`)
so the full dashboard — live map, telemetry, target registration, bounty
board — can be built and demoed even while hardware setup is blocked.
When the real robot pipeline (Cyberwave SDK + YOLOE + embedding matching)
is ready, it swaps in without changing the frontend at all.

## Project structure

```
quarry-robot/
├── venv/                    # your existing Python virtual environment
├── .env                     # CYBERWAVE_API_KEY lives here (never commit this)
├── .gitignore
├── requirements.txt
│
├── backend/
│   ├── __init__.py
│   ├── main.py               # FastAPI app: WebSocket + REST + serves frontend
│   └── mock_feed.py          # simulated robot state (swap for real_feed.py later)
│
└── frontend/
    ├── index.html
    ├── css/
    │   └── style.css
    └── js/
        └── app.js            # WebSocket client, map rendering, form handling
```

## Setup

```powershell
cd C:\projects\quarry-robot
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(You already have `cyberwave` and `python-dotenv` installed from Phase A —
this just adds the web framework pieces: `fastapi`, `uvicorn`, `python-multipart`.)

## Run it

```powershell
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser. You should see:
- A live radar-style map with 6 waypoints (W1–W6) and a robot marker
  patrolling between them
- Telemetry readouts (battery, FPS, detection count) updating in real time
- Amber pulses on the map for random object detections, green pulses for
  target matches
- A "Register Target" form — submit a name and it'll occasionally get
  "found" by the mock feed, appearing in the Bounty Board on the right

This proves the entire live-data pipeline end to end before any hardware
is involved.

## Swapping mock data for the real robot (Phase B/C)

`backend/mock_feed.py` defines the WebSocket message contract:

- `{"type": "init", "waypoints": [...]}`  — sent once on connect
- `{"type": "telemetry", "position": {x, y}, "waypoint": ..., "battery": ..., "fps": ...}`
- `{"type": "detection", "label": ..., "confidence": ..., "waypoint": ..., "is_target_match": bool}`

Build `backend/real_feed.py` with a `telemetry_loop(broadcast)` function
matching this same shape, pulling from:
- Cyberwave SDK for robot position (via AprilTag localization)
- YOLOE-26 for detections
- Your CLIP/OSNet embedding matcher for `is_target_match`

Then in `main.py`, change:
```python
from backend import mock_feed
```
to:
```python
from backend import real_feed as mock_feed
```
No frontend changes needed — this is exactly why the contract was kept
strict from the start.

## Known placeholders / not yet real

- `WAYPOINTS` in `mock_feed.py` are hardcoded x/y coordinates on a 0–100
  grid — replace with your actual measured AprilTag positions once
  they're placed in your test area.
- The video feed panel is a static placeholder — real camera streaming
  gets wired in during Phase B.
- Target photo uploads are accepted and stored in-memory but not yet
  embedded/processed — that's the Phase C ML work.