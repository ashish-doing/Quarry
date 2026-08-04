# QUARRY

**Cyberwave Builders Program — Cohort 2**
*FIELDWORK · ORION · THE OUTRIDER*

[![Architecture](https://img.shields.io/badge/📐%20ARCHITECTURE-DEEP%20DIVE-4ADE80?style=for-the-badge)](./ARCHITECTURE.md)
[![Quick Start](https://img.shields.io/badge/⚡%20QUICK%20START-GO-4ADE80?style=for-the-badge)](#quick-start)
[![Limitations](https://img.shields.io/badge/📍%20HONEST%20LIMITS-READ-F5A623?style=for-the-badge)](#known-limitations--what-this-honestly-cannot-do-yet)
[![Tests](https://img.shields.io/badge/Tests-16%20passing-brightgreen?style=for-the-badge&logo=pytest)](#tests)
[![YOLOE](https://img.shields.io/badge/YOLOE--26-Open--Vocab%20Detection-4ADE80?style=for-the-badge)](#tech-stack)
[![OSNet](https://img.shields.io/badge/OSNet-Re--ID-4ADE80?style=for-the-badge)](#the-data-flywheel--fine-tuning)

---

> *A quarry is the thing being hunted. QUARRY is a real ground robot, hunting real targets registered before the round starts — with a team behind it, watching, voting, and calling the shots.*

## What It Actually Is

Underneath the hunting game is a real, named ML problem: **human-in-the-loop verification to reduce false positives in autonomous vision** — the same architecture behind citizen-science wildlife camera-trap labeling and content-moderation triage. Every confirmed or disputed vote in QUARRY is a human-verified label. The multiplayer voting mechanic isn't the point; it's the data engine. Competitive play is the incentive structure for getting fast, high-quality labels out of people who'd otherwise have no reason to sit and label camera-trap crops.

Wrapped around that: a real multi-site hub. Any number of Field Agents, each with their own real UGV in their own physical space, patrol independently while Spectators join a shared session to watch any agent's live feed, vote to confirm sightings, and push their team up a shared leaderboard. Nobody ever drives another person's robot — enforced in the architecture, not just the UI.

**Core loop:** a UGV (Waveshare UGV Beast, no lidar, vision-only) patrols on a Raspberry Pi, streaming telemetry and camera frames to a companion laptop. The laptop runs real-time open-vocabulary detection (YOLOE-26) plus appearance-based re-identification (OSNet) to recognize registered targets even after they change appearance. Confirmed sightings score real points onto a team leaderboard — and every vote outcome, confirmed or disputed, becomes a labeled training example for the re-identification model.

---

## What Actually Works Right Now

| Capability                                                                      | Status                                                                              |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Cyberwave pairing, live camera frame retrieval (`get_frame(source='remote_edge')`) | ✅ Confirmed — resolved after `get_latest_frame()`/`get_frame(source='cloud')` both failed |
| Manual control (gamepad teleop — drive, camera pan/tilt, lights, photo capture)  | ✅ Verified on real hardware                                                          |
| Independent safety watchdog (0.5s auto-stop on command loss)                     | ✅ Verified — killed the control process mid-drive, confirmed auto-stop               |
| CollisionGuard (proximity emergency-stop, both patrol and teleop paths)          | ✅ Verified — started explicitly in `site_agent.py`, was previously dead code in the multi-site path |
| Multi-site Hub (teams, points, voting, rate-limited anti-spam identity)          | ✅ 16/16 automated tests passing                                                      |
| Real hardware → Hub relay (`site_agent.py`), two simultaneous real robots        | ✅ Confirmed live                                                                     |
| CV detection pipeline (YOLOE-26 + OSNet), temporal smoothing, EMA confidence     | ✅ Confirmed against real live frames                                                 |
| Multi-photo target registration (3–5 angles, best-of-gallery matching)           | ✅ Wired through Hub + local same-machine endpoint                                    |
| Digital twin position sync (`set_pose()`)                                        | ✅ Confirmed visibly moving in Cyberwave's own twin view                              |
| Data flywheel capture (confirmed/disputed crops → `training_data/`)              | ✅ Wired end-to-end in `site_agent.py`                                                |
| Analytics dashboard (Chart.js — team comparison, top hunters, catch feed)        | ✅ Live at `/dashboard`                                                               |
| AprilTag pose pipeline (calibration, detection, venue measurement tool)          | ⏳ Code complete, blocked on physical camera access — see [Known Limitations](#known-limitations--what-this-honestly-cannot-do-yet) |
| OSNet head fine-tuning                                                           | ⏳ Script complete and dry-run verified, blocked on real training data volume         |
| Monocular VO feasibility                                                         | ⏳ Test script complete, not yet run — go/no-go pending, see below                    |
| Autonomous patrol, voice control                                                 | ✅ Built (sweep-pattern patrol, not point-to-point — see limitations)                 |

---

## Architecture

```
flowchart LR
    subgraph FieldAgent["FIELD AGENT (their own machine, their own robot)"]
        UGV["UGV Beast\nPi 4B + ESP32\nno lidar, vision-only"]
        UGV -- "MQTT via Cyberwave edge-core" --> SDK["Cyberwave Python SDK"]
        SDK --> RF["real_feed.py\nYOLOE-26 detection + temporal smoothing\nOSNet re-ID\nCollisionGuard"]
        RF --> SA["site_agent.py\nrelays telemetry + candidates\nserves annotated video locally\nsaves confirmed/disputed crops"]
    end

    SA -- "WebSocket" --> HUB["FastAPI Hub (main.py)\nteams · points · voting\nrate-limited identity"]
    HUB -- "WebSocket" --> SPEC["Spectator browsers\nMission Control HUD · Vote · Chat"]
    HUB -- "/api/state" --> DASH["Analytics dashboard\nChart.js"]

    RF -.->|"confirmed/disputed crops"| TD["training_data/\nconfirmed/&lt;target&gt;/\ndisputed/&lt;target&gt;/"]
    TD --> FT["finetune_osnet_head.py\nfrozen-backbone linear head\nhonest before/after eval"]

    subgraph Control["Driving — never through the Hub"]
        TP["teleop.py — gamepad"]
        PT["patrol.py — autonomous sweep"]
        VC["voice_control.py — Groq Whisper"]
    end
    Control -.->|"direct SDK calls,\nlocal machine only"| SDK
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full WebSocket message contract and sequence flows.

---

## The Data Flywheel & Fine-Tuning

Every vote is a labeled example, not a throwaway UI interaction:

1. `real_feed.py` raises a candidate → crop is held locally, never sent to the Hub.
2. A team votes confirm/dispute. `site_agent.py` correlates the Hub's broadcast candidate ID back to the locally-held crop by matching `(label, waypoint)`.
3. The crop is saved to `training_data/confirmed/<target>/` or `training_data/disputed/<target>/` — a running per-label confirm/dispute tally is kept in `training_data/stats.json`.
4. `finetune_osnet_head.py` trains a linear classification head on top of the frozen OSNet backbone (VRAM-budget-appropriate for the RTX 4050 alongside YOLOE-26), evaluated against a held-out split, reported honestly alongside sample counts — not a bare percentage that overstates confidence a dozen crops can't support.

**Current state, honestly:** the pipeline is built and its data-volume gate has been verified to correctly refuse running on insufficient/wrong data (see [Known Limitations](#known-limitations--what-this-honestly-cannot-do-yet)). It has not yet been run against real training data, because real hunt sessions haven't accumulated any yet. This is a designed-but-blocked state, not a hidden gap — same posture as OSNet fine-tuning was in earlier project documentation.

---

## Screenshots

|                                                                                                                                                                                                 |                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Mission Control HUD](docs/screenshots/screenshot-hud.png)](docs/screenshots/screenshot-hud.png) | [![Analytics dashboard](docs/screenshots/screenshot-analytics.png)](docs/screenshots/screenshot-analytics.png) |
| *Mission Control — single-screen HUD, minimap-dock arena, no tab-switching mid-hunt* | *Analytics — team comparison, top hunters, catch-value distribution, live from `/api/state`* |

*(drop your own screenshots into `docs/screenshots/` with these filenames — currently referenced but not committed)*

---

## Tech Stack

| Layer             | Technology                          | Role                                                     |
| ----------------- | ------------------------------------ | --------------------------------------------------------- |
| Robot              | Waveshare UGV Beast (Pi 4B + ESP32)  | Vision-only ground robot, no lidar                        |
| Platform           | Cyberwave SDK + edge-core            | Pairing, MQTT telemetry, camera frame retrieval           |
| Detection          | YOLOE-26 (via `ultralytics`)         | Real-time open-vocabulary object detection                |
| Re-identification  | OSNet (`torchreid`, `osnet_x0_25`)   | Appearance-based target re-matching + fine-tuning target  |
| Localization       | AprilTag (`tag36h11`, `pupil-apriltags`) | Waypoint pose reference — calibration + detection pipeline built |
| Safety             | `CollisionGuard`                     | Independent proximity-based emergency stop                |
| Backend            | FastAPI + WebSocket                  | Multi-site Hub — teams, scoring, voting, chat              |
| Frontend           | Three.js (arena map) + Chart.js (analytics) + vanilla JS | Mission Control HUD + analytics dashboard |
| Voice              | Groq Whisper API                     | Push-to-talk driving commands                              |
| ML ops             | `scikit-learn`                       | Frozen-embedding linear head for re-id fine-tuning         |
| Testing            | `pytest` + FastAPI `TestClient`      | 16 tests, simulated multi-client WebSocket flows            |

---

## Project Structure

```
quarry/
├── backend/
│   ├── main.py                    FastAPI Hub — multi-site WebSocket contract, teams, scoring
│   ├── mock_feed.py               Legacy single-robot simulated feed (dev reference)
│   ├── real_feed.py               Live telemetry + YOLOE-26/OSNet detection, temporal smoothing
│   ├── site_agent.py              Relays a Field Agent's robot into the Hub + local video overlay + data flywheel
│   ├── teleop.py                  Gamepad driving
│   ├── patrol.py                  Autonomous sweep patrol (coverage tracking, not point-to-point nav)
│   ├── voice_control.py           Push-to-talk driving via Groq Whisper
│   ├── finetune_osnet_head.py     Re-id fine-tuning — data-gated, honest before/after eval
│   ├── vo_feasibility_test.py     Monocular VO go/no-go experiment
│   ├── test_multisite.py          16 tests — Hub contract verified with simulated WebSocket clients
│   └── vision/
│       ├── detector.py            YOLOE-26 wrapper, per-label confidence floors
│       ├── reid.py                OSNet appearance matcher, multi-photo gallery
│       ├── collision_guard.py     Independent proximity emergency-stop
│       ├── apriltag_detector.py   AprilTag pose estimation — refuses to run without real calibration
│       ├── calibrate_camera.py    One-time camera intrinsics calibration
│       └── measure_waypoints.py   Venue tool — records real waypoint positions relative to fixed reference tags
├── frontend/
│   ├── index.html                 Mission Control HUD
│   ├── dashboard.html             Analytics dashboard
│   └── js/                        map3d.js (legacy, unused — held pending removal)
├── find-quarry.ps1                Auto-detects subnet, locates the Pi
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Clone & install

```
git clone https://github.com/ashish-doing/Quarry.git
cd Quarry
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

### 2. Configure

```
# .env (never committed — see .gitignore)
CYBERWAVE_API_KEY=your_key_here
CYBERWAVE_ENVIRONMENT_ID=your_environment_id
GROQ_API_KEY=your_key_here          # only needed for voice_control.py
```

### 3. Run the Hub

```
uvicorn backend.main:app --port 8001
```

Open `http://127.0.0.1:8001` — join as a Spectator, or as a Field Agent if you have your own paired robot.

### 4. Connect your robot (Field Agent only)

```
python -m backend.site_agent --hub ws://127.0.0.1:8001/ws --name "YourName" --location "City, Country" --team Alpha
```

### 5. Drive it — pick one

```
python backend\teleop.py            # gamepad
python backend\patrol.py            # autonomous sweep
python backend\voice_control.py     # push-to-talk (hold SPACE)
```

**Never run more than one of `teleop.py` / `patrol.py` / `voice_control.py` at the same time.**

### 6. Camera calibration + AprilTag pipeline (run in this order)

```
python backend\vision\calibrate_camera.py --source twin    # requires robot powered on
python -m backend.finetune_osnet_head --check-only         # -m required, not direct invocation
python backend\vo_feasibility_test.py --source webcam      # works in any room, no robot needed
```

### 7. Run the tests

```
python -m backend.test_multisite
```

```
16 passed — join, teams, player IDs, points-based scoring, follow-tracking, rate limiting
```

---

## Tests

`test_multisite.py` verifies the Hub's entire multiplayer contract with **real simulated WebSocket clients** (FastAPI `TestClient`), not mocks of the logic — join flow, team assignment, unique-name enforcement, point-based scoring off a registered target's configured value, live follow-tracking, and rate limiting under spam, all checked against the actual broadcast sequence a real client would receive.

| Check                                        | Verifies                                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Join returns `init` + unique `player_id`      | Every player gets a real identity, not just a name                                        |
| Duplicate name (case-insensitive) rejected    | Lightweight anti-spam — no two players share an identity                                  |
| `site_report` / `candidate` reach spectators  | Field Agent → Hub → everyone else, real broadcast                                         |
| Vote sequence to threshold                    | `vote_update` → `match_confirmed` → `leaderboard` → `activity`, in order                  |
| Points from registered target value           | A target worth 250 pts awards 250, not a flat +1                                          |
| Team scores aggregate correctly               | Team total = sum of that team's sites' scores                                             |
| Chat broadcasts with sender identity          | No anonymous actions anywhere                                                              |
| Rate limiting under spam                      | 8 actions / 5s per player, enforced server-side                                            |

---

## Known Limitations — What This Honestly Cannot Do Yet

- **AprilTag physical placement is unfinished.** The full calibration → detection → measurement pipeline is built and code-complete (`calibrate_camera.py`, `apriltag_detector.py`, `measure_waypoints.py`), but `calibrate_camera.py --source twin` requires the physical robot powered on and connected — attempted mid-session and correctly failed with a clean timeout (`get_frame` waiting on the physical edge device), not a code bug. `WAYPOINTS` in `real_feed.py` still holds placeholder coordinates; tags 0–5 are mapped to W1–W6, tags 6–7 reserved as fixed origin/reference markers, but no physical measurement has happened yet.
- **OSNet fine-tuning is blocked on data volume, correctly.** `finetune_osnet_head.py` is written, dry-run verified (`--check-only` correctly reports zero eligible classes against an empty `training_data/`), and refuses to train on generic-object folders that aren't real re-id signal. No real hunt session has run yet to generate the confirmed/disputed crops it needs.
- **Monocular VO is an unresolved experiment, not a feature.** `vo_feasibility_test.py` exists to answer a go/no-go question — whether ORB feature tracking holds up well enough in a likely feature-poor demo room to be worth anything as a supplementary signal. It has not been run yet. Monocular VO has an inherent scale ambiguity regardless of outcome — it was never going to be a primary navigation system, only a possible supplement to AprilTag waypoint gating.
- **Autonomous patrol is a sweep pattern, not true point-to-point navigation.** `get_pose()` returns a rotation quaternion, not a yaw angle, and that conversion was deliberately left unverified rather than guessed. `patrol.py` drives forward, checks position against waypoints, and scans when it arrives, with real coverage tracking (visited-waypoint set, full-sweep-completion detection) — real autonomous behavior, just not GPS-precise routing.
- **Live video in QUARRY's own dashboard is same-machine only.** A remote Spectator watching a different Field Agent's robot can't reach it yet — needs a real tunnel (ngrok or similar), not yet built.
- **Cross-account multi-robot sharing is architecturally supported, tested with two robots in one session** — beyond two hasn't been attempted.
- **Voice control is built, has not had a real end-to-end session with the robot moving.**
- **Battery and FPS telemetry fields are unresolved.** `twin.telemetry` is a publish-only interface, not a readable snapshot — the correct consumer-side method for these two fields hasn't been found yet.
- **Camera frame resolution note:** `get_frame('numpy', source='remote_edge')` returns frames at a different resolution than the driver's nominal stream setting — confirmed during this session. Any calibration or pose work should be checked against the actual shape returned at call time, not assumed from driver config.

Every one of these is a known, named gap — not a hidden one. This list gets shorter as each is resolved, not quietly removed.

---

## Safety Design

- Every drive path (`teleop.py`, `patrol.py`, `voice_control.py`) runs **locally, on the Field Agent's own machine**, calling the Cyberwave SDK directly — never routed through the multiplayer Hub. No Spectator, and no other Field Agent, can ever send a movement command to someone else's robot, under any UI framing.
- **Two independent safety layers**: the Pi's own driver-level movement watchdog (0.5s auto-stop if commands stop arriving, verified by killing the controlling process mid-drive) and `CollisionGuard` (proximity-based emergency stop using detection bbox size as a distance proxy — deliberately described as a coarse "something is right in front of the camera, stop now" reflex, not real depth-based obstacle avoidance, since this hardware has no lidar or depth camera).

---

## Author

**Ashish Kumar** — B.Tech ECE, IIIT Guwahati (Batch 2024–2028)

[![GitHub](https://img.shields.io/badge/GitHub-ashish--doing-181717?style=flat-square&logo=github)](https://github.com/ashish-doing) [![LinkedIn](https://img.shields.io/badge/LinkedIn-ashish--kumar-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/ashish-kumar-014aaa3b9) [![HuggingFace](https://img.shields.io/badge/HuggingFace-ashish--doing-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/ashish-doing)

---

## License

MIT — see [LICENSE](./LICENSE) for details.

---

Built for the **Cyberwave Builders Program — Cohort 2**

*FIELDWORK · ORION · THE OUTRIDER*

*Every hunt needs a team behind it.*