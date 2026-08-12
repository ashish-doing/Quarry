<div align="center">

<img src="branding/quarry-logo.png" alt="QUARRY" width="110" />

</div>

<div align="center">
<img src="branding/quarry-banner.png" alt="QUARRY — Fieldwork" width="640" />
</div>

<br/>

# Building QUARRY: a real robot doing human-in-the-loop verification, not just patrol

*A Cyberwave Builders Program (Cohort 2) build log — what QUARRY actually does, how it's built, and what's still honestly unfinished.*

**Team Fieldwork · Robot: Orion, the Outrider · Builder: Ashish Kumar (Solo)**

---

## The problem underneath the robot

Most autonomous-vision demos stop at "the model detected X." QUARRY starts from a different question: **once a vision model says it found something, who actually checks if it's right — and how do you build that checking into the system instead of bolting it on after?**

That's not a robotics problem first — it's a labeling problem. It's the same shape as citizen-science wildlife camera-trap projects (a model flags "possible leopard," a human confirms), or content-moderation triage (a classifier flags, a reviewer decides). QUARRY puts a real robot underneath that pattern instead of a dataset: one UGV runs an autonomous survey, and every single detection it makes goes to a human vote before it counts as confirmed. Nothing the robot sees becomes "true" without a person agreeing.

We built this on the Waveshare UGV Beast through the Cyberwave SDK, because the interesting part isn't "can a robot detect an object" — that's solved. The interesting part is **what happens after detection**, and whether you can make that verification loop tight enough to actually generate usable, human-labeled training data as a side effect of just running the mission.

## The mission, in plain terms

![Taped arena with all 8 numbered objects laid out, Orion visible in frame](screenshots/screenshot-arena.jpg)
*The actual taped arena — 8 numbered objects laid out, Orion (bottom-right) ready to run the sequence.*

One real **inner UGV** — vision-only, no lidar — drives to 8 numbered objects (cylinders, cones, boxes, in blue/pink marker paper) inside a taped arena, in strict sequential order: object 1, then 2, then 3, and so on. At each one, it stops, waits for a properly-framed shot, and runs three independent identity checks before reporting anything.

Any number of **outer UGVs** — other people's real hardware, in their own physical spaces, connected to the same Cyberwave twin — patrol as overwatch. They don't run the numbered sequence at all. Their job is narrower and faster: share live position with the inner UGV through a shared Hub, and the moment their own camera spots an unregistered person, fire an immediate alert — position, distance to the nearest known waypoint, a short message. No voting gate on alerts; when something might be an intruder, speed matters more than consensus.

**Spectators** watch the inner UGV's real camera feed and the Cyberwave twin environment side by side, and vote on exactly one question per sighting: does what the robot actually captured (label, plus an independently OCR-read marker number) match what the twin claims lives at that location? That's the whole verification loop — a real camera frame, a real independent check, a real human decision, every time.

*Full architecture — WebSocket contract, sequence diagrams, coordinate frames — lives in [`ARCHITECTURE.md`](../ARCHITECTURE.md); this write-up covers the parts worth walking through in prose.*

## Navigation: measuring instead of guessing

Cyberwave's `get_pose()` returns position cleanly, but its rotation comes back as a quaternion — and we made an early, deliberate call not to convert that into a yaw angle anywhere in this project. A wrong yaw conversion means the robot turns confidently in the wrong direction, which is a worse failure than just admitting you don't have heading yet.

So navigation is **motion-derived**: the controller measures position before and after each drive burst, and takes the direction of that measured displacement as its current heading estimate. First tick only, it takes one small blind forward nudge purely to get a second position reading — after that, every heading update comes from real measured movement, never a guess. It's closed-loop at every step, at the cost of not being a full path-planning stack. That trade felt right for an 8-waypoint taped arena; it might not for a more complex space, and that's a fair thing to flag rather than pretend the current approach generalizes.

Waypoint measurement itself uses the same philosophy. The original plan reserved two AprilTags as fixed reference markers and computed everything else relative to them — but with 8 real objects needing 8 real waypoints, there was no tag budget left for dedicated anchors. So we switched to **drive-and-center**: physically drive the robot to sit centered on each tag (the same act it'll do autonomously later), then record `get_pose()`'s (x, y) directly at that moment. The AprilTag detection shown on screen during this process is a confirmation aid only — "yes, I'm centered on tag 3" — never a measurement input. This sidesteps the yaw problem entirely: drive-and-center only ever needs position, never rotation.

![Camera calibration checkerboard capture, in progress](screenshots/screenshot-calibration.png)
*`calibrate_camera.py` mid-run — checkerboard capture for real camera intrinsics, a prerequisite for trustworthy AprilTag pose estimates. This step has to happen before any waypoint measurement is trusted.*

## Object identity: three independent signals, one primary

**OSNet appearance re-identification** is the primary signal. Each object gets registered offline, before a mission starts, from 3–5 real photos at different angles — matching takes the *best* similarity across a target's whole photo gallery, not an average, since averaging would blur out exactly the appearance variation multi-angle registration is meant to capture. The gallery persists to disk (`target_gallery.pkl`) specifically because registration and the live mission run in separate processes; without that persistence step, embeddings built during registration would never reach the process actually driving the robot.

**OCR cross-check** runs on top of that, not instead of it. Every registered object has a printed marker number; Tesseract reads it off the capture crop and compares it against what OSNet's match implies. The one rule we were strict about here: **no OCR signal is never treated as an anomaly.** A bad-angle frame that yields nothing readable isn't evidence anything's wrong — only a genuine number *mismatch* gets flagged. That distinction has its own dedicated test coverage, because it's the kind of thing that's easy to get subtly backwards.

**Shape/color detection** is the third, lightest signal — a coarse heuristic reading cone/cylinder/box shape and blue/pink color off the same crop. *Flagging this honestly: this piece is still in active development in a parallel work stream as of this write-up, so treat the shape/color fields as provisional until that's confirmed working end-to-end against real captures.*

None of the three silently override each other or the human vote. Spectators see all available signals and still decide — this is the actual point of a human-in-the-loop system, not a bug to be optimized away.

## The digital twin side of the mission

![Cyberwave QUARRY_DEMO twin scene — CoreZone, HuntZone, decoy objects, outer UGV slots](screenshots/screenshot-twin-scene.png)
*The `QUARRY_DEMO` scene in Cyberwave's workbench — CoreZone/HuntZone ground planes, decoy props (storage cabinet, task lamp, ladder, utility cart, equipment rack, work table, pallet stack, robot charging dock), a measured waypoint, and — notably — three unclaimed outer-UGV slots already modeled for the overwatch role.*

This is the piece that makes the spectator vote meaningful: the twin isn't just a pretty visualization sitting next to the real feed, it's the other half of the identity check. When Orion reports "cone, object 02," the vote isn't "does this look like a cone" — it's "does this match what the twin says lives at that exact location." The `OuterSlot1/2/3_Unclaimed` entries in the scene are the twin-side placeholders for outer UGVs before anyone claims an overwatch role — visible proof the multi-site architecture isn't just a diagram, it's modeled in the actual environment.

## The intruder-alert idea

This is the part of the system we designed and built end-to-end, but didn't stage for the demo recording — worth being upfront about that rather than filming something that wasn't really happening.

The concept: outer UGVs are running in completely separate physical spaces, connected to the same Hub. If any of them spots an unregistered person, it fires an alert immediately — no waypoint gating, only a 15-second cooldown, because for a possible-intruder signal, speed matters more than deduplication. That alert broadcasts to everyone connected, and if an inner UGV is currently active, the Hub folds its last known position into the alert message, so an outer site's operator has a rough sense of where the inner mission stands relative to what they just saw. It's deliberately *not* cross-checked against registered mission objects — those are physical props (cylinders, cones, boxes), not people, so there's no "known person" exclusion to build.

For this submission's recorded demo, only the inner UGV is physically running — we didn't have a second robot/operator available for the recording session. The intruder-alert path is real, built, and part of the architecture (see the outer slots in the twin scene above); it's just not part of the filmed clip. Worth being direct about that distinction rather than implying a live multi-site demo that isn't happening.

## Safety — two layers, independent of each other

Every drive path — teleop, voice control, or the inner agent's own driving loop — runs locally on that operator's own machine, calling the Cyberwave SDK directly, never routed through the shared Hub. No spectator and no other field agent can ever send a movement command to someone else's robot; that's structural, not a UI convention.

On top of that, two safety layers run independently, with no shared state between them:
1. The Pi's own driver-level movement watchdog — auto-stops within 0.5s of losing its controlling process. Verified by actually killing the control process mid-drive and confirming the stop.
2. `CollisionGuard` — a proximity emergency-stop using detection bbox size as a distance proxy, since this hardware has no lidar or depth camera. This is explicitly *not* depth-based obstacle avoidance; it's a coarse "something is right in front of the camera, stop now" reflex, and we've been careful not to demo it as anything more than that.

## Test coverage — and what isn't covered

Nav geometry (heading math, waypoint offset) has 13 unit tests, all pure Python with zero heavy dependencies — no camera, no GPU, no Cyberwave credentials needed to run them. Object registry loading and OCR decision logic have 9 tests, including the specific "no signal ≠ anomaly" case called out above. The multi-site Hub contract — join flow, voting thresholds, confirmed-objects map, dispute resolution, rate limiting — has real WebSocket-level integration tests using FastAPI's test client, not mocked-out assertions.

**Honest gap:** `real_feed.py`'s live candidate-raising path — the code that actually runs during a mission — isn't unit-tested directly, because that module loads the CV stack (YOLOE-26, OSNet) at import time, and importing it in a test process risks failing on any machine that doesn't have that stack configured. The underlying math it depends on is tested via the pure-Python `nav_math.py` module; the integration itself is only exercised by running real sessions, not a unit test. That's a real, acknowledged limitation, not an oversight we're hoping nobody notices.

## What's honestly still unfinished

In the spirit of not overclaiming anything:

- **Waypoint positions are still placeholder** until `measure_waypoints.py` gets run at the actual demo venue — the calibration step above has to complete first.
- **`twin_semantic_label` is `null`** for all 8 objects — the Cyberwave twin's actual object placement hasn't been reconciled with our object registry yet, so the identity-vs-twin comparison spectators are meant to vote on will show "unknown" until that's done.
- **The dual real-feed + twin-view spectator panel isn't built yet** — genuinely blocked on confirming what the Cyberwave twin viewer renders and how to embed it, not a forgotten task.
- **OSNet fine-tuning is correctly gated, not yet run** — the pipeline refuses to fine-tune below a minimum data volume (8 confirmed samples per class, 2+ classes), and no real mission session has generated that volume yet.
- **Remote video is same-machine-only** — a spectator watching a field agent's feed from elsewhere can't reach the annotated stream yet; that needs a real tunnel (ngrok or similar), not yet built.
- **Shape/color detection** — as noted above, still being finished in a parallel work stream as of this write-up.

Every one of these is tracked as a known, named gap in the repo's own [`README.md`](../README.md) and [`FMEA.md`](../FMEA.md) — not something surfaced for the first time in this post.

## Tech stack

Waveshare UGV Beast (Pi 4B + ESP32, vision-only) → Cyberwave SDK + edge-core for pairing, MQTT telemetry, and the shared digital twin → YOLOE-26 (`ultralytics`) for open-vocabulary detection → OSNet (`torchreid`) for appearance re-identification → Tesseract (`pytesseract`) for the OCR cross-check → AprilTag (`pupil-apriltags`) for one-time waypoint measurement → FastAPI + WebSocket for the multi-site Hub → `scikit-learn` for the re-id fine-tuning head → `pytest` throughout.

Repo: [github.com/ashish-doing/Quarry](https://github.com/ashish-doing/Quarry)

---

<div align="center">
<img src="branding/quarry-logo.png" alt="QUARRY seal" width="60" />

<br/>

*Built for the Cyberwave Builders Program, Cohort 2. Feedback and questions welcome — this is a working system with clearly labeled edges, not a finished product pretending otherwise.*

*Team Fieldwork · Robot: Orion, the Outrider*

</div>