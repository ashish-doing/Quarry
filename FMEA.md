# QUARRY — Failure Mode and Effects Analysis (FMEA)

Severity (S): 1 (negligible) – 5 (safety/data-loss critical)
Likelihood (L): 1 (rare) – 5 (frequent, seen in testing)
Risk = S × L. Mitigation column states what's actually built, not aspirational.

| # | Component | Failure Mode | Effect | S | L | Risk | Mitigation (built) |
|---|---|---|---|---|---|---|---|
| 1 | Robot control | Control process crashes or loses connection mid-drive | Robot keeps moving uncommanded | 5 | 2 | 10 | Pi's own 0.5s movement watchdog — verified by killing the control process mid-drive |
| 2 | Vision | Object very close to camera (potential collision) | Robot drives into obstacle | 5 | 3 | 15 | `CollisionGuard` — independent async loop, bbox-area proxy, 0.15s check interval, 30% frame threshold |
| 3 | Vision | `CollisionGuard`'s own frame read fails | Guard silently stops checking | 5 | 2 | 10 | `try/except` around frame read; loop continues on next tick rather than crashing — **known gap: a sustained read failure gives no operator alert**, logged at debug level only |
| 4 | Vision | False-positive "person" detection (clothes on furniture, etc.) | Wrong candidate raised, wastes voter attention | 2 | 4 | 8 | Dedicated `"clothing"` label competes against `"person"`; per-label confidence floor (0.65 vs 0.45 default) |
| 5 | Vision | Stationary object re-raises as new candidate every cooldown window | Sightings panel floods (confirmed live: 100+ in 2-3 min at old 4s cooldown) | 2 | 5 | 10 | Cooldown raised to 25s + "don't raise if one's already pending" check |
| 6 | Vision | Single-frame flicker misdetection | Candidate raised on a fluke | 2 | 3 | 6 | Temporal smoothing — label must appear in ≥3 of last 5 ticks before being trusted |
| 7 | Re-identification | OSNet backend fails to load (no CUDA, missing weights) | Targets never auto-match by appearance | 3 | 2 | 6 | Degrades to `available = False`; falls back to manual team voting only, does not crash the feed |
| 8 | Multiplayer | Two players claim the same name | Identity collision, unclear attribution | 2 | 2 | 4 | Case-insensitive uniqueness check at join, rejected with clear reason |
| 9 | Multiplayer | A player spams votes/pings/chat | Server load, panel spam for others | 2 | 3 | 6 | Rate limiter — 8 actions / 5s per connection, server-enforced |
| 10 | Multiplayer | A disputed candidate never resolves | Sightings panel accumulates permanently-pending items | 3 | 3 | 9 | Symmetric dispute-rejection threshold added — `net_confirms <= -CONFIRM_THRESHOLD` triggers `match_rejected`, same as the confirm path |
| 11 | Data flywheel | Crop save fails (bad image, disk full, permissions) | Training data silently incomplete | 2 | 2 | 4 | `try/except` around `cv2.imwrite`; failure printed, does not crash the relay loop |
| 12 | Data flywheel | Generic-object folders mistaken for re-id signal | Fine-tune trains on the wrong thing, silently wrong "improvement" number | 4 | 2 | 8 | `finetune_osnet_head.py` cross-checks folder names against real registered targets, skips and logs anything else |
| 13 | Data flywheel | Fine-tune run against too little data | Overfit, meaningless accuracy delta reported with false confidence | 3 | 3 | 9 | Hard gate — refuses below `MIN_SAMPLES_PER_CLASS` (8) or `MIN_CLASSES` (2); report explicitly flags small `n_eval` |
| 14 | AprilTag pipeline | Pose estimated without real camera calibration | Plausible-looking but wrong distance/position | 4 | 2 | 8 | `apriltag_detector.py` refuses to run without `camera_intrinsics.json` present |
| 15 | AprilTag pipeline | Tag decode fails at distance/angle | Waypoint measurement missed at venue | 2 | 3 | 6 | Range constraint calculated and documented (~0.5–1.0m for 100mm tags at this camera's FOV/resolution) — placement planned around it, not discovered live |
| 16 | Digital twin sync | `set_pose()` write fails | Cyberwave platform's own twin view goes stale | 1 | 2 | 2 | `try/except`, logged at debug, does not break local telemetry |
| 17 | Multi-site | A Field Agent disconnects mid-match | Their site's candidates/telemetry vanish for spectators | 2 | 2 | 4 | `site_removed` broadcast on disconnect, cleanly removes the site rather than leaving stale data |
| 18 | Voice control | Short-word misrecognition ("back" → "thank you") | Wrong command executed | 3 | 3 | 9 | Whisper prompt-biasing toward known command vocabulary — **honestly stated to reduce, not eliminate**, this risk |
| 19 | Navigation | Directed point-to-point steering | Wrong-direction drive from a bad yaw guess | 4 | — | — | **Deliberately not implemented** — quaternion→yaw conversion left unverified rather than guessed; `patrol.py` uses coverage-tracking sweep instead |
| 20 | Video/remote | Remote spectator can't reach a same-machine-only annotated feed | Feature works locally, appears broken remotely | 2 | 5 | 10 | **Known, undocumented-as-fixed gap** — needs ngrok/tunnel, not yet built; UI should communicate this rather than silently fail |

## Highest-risk open items (Risk ≥ 9, not yet fully mitigated)

- **#2 (Risk 15)** — CollisionGuard is a coarse bbox-area proxy, not true depth sensing. Mitigated as well as vision-only hardware allows, but this is a hard hardware ceiling, not a software gap that more engineering closes.
- **#3 (Risk 10)** — no operator-facing alert if CollisionGuard's own frame reads start failing silently. Real gap — a monitoring/heartbeat check would close this.
- **#18 (Risk 9)** — voice misrecognition is honestly bounded, not solved. Acceptable given voice control is explicitly lower-priority than the CV/safety core.
- **#10, #13 (Risk 9 each)** — both have real mitigations already built (symmetric dispute resolution, data-volume gating); listed here because the underlying failure mode is still possible, just now handled rather than silent.