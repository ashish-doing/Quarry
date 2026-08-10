"""
main.py -- QUARRY Mission Control Hub, co-op recon edition.

Run with:
    uvicorn backend.main:app --reload
Open http://127.0.0.1:8000

MISSION FRAME
-------------
One INNER UGV runs the actual objective: a sequential numbered-object
survey inside a taped arena (see inner_agent.py / sequential_patrol.py).
Any number of OUTER UGVs -- other people's real hardware, dropped into
the same Cyberwave twin from their own space -- patrol their own area
as a second, cooperative task: watching for anyone who isn't part of
the mission. Inner and outer share live position with each other
through this Hub, so an outer UGV spotting an intruder can tell the
inner UGV roughly where that is relative to where the inner UGV
currently stands. Nobody drives anybody else's robot -- enforced in the
architecture, not just the UI, same as before.

Spectators watch BOTH the real camera feed and the twin environment for
whichever site they're following, and vote only on object IDENTITY: did
the inner UGV's capture (label + OCR-read marker number) actually match
what the twin claims lives at that location? Confirmed matches build a
running `confirmed_objects` map -- the answer to "where is object 2" is
just a lookup into that map, not a live query.

WHAT WAS REMOVED FROM THE EARLIER TEAM-HUNT VERSION: teams, points,
the leaderboard, and the live "+TARGET" photo-registration endpoint
(objects are registered once, offline, before a run starts, via
real_feed.state.register_target() -- see objects_registry.py). This is
a deliberate narrative and scope change, not an oversight; a stale
Site.team or Player.team field would silently reintroduce the old
framing if left in.

WebSocket message contract
---------------------------
Server -> Client:
  {"type": "init", "phase", "sites": {site_id: {...}}, "players": [...],
   "confirmed_objects": {label: {...}}, "chat_log", "activity_log",
   "alerts": [...]}
  {"type": "presence", "players": [...]}
  {"type": "join_rejected", "reason"}
  {"type": "activity", "text", "kind", "timestamp"}
  {"type": "site_update", "site_id", "telemetry"}
  {"type": "candidate", "site_id", "id", "label", "confidence",
   "instant_confidence", "waypoint", "is_registered_target", "timestamp",
   "location_offset", "ocr_number", "ocr_anomaly", "image",
   "twin_semantic_label"}
  {"type": "vote_update", "site_id", "id", "confirms", "disputes"}
  {"type": "match_confirmed", "site_id", "id", "label", "waypoint",
   "timestamp", "contributors", "location_offset", "ocr_number",
   "ocr_anomaly", "image", "twin_semantic_label"}
  {"type": "match_rejected", "site_id", "id", "label", "waypoint", "timestamp"}
  {"type": "alert", "id", "site_id", "site_owner", "site_role", "kind",
   "position", "nearest_waypoint", "distance_m", "message", "timestamp"}
  {"type": "team_ping", "site_id", "player", "waypoint", "text", "timestamp"}
  {"type": "chat", "sender", "text", "timestamp"}
  {"type": "rate_limited", "action"}
  {"type": "followers_update", "site_id", "followers"}
  {"type": "collision_alert"|"collision_clear", "site_id", "timestamp"}
  {"type": "site_removed", "site_id"}

Client -> Server:
  {"type": "join", "name", "role" ("field_agent"|"spectator"),
   "site_role" ("inner"|"outer", field_agent only), "location", "avatar"}
  {"type": "site_report", "waypoints", "telemetry"}    (field agent's own relay process only)
  {"type": "candidate_report", "label", "confidence", "instant_confidence",
   "waypoint", "is_registered_target", "location_offset", "ocr_number",
   "ocr_anomaly", "image", "twin_semantic_label"}       (field agent only)
  {"type": "alert_report", "kind", "position", "nearest_waypoint",
   "distance_m", "message"}                              (field agent only)
  {"type": "vote", "site_id", "id", "vote"}
  {"type": "ping", "site_id", "waypoint", "text"}
  {"type": "chat", "text"}
  {"type": "follow", "site_id"}   (spectators only -- which feed they're watching)
  {"type": "safety_event", "event" ("collision_alert"|"collision_clear"), "timestamp"}  (field agent only)

NOTE on `image`: a small base64 JPEG, sent once per candidate at capture
time (not every frame) -- reverses the earlier same-machine-only crop
rule specifically to support the end-of-patrol proof gallery. Still
bounded: only sent when a candidate is actually raised (gated by the
existing cooldown/smoothing logic in real_feed.py), never per-tick.
"""

import json
import re
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI(title="QUARRY Mission Control -- Hub")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

CONFIRM_THRESHOLD = 2
DISPUTE_THRESHOLD = 2   # symmetric to CONFIRM_THRESHOLD, see the vote handler
RATE_LIMIT_WINDOW_S = 5.0
RATE_LIMIT_MAX_ACTIONS = 8  # votes + pings + chat combined, per player, per window
SITE_ROLES = ("inner", "outer")
ALERT_LOG_MAXLEN = 50


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "site"


def _arena_position(site_id: str):
    """Deterministic, arbitrary (x, y) for the shared arena map -- this is
    NOT real geography. Sites have no real spatial relationship to each
    other; this just gives every site a stable, distinct spot in a shared
    visual scene so the map has consistent layout across all clients and
    across server restarts (Python's built-in hash() is randomized per
    process for strings, so it's not usable here -- md5 is)."""
    import hashlib
    h = int(hashlib.md5(site_id.encode()).hexdigest(), 16)
    return {"x": (h % 71) - 35, "y": ((h // 71) % 71) - 35}


class Candidate:
    def __init__(self, label, confidence, waypoint, is_registered_target, site_id,
                 instant_confidence=None, location_offset=None, ocr_number=None,
                 ocr_anomaly=None, image=None, twin_semantic_label=None,
                 detected_shape=None, detected_color=None):
        self.id = str(uuid.uuid4())[:8]
        self.site_id = site_id
        self.label = label
        self.confidence = confidence
        self.instant_confidence = confidence if instant_confidence is None else instant_confidence
        self.waypoint = waypoint
        self.is_registered_target = is_registered_target
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.votes: Dict[str, str] = {}
        self.status = "pending"
        self.location_offset = location_offset
        self.ocr_number = ocr_number
        self.ocr_anomaly = ocr_anomaly
        self.image = image                          # base64 JPEG or None
        self.twin_semantic_label = twin_semantic_label  # what the twin claims is here
        self.detected_shape = detected_shape          # Task 2: coarse shape heuristic, or None
        self.detected_color = detected_color           # Task 2: blue/pink heuristic, or None

    def net_confirms(self):
        c = sum(1 for v in self.votes.values() if v == "confirm")
        d = sum(1 for v in self.votes.values() if v == "dispute")
        return c - d

    def vote_tally(self):
        c = sum(1 for v in self.votes.values() if v == "confirm")
        d = sum(1 for v in self.votes.values() if v == "dispute")
        return c, d

    def to_dict(self):
        return {
            "type": "candidate", "site_id": self.site_id, "id": self.id, "label": self.label,
            "confidence": self.confidence, "instant_confidence": self.instant_confidence,
            "waypoint": self.waypoint,
            "is_registered_target": self.is_registered_target, "timestamp": self.timestamp,
            "location_offset": self.location_offset,
            "ocr_number": self.ocr_number,
            "ocr_anomaly": self.ocr_anomaly,
            "image": self.image,
            "twin_semantic_label": self.twin_semantic_label,
            "detected_shape": self.detected_shape,
            "detected_color": self.detected_color,
        }


class Site:
    def __init__(self, site_id: str, owner: str, location: str, role: str, waypoints: list):
        self.site_id = site_id
        self.owner = owner
        self.location = location
        self.role = role  # "inner" | "outer"
        self.waypoints = waypoints
        self.telemetry = {}
        self.candidates: Dict[str, Candidate] = {}
        self.followers: set = set()

    def to_dict(self):
        return {
            "site_id": self.site_id, "owner": self.owner, "location": self.location,
            "role": self.role, "waypoints": self.waypoints, "telemetry": self.telemetry,
            "candidates": [c.to_dict() for c in self.candidates.values()],
            "followers": list(self.followers),
            "arena_position": _arena_position(self.site_id),
        }


class Player:
    def __init__(self, ws: WebSocket, name: str, role: str, location: Optional[str], avatar: str):
        self.ws = ws
        self.player_id = str(uuid.uuid4())[:8]
        self.name = name
        self.role = role  # "field_agent" | "spectator"
        self.location = location
        self.avatar = avatar
        self.site_id: Optional[str] = None
        self.following: Optional[str] = None
        self.connected = True
        self._action_times: deque = deque()

    def rate_limited(self) -> bool:
        now = time.monotonic()
        while self._action_times and now - self._action_times[0] > RATE_LIMIT_WINDOW_S:
            self._action_times.popleft()
        if len(self._action_times) >= RATE_LIMIT_MAX_ACTIONS:
            return True
        self._action_times.append(now)
        return False

    def to_dict(self):
        return {
            "player_id": self.player_id, "name": self.name, "role": self.role,
            "location": self.location, "avatar": self.avatar, "connected": self.connected,
            "following": self.following,
        }


class Hub:
    def __init__(self):
        self.players: Dict[WebSocket, Player] = {}
        self.sites: Dict[str, Site] = {}
        self.chat_log: deque = deque(maxlen=200)
        self.activity_log: deque = deque(maxlen=200)
        self.alerts: deque = deque(maxlen=ALERT_LOG_MAXLEN)
        # The actual "where is object N" answer -- keyed by the object's
        # registered label (e.g. "object-02"), populated as each object
        # gets confirmed. This IS the end-of-patrol proof gallery's data
        # source; no separate storage needed.
        self.confirmed_objects: Dict[str, dict] = {}

    def taken_names(self):
        return {p.name.lower() for p in self.players.values()}

    def inner_site(self) -> Optional[Site]:
        """Returns the first site with role 'inner', or None if no inner
        UGV is currently connected. Used to fold the inner UGV's current
        position into an outer UGV's intruder alert message. If somehow
        more than one site claims 'inner' (shouldn't happen in normal
        use -- the mission has one real inner UGV), the first one found
        wins; this isn't validated/rejected at join time because a
        misconfigured second inner agent is a demo-setup mistake, not a
        security boundary."""
        for site in self.sites.values():
            if site.role == "inner":
                return site
        return None

    def snapshot(self):
        return {
            "sites": {sid: s.to_dict() for sid, s in self.sites.items()},
            "players": [p.to_dict() for p in self.players.values()],
            "chat_log": list(self.chat_log),
            "activity_log": list(self.activity_log),
            "alerts": list(self.alerts),
            "confirmed_objects": self.confirmed_objects,
        }


hub = Hub()


async def broadcast(message: dict):
    dead = []
    for ws in list(hub.players.keys()):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        hub.players.pop(ws, None)


async def broadcast_presence():
    await broadcast({"type": "presence", "players": [p.to_dict() for p in hub.players.values()]})


async def log_activity(text: str, kind: str = "info"):
    entry = {"type": "activity", "text": text, "kind": kind,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    hub.activity_log.append(entry)
    await broadcast(entry)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    player: Optional[Player] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # -- join must happen before anything else -------------------
            if msg_type == "join":
                if player is not None:
                    continue
                name = (msg.get("name") or "").strip()[:24]
                role = msg.get("role") if msg.get("role") in ("field_agent", "spectator") else "spectator"
                location = (msg.get("location") or "").strip()[:40] if role == "field_agent" else None
                avatar = (msg.get("avatar") or "\u25c8").strip()[:4]
                site_role = msg.get("site_role") if msg.get("site_role") in SITE_ROLES else None

                if not name:
                    await websocket.send_text(json.dumps({"type": "join_rejected", "reason": "Name required."}))
                    continue
                if name.lower() in hub.taken_names():
                    await websocket.send_text(json.dumps({
                        "type": "join_rejected", "reason": f"'{name}' is already taken this session."
                    }))
                    continue
                if role == "field_agent" and not location:
                    await websocket.send_text(json.dumps({
                        "type": "join_rejected", "reason": "Field Agents must set a location label."
                    }))
                    continue
                if role == "field_agent" and site_role is None:
                    await websocket.send_text(json.dumps({
                        "type": "join_rejected",
                        "reason": "Field Agents must choose Inner or Outer.",
                    }))
                    continue

                player = Player(websocket, name, role, location, avatar)
                hub.players[websocket] = player

                if role == "field_agent":
                    site_id = _slugify(name)
                    hub.sites[site_id] = Site(site_id, owner=name, location=location, role=site_role, waypoints=[])
                    player.site_id = site_id

                await websocket.send_text(json.dumps({"type": "init", **hub.snapshot(), "your_player_id": player.player_id}))
                await broadcast_presence()
                role_label = f"{site_role.title()} Field Agent ({location})" if role == "field_agent" else "Spectator"
                await log_activity(f"{name} joined as {role_label}.", "join")
                continue

            if player is None:
                continue

            # -- field-agent-only: reporting their own site's live data ---
            if msg_type == "site_report" and player.role == "field_agent" and player.site_id:
                site = hub.sites[player.site_id]
                if "waypoints" in msg:
                    site.waypoints = msg["waypoints"]
                site.telemetry = msg.get("telemetry", {})
                await broadcast({"type": "site_update", "site_id": site.site_id, "telemetry": site.telemetry})
                continue

            if msg_type == "candidate_report" and player.role == "field_agent" and player.site_id:
                candidate = Candidate(
                    msg.get("label"), msg.get("confidence"), msg.get("waypoint"),
                    msg.get("is_registered_target", False), site_id=player.site_id,
                    instant_confidence=msg.get("instant_confidence"),
                    location_offset=msg.get("location_offset"),
                    ocr_number=msg.get("ocr_number"),
                    ocr_anomaly=msg.get("ocr_anomaly"),
                    image=msg.get("image"),
                    twin_semantic_label=msg.get("twin_semantic_label"),
                    detected_shape=msg.get("detected_shape"),
                    detected_color=msg.get("detected_color"),
                )
                hub.sites[player.site_id].candidates[candidate.id] = candidate
                await broadcast(candidate.to_dict())
                continue

            if msg_type == "alert_report" and player.role == "field_agent" and player.site_id:
                site = hub.sites[player.site_id]
                inner = hub.inner_site()
                inner_note = (
                    f" Inner UGV last at {inner.telemetry.get('position')}."
                    if inner and inner.site_id != site.site_id and inner.telemetry.get("position")
                    else ""
                )
                alert = {
                    "type": "alert",
                    "id": str(uuid.uuid4())[:8],
                    "site_id": site.site_id,
                    "site_owner": site.owner,
                    "site_role": site.role,
                    "kind": msg.get("kind", "intruder"),
                    "position": msg.get("position"),
                    "nearest_waypoint": msg.get("nearest_waypoint"),
                    "distance_m": msg.get("distance_m"),
                    "message": (msg.get("message") or "").strip()[:200] + inner_note,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                hub.alerts.appendleft(alert)
                await broadcast(alert)
                await log_activity(
                    f"\u26a0 ALERT -- {site.owner} ({site.role}) reported {alert['kind']} "
                    f"near {alert['nearest_waypoint']}", "danger",
                )
                continue

            if msg_type == "safety_event" and player.role == "field_agent" and player.site_id:
                event = msg.get("event")
                if event not in ("collision_alert", "collision_clear"):
                    continue
                await broadcast({
                    "type": event, "site_id": player.site_id,
                    "timestamp": msg.get("timestamp", time.time()),
                })
                continue

            # -- rate-limited actions available to everyone ---------------
            if msg_type in ("vote", "ping", "chat") and player.rate_limited():
                await websocket.send_text(json.dumps({"type": "rate_limited", "action": msg_type}))
                continue

            if msg_type == "chat":
                text = (msg.get("text") or "").strip()[:280]
                if not text:
                    continue
                entry = {"type": "chat", "sender": player.name, "text": text,
                          "timestamp": datetime.now(timezone.utc).isoformat()}
                hub.chat_log.append(entry)
                await broadcast(entry)
                continue

            if msg_type == "ping":
                site_id = msg.get("site_id")
                if site_id not in hub.sites:
                    continue
                await broadcast({
                    "type": "team_ping", "site_id": site_id, "player": player.name,
                    "waypoint": msg.get("waypoint"), "text": msg.get("text", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            if msg_type == "follow":
                site_id = msg.get("site_id")
                if site_id not in hub.sites:
                    continue
                if player.following and player.following in hub.sites:
                    hub.sites[player.following].followers.discard(player.name)
                    await broadcast({"type": "followers_update", "site_id": player.following,
                                       "followers": list(hub.sites[player.following].followers)})
                player.following = site_id
                hub.sites[site_id].followers.add(player.name)
                await broadcast({"type": "followers_update", "site_id": site_id,
                                   "followers": list(hub.sites[site_id].followers)})
                await log_activity(f"{player.name} is now watching {hub.sites[site_id].owner}'s feed.", "info")
                continue

            if msg_type == "vote":
                site_id = msg.get("site_id")
                site = hub.sites.get(site_id)
                if not site:
                    continue
                candidate = site.candidates.get(msg.get("id"))
                if not candidate or candidate.status != "pending":
                    continue
                candidate.votes[player.name] = msg.get("vote")
                confirms, disputes = candidate.vote_tally()
                await broadcast({
                    "type": "vote_update", "site_id": site_id, "id": candidate.id,
                    "confirms": confirms, "disputes": disputes,
                })
                if candidate.net_confirms() >= CONFIRM_THRESHOLD:
                    candidate.status = "confirmed"
                    contributors = [p for p, v in candidate.votes.items() if v == "confirm"]
                    entry = {
                        "site_id": site_id, "id": candidate.id, "label": candidate.label,
                        "waypoint": candidate.waypoint, "timestamp": candidate.timestamp,
                        "contributors": contributors,
                        "location_offset": candidate.location_offset,
                        "ocr_number": candidate.ocr_number,
                        "ocr_anomaly": candidate.ocr_anomaly,
                        "image": candidate.image,
                        "twin_semantic_label": candidate.twin_semantic_label,
                        "detected_shape": candidate.detected_shape,
                        "detected_color": candidate.detected_color,
                    }
                    # The actual "where is object N" answer, keyed by label --
                    # overwritten on re-confirmation if a target is somehow
                    # confirmed twice, which is fine, the latest is the truth.
                    hub.confirmed_objects[candidate.label] = entry
                    del site.candidates[candidate.id]
                    await broadcast({"type": "match_confirmed", **entry})
                    anomaly_note = f" [OCR ANOMALY: {candidate.ocr_anomaly}]" if candidate.ocr_anomaly else ""
                    await log_activity(
                        f'CONFIRMED -- "{candidate.label}" @ {candidate.waypoint} '
                        f"({site.owner}'s site) by {', '.join(contributors)}{anomaly_note}",
                        "match",
                    )
                elif candidate.net_confirms() <= -DISPUTE_THRESHOLD:
                    candidate.status = "rejected"
                    del site.candidates[candidate.id]
                    await broadcast({
                        "type": "match_rejected", "site_id": site_id, "id": candidate.id,
                        "label": candidate.label, "waypoint": candidate.waypoint,
                        "timestamp": candidate.timestamp,
                    })
                    await log_activity(
                        f'Sighting disputed away -- "{candidate.label}" @ {candidate.waypoint} '
                        f"({site.owner}'s site)",
                        "info",
                    )
                continue

    except WebSocketDisconnect:
        pass
    finally:
        if player is not None:
            hub.players.pop(websocket, None)
            player.connected = False
            if player.role == "field_agent" and player.site_id in hub.sites:
                del hub.sites[player.site_id]
                await broadcast({"type": "site_removed", "site_id": player.site_id})
            await broadcast_presence()
            await log_activity(f"{player.name} disconnected.", "leave")


@app.get("/api/state")
async def get_state():
    return hub.snapshot()


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")