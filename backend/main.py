"""
main.py -- QUARRY Mission Control, multi-site Hub edition.

Run with:
    uvicorn backend.main:app --reload
Open http://127.0.0.1:8000 -- everyone who connects joins the SAME shared
session, either as a Field Agent (has their own robot, runs their own
backend.site_agent.py locally to report in) or a Spectator (no hardware,
just watches/votes/chats).

ARCHITECTURE (see rebuild brief Section 2 -- this is not PUBG):
There is no shared physical world. Each Field Agent's robot lives in
their own room, driven only by them -- this Hub only aggregates
telemetry, candidate sightings, votes, and chat across sites. It never
becomes a control channel for anyone's hardware. A Field Agent's own
site_agent.py (running alongside their own real_feed.py) is the only
thing that can report new telemetry/candidates for their site; nobody
else's WebSocket message can touch another player's site data.

Lightweight anti-spam "profile": a name is claimed uniquely per session
(case-insensitive, no two players share one), and votes/pings/chat are
rate-limited per connection. This is identity-for-accountability, not
real auth -- there's no password, no persistence across sessions. If
you need real accounts later, that's a separate, bigger piece.

WebSocket message contract
---------------------------
Server -> Client:
  {"type": "init", "phase", "sites": {site_id: {...}}, "players": [...],
   "leaderboard", "team_scores", "chat_log", "activity_log", "targets"}
  {"type": "presence", "players": [...]}
  {"type": "join_rejected", "reason"}
  {"type": "activity", "text", "kind", "timestamp"}
  {"type": "site_update", "site_id", "telemetry"}
  {"type": "candidate", "site_id", "id", "label", "confidence",
   "waypoint", "is_registered_target", "timestamp"}
  {"type": "vote_update", "site_id", "id", "confirms", "disputes"}
  {"type": "match_confirmed", "site_id", "id", "label", "confidence",
   "waypoint", "timestamp", "contributors", "points"}
  {"type": "leaderboard", "entries", "team_scores"}
  {"type": "team_ping", "site_id", "player", "waypoint", "text", "timestamp"}
  {"type": "chat", "sender", "text", "timestamp"}
  {"type": "rate_limited", "action"}
  {"type": "followers_update", "site_id", "followers"}
  {"type": "targets_update", "targets"}
  {"type": "collision_alert"|"collision_clear", "site_id", "timestamp"}
  {"type": "site_removed", "site_id"}

Client -> Server:
  {"type": "join", "name", "role" ("field_agent"|"spectator"), "team"
   ("Alpha"|"Bravo"), "location" (field_agent only), "avatar"}
  {"type": "site_report", "waypoints", "telemetry"}         (field agent's own site_agent.py only)
  {"type": "candidate_report", "label", "confidence", "waypoint", "is_registered_target"}  (field agent only)
  {"type": "vote", "site_id", "id", "vote"}
  {"type": "ping", "site_id", "waypoint", "text"}
  {"type": "chat", "text"}
  {"type": "follow", "site_id"}   (spectators only -- which feed they're watching)
  {"type": "safety_event", "event" ("collision_alert"|"collision_clear"), "timestamp"}  (field agent only)
"""

import json
import re
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="QUARRY Mission Control -- Hub")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

CONFIRM_THRESHOLD = 2
RATE_LIMIT_WINDOW_S = 5.0
RATE_LIMIT_MAX_ACTIONS = 8  # votes + pings + chat combined, per player, per window
TEAMS = ["Alpha", "Bravo"]
DEFAULT_TARGET_POINTS = 100
GENERIC_DETECTION_POINTS = 10  # a sighting that isn't a registered target


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
    def __init__(self, label, confidence, waypoint, is_registered_target, site_id):
        self.id = str(uuid.uuid4())[:8]
        self.site_id = site_id
        self.label = label
        self.confidence = confidence
        self.waypoint = waypoint
        self.is_registered_target = is_registered_target
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.votes: Dict[str, str] = {}
        self.status = "pending"

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
            "confidence": self.confidence, "waypoint": self.waypoint,
            "is_registered_target": self.is_registered_target, "timestamp": self.timestamp,
        }


class Site:
    def __init__(self, site_id: str, owner: str, location: str, team: str, waypoints: list):
        self.site_id = site_id
        self.owner = owner
        self.location = location
        self.team = team
        self.waypoints = waypoints
        self.telemetry = {}
        self.candidates: Dict[str, Candidate] = {}
        self.followers: set = set()  # player names currently watching this feed
        self.score = 0

    def to_dict(self):
        return {
            "site_id": self.site_id, "owner": self.owner, "location": self.location,
            "team": self.team, "waypoints": self.waypoints, "telemetry": self.telemetry,
            "candidates": [c.to_dict() for c in self.candidates.values()],
            "followers": list(self.followers), "score": self.score,
            "arena_position": _arena_position(self.site_id),
        }


class Player:
    def __init__(self, ws: WebSocket, name: str, role: str, team: str, location: Optional[str], avatar: str):
        self.ws = ws
        self.player_id = str(uuid.uuid4())[:8]
        self.name = name
        self.role = role  # "field_agent" | "spectator"
        self.team = team  # "Alpha" | "Bravo"
        self.location = location
        self.avatar = avatar
        self.site_id: Optional[str] = None  # set once their Site is created, field agents only
        self.following: Optional[str] = None  # site_id a spectator is currently watching
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
            "player_id": self.player_id, "name": self.name, "role": self.role, "team": self.team,
            "location": self.location, "avatar": self.avatar, "connected": self.connected,
            "following": self.following,
        }


class Hub:
    def __init__(self):
        self.players: Dict[WebSocket, Player] = {}
        self.sites: Dict[str, Site] = {}
        self.bounty_log: List[dict] = []
        self.leaderboard: Dict[str, int] = {}
        self.chat_log: deque = deque(maxlen=200)
        self.activity_log: deque = deque(maxlen=200)
        self.targets: Dict[str, dict] = {}  # name -> {"note", "points"}

    def taken_names(self):
        return {p.name.lower() for p in self.players.values()}

    def leaderboard_sorted(self):
        return sorted(
            [{"name": n, "score": s} for n, s in self.leaderboard.items()],
            key=lambda e: -e["score"],
        )

    def team_scores_sorted(self):
        totals = {team: 0 for team in TEAMS}
        for site in self.sites.values():
            if site.team in totals:
                totals[site.team] += site.score
        return sorted(
            [{"team": t, "score": s} for t, s in totals.items()],
            key=lambda e: -e["score"],
        )

    def points_for(self, label: str, is_registered_target: bool) -> int:
        if is_registered_target and label in self.targets:
            return self.targets[label].get("points", DEFAULT_TARGET_POINTS)
        return GENERIC_DETECTION_POINTS

    def snapshot(self):
        return {
            "sites": {sid: s.to_dict() for sid, s in self.sites.items()},
            "players": [p.to_dict() for p in self.players.values()],
            "leaderboard": self.leaderboard_sorted(),
            "team_scores": self.team_scores_sorted(),
            "chat_log": list(self.chat_log),
            "activity_log": list(self.activity_log),
            "targets": self.targets,
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
                    continue  # already joined this connection
                name = (msg.get("name") or "").strip()[:24]
                role = msg.get("role") if msg.get("role") in ("field_agent", "spectator") else "spectator"
                team = msg.get("team") if msg.get("team") in TEAMS else TEAMS[0]
                location = (msg.get("location") or "").strip()[:40] if role == "field_agent" else None
                avatar = (msg.get("avatar") or "◈").strip()[:4]

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

                player = Player(websocket, name, role, team, location, avatar)
                hub.players[websocket] = player

                if role == "field_agent":
                    site_id = _slugify(name)
                    hub.sites[site_id] = Site(site_id, owner=name, location=location, team=team, waypoints=[])
                    player.site_id = site_id

                await websocket.send_text(json.dumps({"type": "init", **hub.snapshot(), "your_player_id": player.player_id}))
                await broadcast_presence()
                role_label = f"Field Agent ({location}, Team {team})" if role == "field_agent" else f"Spectator (Team {team})"
                await log_activity(f"{name} joined as {role_label}.", "join")
                continue

            if player is None:
                continue  # ignore everything until joined

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
                )
                hub.sites[player.site_id].candidates[candidate.id] = candidate
                await broadcast(candidate.to_dict())
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
                # remove from whichever site they were following before
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
                    points = hub.points_for(candidate.label, candidate.is_registered_target)
                    entry = {
                        "site_id": site_id, "id": candidate.id, "label": candidate.label,
                        "confidence": candidate.confidence, "waypoint": candidate.waypoint,
                        "timestamp": candidate.timestamp, "contributors": contributors, "points": points,
                    }
                    hub.bounty_log.insert(0, entry)
                    hub.bounty_log = hub.bounty_log[:50]
                    for name in contributors:
                        hub.leaderboard[name] = hub.leaderboard.get(name, 0) + points
                    site.score += points  # credits the Field Agent's team, not just voters
                    del site.candidates[candidate.id]
                    await broadcast({"type": "match_confirmed", **entry})
                    await broadcast({"type": "leaderboard", "entries": hub.leaderboard_sorted(),
                                       "team_scores": hub.team_scores_sorted()})
                    await log_activity(
                        f'TARGET CONFIRMED -- "{candidate.label}" @ {candidate.waypoint} '
                        f"({site.owner}'s site, Team {site.team}) by {', '.join(contributors)} -- +{points} pts",
                        "match",
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


@app.post("/api/targets")
async def register_target(name: str = Form(...), note: str = Form(""), points: int = Form(DEFAULT_TARGET_POINTS),
                            photo: UploadFile = File(None)):
    # Global target registry across the hub for now -- per-site re-id
    # embeddings are a real follow-up piece, not solved in this pass.
    hub.targets[name] = {"note": note, "points": max(1, min(1000, points))}
    photo_info = {"filename": photo.filename, "content_type": photo.content_type} if photo else None
    await broadcast({"type": "targets_update", "targets": hub.targets})
    return {"status": "registered", "name": name, "points": hub.targets[name]["points"], "photo": photo_info}


@app.get("/api/state")
async def get_state():
    return hub.snapshot()


app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")