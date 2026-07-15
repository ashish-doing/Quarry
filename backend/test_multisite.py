"""
test_multisite.py -- verifies the Hub's multi-site contract with real
simulated WebSocket clients, per the rebuild brief's own testing
standard (Section 6): join -> site-report -> vote -> chat -> leaderboard,
with actual printed output checked against expectations, before this is
ever wired into the UI.

Run with:
    python backend/test_multisite.py

Uses FastAPI's in-process TestClient -- no need for a running uvicorn
server or real network for this pass.
"""

from fastapi.testclient import TestClient

from backend.main import app, _slugify as _slug

client = TestClient(app)


def expect(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(f"Test failed: {label}")


def run():
    # register a target with a custom point value BEFORE anyone joins,
    # via the real HTTP endpoint (not a shortcut) -- confirms the points
    # actually flow through to the leaderboard, not just a flat +1
    reg = client.post("/api/targets", data={"name": "Archer", "note": "high-value", "points": 250})
    expect("Target registration returns 200", reg.status_code == 200)
    expect("Target registration echoes points", reg.json()["points"] == 250)

    with client.websocket_connect("/ws") as priya, \
         client.websocket_connect("/ws") as alex:

        # -- joins --------------------------------------------------------
        priya.send_json({"type": "join", "name": "Priya", "role": "field_agent",
                          "team": "Alpha", "location": "Berlin, DE", "avatar": "🤖"})
        init = priya.receive_json()
        expect("Priya join returns init", init["type"] == "init")
        expect("Priya gets a player_id", "your_player_id" in init and len(init["your_player_id"]) > 0)
        expect("Priya's site carries her team", init["sites"][_slug("Priya")]["team"] == "Alpha")
        # broadcasts reach the sender too -- drain Priya's own presence+activity
        priya.receive_json(); priya.receive_json()

        alex.send_json({"type": "join", "name": "Alex", "role": "spectator", "team": "Bravo"})
        alex.receive_json()  # init
        alex.receive_json(); alex.receive_json()  # own presence + activity, self-included
        priya.receive_json(); priya.receive_json()  # sees Alex's presence + activity too

        # -- duplicate name rejected ---------------------------------------
        dupe_ok = True
        with client.websocket_connect("/ws") as dupe:
            dupe.send_json({"type": "join", "name": "priya", "role": "spectator", "team": "Alpha"})
            resp = dupe.receive_json()
            dupe_ok = resp["type"] == "join_rejected"
        expect("Duplicate name (case-insensitive) rejected", dupe_ok)

        # -- follow tracking ("who's watching whom") -------------------------
        alex.send_json({"type": "follow", "site_id": _slug("Priya")})
        follow_msg = alex.receive_json()
        expect("Follow broadcasts followers_update", follow_msg["type"] == "followers_update" and "Alex" in follow_msg["followers"])
        priya.receive_json()  # priya's copy
        priya.receive_json()  # activity log line for the follow
        alex.receive_json()   # alex's own copy of that same activity line

        # -- field agent reports site + a candidate matching the registered target --
        priya.send_json({"type": "site_report", "waypoints": [{"id": "W1", "x": 0, "y": 0}],
                          "telemetry": {"position": {"x": 1, "y": 1}, "battery": 87}})
        priya.receive_json()  # self-broadcast copy
        site_update = alex.receive_json()
        expect("Spectator receives site_update", site_update["type"] == "site_update")

        priya.send_json({"type": "candidate_report", "label": "Archer", "confidence": 0.9,
                          "waypoint": "W1", "is_registered_target": True})
        priya.receive_json()  # self-broadcast copy
        candidate_msg = alex.receive_json()
        expect("Spectator receives candidate", candidate_msg["type"] == "candidate")
        candidate_id = candidate_msg["id"]
        site_id = candidate_msg["site_id"]

        # -- voting to confirm (threshold = 2) -------------------------------
        alex.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "confirm"})
        alex.receive_json()  # self-broadcast copy
        vu1 = priya.receive_json()
        expect("First vote broadcasts vote_update", vu1["type"] == "vote_update" and vu1["confirms"] == 1)

        priya.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "confirm"})
        seq = [priya.receive_json() for _ in range(4)]  # self-copy: vote_update, match_confirmed, leaderboard, activity
        alex_seq = [alex.receive_json() for _ in range(4)]
        types = [m["type"] for m in alex_seq]
        expect("Threshold reached -> full broadcast sequence",
               types == ["vote_update", "match_confirmed", "leaderboard", "activity"])
        expect("Confirmed target awards its registered point value (250), not a flat 1",
               alex_seq[1]["points"] == 250)
        expect("Leaderboard reflects real points, not flat +1",
               {e["name"]: e["score"] for e in alex_seq[2]["entries"]} == {"Priya": 250, "Alex": 250})
        expect("Team Alpha (Priya's team) leads team_scores with 250",
               alex_seq[2]["team_scores"][0] == {"team": "Alpha", "score": 250})

        # -- chat -------------------------------------------------------------
        alex.send_json({"type": "chat", "text": "nice catch!"})
        chat_msg = alex.receive_json()  # self-broadcast copy, check it directly
        expect("Chat message broadcasts with sender name", chat_msg["type"] == "chat" and chat_msg["sender"] == "Alex")
        priya.receive_json()  # priya's copy, not asserted further

        # -- rate limiting -----------------------------------------------------
        limited = False
        for _ in range(12):
            alex.send_json({"type": "chat", "text": "spam"})
        for _ in range(12):
            msg = alex.receive_json()
            if msg["type"] == "rate_limited":
                limited = True
                break
        expect("Rate limiting kicks in under spam", limited)

    print("\nAll multi-site Hub checks passed.")


if __name__ == "__main__":
    run()