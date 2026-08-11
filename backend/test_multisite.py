"""
test_multisite.py -- verifies the Hub's multi-site contract with real
simulated WebSocket clients: join -> site-report -> candidate -> vote ->
confirmed_objects -> chat -> rate limiting, with actual printed output
checked against expectations.

Rewritten against the co-op recon mission contract (main.py, post
inner/outer pivot) -- the old version asserted on team/leaderboard/
points/the /api/targets registration endpoint, none of which exist
anymore. Every assertion here is checked against what main.py actually
does now, not just trimmed down from the old version.

Run with:
    python backend/test_multisite.py

Uses FastAPI's in-process TestClient -- no need for a running uvicorn
server or real network for this pass.
"""

from fastapi.testclient import TestClient

from backend.main import app, hub, _slugify as _slug

client = TestClient(app)


def expect(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(f"Test failed: {label}")


def run():
    hub.sites.clear()
    hub.players.clear()
    hub.confirmed_objects.clear()

    with client.websocket_connect("/ws") as priya, \
         client.websocket_connect("/ws") as alex:

        # -- joins --------------------------------------------------------
        # site_role is now REQUIRED for field_agent joins (no more team) --
        # confirming that requirement is actually enforced is worth its
        # own check, not just happy-pathing straight past it.
        priya.send_json({"type": "join", "name": "Priya", "role": "field_agent",
                          "site_role": "inner", "location": "Berlin, DE", "avatar": "\U0001F916"})
        init = priya.receive_json()
        expect("Priya join returns init", init["type"] == "init")
        expect("Priya gets a player_id", "your_player_id" in init and len(init["your_player_id"]) > 0)
        expect("Priya's site carries her site_role (inner)", init["sites"][_slug("Priya")]["role"] == "inner")
        expect("init carries a confirmed_objects map (even if empty)", "confirmed_objects" in init)
        expect("init carries an alerts list (even if empty)", "alerts" in init)
        # broadcasts reach the sender too -- drain Priya's own presence+activity
        priya.receive_json(); priya.receive_json()

        alex.send_json({"type": "join", "name": "Alex", "role": "spectator"})
        alex.receive_json()  # init
        alex.receive_json(); alex.receive_json()  # own presence + activity, self-included
        priya.receive_json(); priya.receive_json()  # sees Alex's presence + activity too

        # -- field_agent join REJECTED without site_role --------------------
        rejected_no_role = False
        with client.websocket_connect("/ws") as no_role:
            no_role.send_json({"type": "join", "name": "NoRole", "role": "field_agent",
                                "location": "Nowhere"})
            resp = no_role.receive_json()
            rejected_no_role = resp["type"] == "join_rejected"
        expect("field_agent join without site_role is rejected", rejected_no_role)

        # -- duplicate name rejected ---------------------------------------
        dupe_ok = True
        with client.websocket_connect("/ws") as dupe:
            dupe.send_json({"type": "join", "name": "priya", "role": "spectator"})
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

        # -- field agent reports site telemetry --------------------------------
        priya.send_json({"type": "site_report", "waypoints": [{"id": "W1", "x": 0, "y": 0}],
                          "telemetry": {"position": {"x": 1, "y": 1}, "battery": 87}})
        priya.receive_json()  # self-broadcast copy
        site_update = alex.receive_json()
        expect("Spectator receives site_update", site_update["type"] == "site_update")

        # -- candidate report, carrying the mission-narrative fields --------
        priya.send_json({
            "type": "candidate_report", "label": "object-03", "confidence": 0.9,
            "instant_confidence": 0.85, "waypoint": "W3", "is_registered_target": True,
            "location_offset": {"waypoint": "W3", "offset_x": 0.1, "offset_y": -0.2},
            "ocr_number": "03", "ocr_anomaly": None,
            "image": "ZmFrZS1qcGVn",  # not a real JPEG, just checking the field round-trips
            "twin_semantic_label": "tree",
        })
        priya.receive_json()  # self-broadcast copy
        candidate_msg = alex.receive_json()
        expect("Spectator receives candidate", candidate_msg["type"] == "candidate")
        expect("Candidate carries image field", candidate_msg["image"] == "ZmFrZS1qcGVn")
        expect("Candidate carries twin_semantic_label", candidate_msg["twin_semantic_label"] == "tree")
        expect("Candidate carries ocr_number", candidate_msg["ocr_number"] == "03")
        expect("Candidate has no leftover 'points' field", "points" not in candidate_msg)
        candidate_id = candidate_msg["id"]
        site_id = candidate_msg["site_id"]

        # -- voting to confirm (threshold = 2) -------------------------------
        alex.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "confirm"})
        alex.receive_json()  # self-broadcast copy
        vu1 = priya.receive_json()
        expect("First vote broadcasts vote_update", vu1["type"] == "vote_update" and vu1["confirms"] == 1)

        priya.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "confirm"})
        # self-copy: vote_update, match_confirmed, activity (no leaderboard broadcast anymore)
        seq = [priya.receive_json() for _ in range(3)]
        alex_seq = [alex.receive_json() for _ in range(3)]
        types = [m["type"] for m in alex_seq]
        expect("Threshold reached -> vote_update, match_confirmed, activity (no leaderboard)",
               types == ["vote_update", "match_confirmed", "activity"])
        confirmed_msg = alex_seq[1]
        expect("match_confirmed carries the same image/twin_semantic_label",
               confirmed_msg["image"] == "ZmFrZS1qcGVn" and confirmed_msg["twin_semantic_label"] == "tree")
        expect("match_confirmed has no 'points' field", "points" not in confirmed_msg)

        # -- confirmed_objects is the actual "where is object N" answer ------
        expect("confirmed_objects now contains this label",
               "object-03" in hub.confirmed_objects)
        expect("confirmed_objects entry carries location_offset",
               hub.confirmed_objects["object-03"]["location_offset"]["offset_x"] == 0.1)

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