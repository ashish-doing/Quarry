// app.js -- QUARRY unified Hub client. One screen, no tabs: arena map,
// focused live feed, team scores, squad, activity, sightings, and chat
// are all visible simultaneously so nobody has to switch views mid-hunt.

const $ = (id) => document.getElementById(id);
const AVATARS = ["🤖", "🦅", "🦊", "🐺", "🐍", "🎯", "🛰️", "⚡", "🦂", "🐉", "🛡️", "🔭"];

let ws = null;
let myName = "", myRole = "spectator", myTeam = "Alpha", myAvatar = AVATARS[0], myPlayerId = null;
let sites = {};
let players = [];
let followingSiteId = null;
let openSightings = new Map();
let arenaReady = false;

// ---------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------
setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString("en-GB"); }, 1000);

// ---------------------------------------------------------------------
// Join card wiring: avatar grid, role, team
// ---------------------------------------------------------------------
function buildAvatarGrid() {
  const grid = $("avatar-grid");
  AVATARS.forEach((a, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "avatar-option" + (i === 0 ? " selected" : "");
    btn.textContent = a;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".avatar-option").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      myAvatar = a;
    });
    grid.appendChild(btn);
  });
}
buildAvatarGrid();

document.querySelectorAll(".role-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".role-btn").forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    myRole = btn.dataset.role;
    $("join-location").classList.toggle("hidden", myRole !== "field_agent");
  });
});
document.querySelectorAll(".team-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".team-btn").forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    myTeam = btn.dataset.team;
  });
});

$("join-btn").addEventListener("click", () => {
  const name = $("join-name").value.trim();
  const location = $("join-location").value.trim();
  $("join-error").classList.add("hidden");
  if (!name) { $("join-name").focus(); return; }
  if (myRole === "field_agent" && !location) { $("join-location").focus(); return; }

  myName = name;
  const doJoin = () => ws.send(JSON.stringify({
    type: "join", name, role: myRole, team: myTeam, avatar: myAvatar,
    location: myRole === "field_agent" ? location : undefined,
  }));
  if (ws && ws.readyState === WebSocket.OPEN) doJoin();
  else if (ws) ws.addEventListener("open", doJoin, { once: true });
});
$("join-name").addEventListener("keydown", (e) => { if (e.key === "Enter") $("join-btn").click(); });

// ---------------------------------------------------------------------
// Drop sequence -- plays after the Hub accepts the join, before the HUD appears
// ---------------------------------------------------------------------
function playDropSequence() {
  $("join-overlay").classList.add("hidden");
  const dropOverlay = $("drop-overlay");
  dropOverlay.classList.remove("hidden");
  $("drop-avatar").textContent = myAvatar;
  $("drop-name").textContent = myName;
  const teamEl = $("drop-team");
  teamEl.textContent = `TEAM ${myTeam.toUpperCase()} — ${myRole === "field_agent" ? "FIELD AGENT" : "SPECTATOR"}`;
  teamEl.style.color = myTeam === "Alpha" ? "#38BDF8" : "#F0453A";

  let n = 3;
  const countEl = $("drop-count");
  countEl.textContent = n;
  const timer = setInterval(() => {
    n -= 1;
    if (n <= 0) {
      clearInterval(timer);
      dropOverlay.classList.add("hidden");
      revealHUD();
      return;
    }
    countEl.textContent = n;
  }, 700);
}

function revealHUD() {
  const badge = $("me-badge");
  badge.classList.remove("hidden");
  badge.innerHTML = `<span class="avatar-badge">${myAvatar}</span>${myName} <span class="team-tag ${myTeam}">${myTeam}</span>`;
  if (!arenaReady) {
    initArena("arena-container", (siteId) => sendFollow(siteId));
    arenaReady = true;
    renderArenaMarkers();
  }
}

// ---------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------
function toast(text, kind = "info") {
  const stack = $("toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = text;
  stack.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity 0.3s"; setTimeout(() => el.remove(), 300); }, 4000);
}

// ---------------------------------------------------------------------
// Arena markers
// ---------------------------------------------------------------------
function renderArenaMarkers() {
  if (!arenaReady) return;
  Object.entries(sites).forEach(([siteId, site]) => upsertSiteMarker(siteId, site));
  removeStaleMarkers(Object.keys(sites));
  if (followingSiteId) highlightFollowed(followingSiteId);
}

function sendFollow(siteId) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  followingSiteId = siteId;
  ws.send(JSON.stringify({ type: "follow", site_id: siteId }));
  renderFeedPanel();
  if (arenaReady) highlightFollowed(siteId);
}

// ---------------------------------------------------------------------
// Focused feed panel
// ---------------------------------------------------------------------
let feedRefreshTimer = null;
const LOCAL_FRAME_SERVER = "http://127.0.0.1:8099/latest.jpg";
// NOTE: hardcoded to localhost -- only works when the browser and the
// Field Agent's site_agent.py are on the SAME machine, which is true for
// today's testing. A remote spectator watching a different Field Agent's
// feed needs a real per-site URL, which depends on the still-pending
// ngrok/tunnel work -- not solved here, flagging honestly rather than
// pretending this generalizes.

function stopFeedRefresh() {
  if (feedRefreshTimer) clearInterval(feedRefreshTimer);
  feedRefreshTimer = null;
}

function renderFeedPanel() {
  const site = sites[followingSiteId];
  stopFeedRefresh();
  if (!site) {
    $("feed-title").textContent = "Live Feed — select a site";
    $("feed-video").innerHTML = "Click a marker on the Arena above";
    $("feed-watchers").textContent = "";
    $("feed-telemetry").innerHTML = "";
    return;
  }
  $("feed-title").textContent = `Live Feed — ${site.owner}'s site (${site.location || "?"})`;
  $("feed-watchers").textContent = site.followers.length ? `👁 ${site.followers.join(", ")}` : "";
  $("feed-video").innerHTML = `<img id="feed-img" src="${LOCAL_FRAME_SERVER}" style="width:100%;height:100%;object-fit:contain;" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
    <div style="display:none; align-items:center; justify-content:center; height:100%; color:var(--text-dim); font-family:var(--font-mono); font-size:11px;">No local feed reachable — is site_agent.py running on this machine?</div>`;
  feedRefreshTimer = setInterval(() => {
    const img = document.getElementById("feed-img");
    if (img) img.src = `${LOCAL_FRAME_SERVER}?t=${Date.now()}`;
  }, 300);
  const t = site.telemetry || {};
  const pos = t.position || {};
  $("feed-telemetry").innerHTML = `
    <div class="ft-item"><span class="ft-label">POSITION</span><span class="ft-value">${pos.x?.toFixed?.(1) ?? "--"},${pos.y?.toFixed?.(1) ?? "--"}</span></div>
    <div class="ft-item"><span class="ft-label">HEADING</span><span class="ft-value">${t.heading ?? "--"}°</span></div>
    <div class="ft-item"><span class="ft-label">SPEED</span><span class="ft-value">${t.speed ?? "--"}</span></div>
    <div class="ft-item"><span class="ft-label">BATTERY</span><span class="ft-value">${t.battery ?? "--"}%</span></div>
  `;
}

// ---------------------------------------------------------------------
// Team scores (totals on top, per-site ranking within each team below)
// ---------------------------------------------------------------------
function renderTeamScores(teamScoresData) {
  const container = $("team-scores");
  container.innerHTML = "";
  const totals = teamScoresData || [];
  totals.forEach((teamEntry) => {
    const teamSites = Object.values(sites)
      .filter((s) => s.team === teamEntry.team)
      .sort((a, b) => b.score - a.score);
    const block = document.createElement("div");
    block.className = `team-block ${teamEntry.team}`;
    block.innerHTML = `<div class="team-block-header"><span>TEAM ${teamEntry.team.toUpperCase()}</span><span>${teamEntry.score} pts</span></div>`;
    if (teamSites.length === 0) {
      block.innerHTML += `<div class="team-site-row"><span>No sites yet</span><span></span></div>`;
    } else {
      teamSites.forEach((s) => {
        block.innerHTML += `<div class="team-site-row"><span>${s.owner}</span><span>${s.score} pts</span></div>`;
      });
    }
    container.appendChild(block);
  });
}

// ---------------------------------------------------------------------
// Squad + Activity
// ---------------------------------------------------------------------
function renderSquad() {
  $("squad-count").textContent = `(${players.length})`;
  const list = $("squad-list");
  list.innerHTML = "";
  players.forEach((p) => {
    const row = document.createElement("div");
    row.className = "squad-row";
    row.innerHTML = `
      <span class="avatar-badge">${p.avatar || "◈"}</span>
      <div class="squad-row-info">
        <div class="squad-row-name ${p.name === myName ? "you" : ""}">${p.name}</div>
        <div class="squad-row-meta">
          <span class="role-pill ${p.role}">${p.role === "field_agent" ? "AGENT" : "SPEC"}</span>
          Team ${p.team}${p.location ? " · " + p.location : ""}
        </div>
      </div>
      <span class="conn-dot ${p.connected ? "" : "offline"}"></span>
    `;
    list.appendChild(row);
  });
}

function renderActivity(entry) {
  const feed = $("activity-feed");
  const line = document.createElement("div");
  line.className = `act-line ${entry.kind || ""}`;
  const time = new Date(entry.timestamp).toLocaleTimeString("en-GB");
  line.innerHTML = `<span class="act-time">${time}</span>${entry.text}`;
  feed.prepend(line);
  while (feed.children.length > 60) feed.removeChild(feed.lastChild);
}

// ---------------------------------------------------------------------
// Sightings
// ---------------------------------------------------------------------
function renderSightings() {
  const list = $("sightings-list");
  list.innerHTML = "";
  if (openSightings.size === 0) {
    list.innerHTML = `<li class="empty">No pending sightings.</li>`;
    return;
  }
  for (const [id, entry] of openSightings) {
    const { data, confirms = 0, disputes = 0, myVote } = entry;
    const owner = sites[data.site_id]?.owner || data.site_id;
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="sighting-site">${owner}'s site</span>
      <div class="sighting-header">
        <span>${data.label}${data.is_registered_target ? " ★" : ""} @ ${data.waypoint}</span>
        <span class="conf">${(data.confidence * 100).toFixed(0)}% avg<span class="conf-instant"> (${((data.instant_confidence ?? data.confidence) * 100).toFixed(0)}% now)</span></span>
      </div>
      <div class="sighting-actions">
        <button class="btn-confirm" data-id="${id}" data-site="${data.site_id}" data-vote="confirm" ${myVote === "confirm" ? "disabled" : ""}>CONFIRM</button>
        <button class="btn-dispute" data-id="${id}" data-site="${data.site_id}" data-vote="dispute" ${myVote === "dispute" ? "disabled" : ""}>DISPUTE</button>
      </div>
      <div class="sighting-tally">confirms: ${confirms} · disputes: ${disputes}${myVote ? ` · you voted ${myVote}` : ""}</div>
    `;
    list.appendChild(li);
  }
  list.querySelectorAll("button[data-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.id, siteId = btn.dataset.site, vote = btn.dataset.vote;
      ws.send(JSON.stringify({ type: "vote", site_id: siteId, id, vote }));
      const entry = openSightings.get(id);
      if (entry) { entry.myVote = vote; renderSightings(); }
    });
  });
}

// ---------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------
function renderChatMessage(msg) {
  const log = $("chat-log");
  const row = document.createElement("div");
  row.className = "chat-msg";
  const time = new Date(msg.timestamp).toLocaleTimeString("en-GB");
  row.innerHTML = `<div class="chat-msg-body"><span class="chat-msg-sender">${msg.sender}</span>${msg.text} <span class="chat-msg-time">${time}</span></div>`;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "chat", text }));
  input.value = "";
});

// ---------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------
function connect() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${window.location.host}/ws`);
  const dot = $("link-dot"), label = $("link-label");

  ws.onopen = () => { dot.className = "status-dot live"; label.textContent = "LINK ACTIVE"; };
  ws.onclose = () => { dot.className = "status-dot dead"; label.textContent = "LINK LOST — reconnecting…"; setTimeout(connect, 2000); };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case "init":
        sites = msg.sites || {};
        players = msg.players || [];
        myPlayerId = msg.your_player_id;
        renderSquad();
        renderTeamScores(msg.team_scores || []);
        (msg.chat_log || []).forEach(renderChatMessage);
        (msg.activity_log || []).slice(-30).forEach(renderActivity);
        Object.values(sites).forEach((s) => (s.candidates || []).forEach((c) =>
          openSightings.set(c.id, { data: c, confirms: 0, disputes: 0 })));
        renderSightings();
        playDropSequence();
        break;

      case "join_rejected":
        $("join-error").textContent = msg.reason;
        $("join-error").classList.remove("hidden");
        break;

      case "presence":
        players = msg.players;
        renderSquad();
        break;

      case "activity":
        renderActivity(msg);
        if (msg.kind === "join") toast(msg.text, "info");
        break;

      case "site_update":
        if (!sites[msg.site_id]) sites[msg.site_id] = { site_id: msg.site_id, owner: msg.site_id, candidates: [], followers: [], score: 0, team: "Alpha" };
        sites[msg.site_id].telemetry = msg.telemetry;
        renderArenaMarkers();
        if (msg.site_id === followingSiteId) renderFeedPanel();
        break;

      case "candidate":
        openSightings.set(msg.id, { data: msg, confirms: 0, disputes: 0 });
        renderSightings();
        if (arenaReady) pulseNewCandidate(msg.site_id);
        toast(`New sighting: ${msg.label} (${(msg.confidence * 100).toFixed(0)}%)`, "warn");
        break;

      case "vote_update": {
        const entry = openSightings.get(msg.id);
        if (entry) { entry.confirms = msg.confirms; entry.disputes = msg.disputes; renderSightings(); }
        break;
      }

      case "match_confirmed":
        openSightings.delete(msg.id);
        renderSightings();
        toast(`TARGET CONFIRMED: ${msg.label} (+${msg.points} pts)`, "match");
        break;

      case "leaderboard":
        renderTeamScores(msg.team_scores || []);
        break;

      case "site_removed":
        delete sites[msg.site_id];
        if (msg.site_id === followingSiteId) { followingSiteId = null; renderFeedPanel(); }
        renderArenaMarkers();
        break;

      case "followers_update":
        if (sites[msg.site_id]) sites[msg.site_id].followers = msg.followers;
        if (msg.site_id === followingSiteId) renderFeedPanel();
        break;

      case "team_ping":
        renderActivity({ text: `${msg.player} pinged ${msg.waypoint}${msg.text ? ": " + msg.text : ""}`, kind: "info", timestamp: msg.timestamp });
        break;

      case "chat":
        renderChatMessage(msg);
        break;

      case "collision_alert":
        renderActivity({
          text: `⚠ SAFETY STOP — ${sites[msg.site_id]?.owner || "a"} robot stopped, object too close`,
          kind: "danger",
          timestamp: new Date(msg.timestamp * 1000).toISOString(),
        });
        toast("SAFETY STOP triggered", "warn");
        break;

      case "collision_clear":
        renderActivity({
          text: `✓ SAFETY — ${sites[msg.site_id]?.owner || "a"} robot re-armed, path clear`,
          kind: "info",
          timestamp: new Date(msg.timestamp * 1000).toISOString(),
        });
        break;

      case "rate_limited":
        toast("Slow down — rate limited.", "warn");
        break;
    }
  };
}

connect();