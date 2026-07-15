// arena.js -- shared arena map for QUARRY multi-site Hub.
// Every connected Field Agent site gets a marker at its server-assigned
// arena_position (arbitrary, deterministic -- NOT real GPS, sites have no
// real spatial relationship). Team-colored. Click a marker to follow that
// site's feed.

let scene, camera, renderer, raycaster;
let markerNodes = {}; // site_id -> {group, disc}
let radarSweep = null;
const TEAM_COLORS = { Alpha: 0x38bdf8, Bravo: 0xf0453a };
let rotY = 0.5, rotX = -0.6, dragging = false, lastX = 0, lastY = 0, zoom = 55;

function arenaLabel(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 30px JetBrains Mono, monospace";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(9, 2.2, 1);
  return sprite;
}

function initArena(containerId, onMarkerClick) {
  const container = document.getElementById(containerId);
  const width = container.clientWidth, height = container.clientHeight;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0a0d0b, 60, 160);
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 600);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const grid = new THREE.GridHelper(140, 28, 0x2a3a30, 0x1b2620);
  scene.add(grid);
  scene.add(new THREE.AmbientLight(0x6f8b7c, 1.0));
  const key = new THREE.PointLight(0x4ade80, 1.0, 120);
  key.position.set(0, 40, 0);
  scene.add(key);

  // radar sweep -- slow rotating translucent wedge, purely atmospheric
  const sweepGeo = new THREE.CircleGeometry(70, 32, 0, Math.PI / 6);
  const sweepMat = new THREE.MeshBasicMaterial({ color: 0x4ade80, transparent: true, opacity: 0.06, side: THREE.DoubleSide });
  radarSweep = new THREE.Mesh(sweepGeo, sweepMat);
  radarSweep.rotation.x = -Math.PI / 2;
  radarSweep.position.y = 0.02;
  scene.add(radarSweep);

  raycaster = new THREE.Raycaster();
  bindControls(renderer.domElement, onMarkerClick);
  animate();
  window.addEventListener("resize", () => onResize(container));
}

function bindControls(el, onMarkerClick) {
  let didDrag = false;
  el.addEventListener("mousedown", (e) => { dragging = true; didDrag = false; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => (dragging = false));
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    didDrag = true;
    rotY += (e.clientX - lastX) * 0.006;
    rotX = Math.max(-1.3, Math.min(-0.2, rotX + (e.clientY - lastY) * 0.004));
    lastX = e.clientX; lastY = e.clientY;
  });
  el.addEventListener("wheel", (e) => {
    zoom = Math.max(20, Math.min(100, zoom + e.deltaY * 0.03));
    e.preventDefault();
  }, { passive: false });

  el.addEventListener("click", (e) => {
    if (didDrag) return;
    const rect = el.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(mouse, camera);
    const discs = Object.values(markerNodes).map((n) => n.disc);
    const hits = raycaster.intersectObjects(discs);
    if (hits.length > 0) {
      const siteId = Object.entries(markerNodes).find(([, n]) => n.disc === hits[0].object)[0];
      onMarkerClick(siteId);
    }
  });
}

function upsertSiteMarker(siteId, siteData) {
  if (markerNodes[siteId]) {
    markerNodes[siteId].group.userData.owner = siteData.owner;
    return;
  }
  const pos = siteData.arena_position || { x: 0, y: 0 };
  const color = TEAM_COLORS[siteData.team] || 0x6e8177;

  const group = new THREE.Group();
  group.position.set(pos.x, 0, pos.y);

  const postGeo = new THREE.CylinderGeometry(0.2, 0.2, 3, 8);
  const postMat = new THREE.MeshStandardMaterial({ color: 0x223028, emissive: 0x0d1611 });
  const post = new THREE.Mesh(postGeo, postMat);
  post.position.y = 1.5;
  group.add(post);

  const discGeo = new THREE.CylinderGeometry(1.6, 1.6, 0.3, 24);
  const discMat = new THREE.MeshStandardMaterial({ color: 0x1b2620, emissive: color, emissiveIntensity: 0.6 });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.position.y = 3.2;
  group.add(disc);

  // glowing ground ring -- gives each site real presence instead of a flat dot
  const ringGeo = new THREE.RingGeometry(2.4, 2.9, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.5 });
  const groundRing = new THREE.Mesh(ringGeo, ringMat);
  groundRing.rotation.x = -Math.PI / 2;
  groundRing.position.y = 0.05;
  group.add(groundRing);

  const label = arenaLabel(siteData.owner, siteData.team === "Alpha" ? "#38BDF8" : "#F0453A");
  label.position.set(0, 4.6, 0);
  group.add(label);

  scene.add(group);
  markerNodes[siteId] = { group, disc };
}

function removeStaleMarkers(activeSiteIds) {
  Object.keys(markerNodes).forEach((siteId) => {
    if (!activeSiteIds.includes(siteId)) {
      scene.remove(markerNodes[siteId].group);
      delete markerNodes[siteId];
    }
  });
}

function highlightFollowed(siteId) {
  Object.entries(markerNodes).forEach(([id, node]) => {
    node.disc.material.emissiveIntensity = id === siteId ? 1.6 : 0.6;
  });
}

function pulseNewCandidate(siteId) {
  const node = markerNodes[siteId];
  if (!node) return;
  const ringGeo = new THREE.RingGeometry(1.8, 2.1, 24);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xf5a623, side: THREE.DoubleSide, transparent: true, opacity: 1 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 3.2;
  node.group.add(ring);
  const start = performance.now();
  function grow() {
    const t = Math.min((performance.now() - start) / 900, 1);
    ring.scale.setScalar(1 + t * 3);
    ring.material.opacity = 1 - t;
    if (t < 1) requestAnimationFrame(grow); else node.group.remove(ring);
  }
  grow();
}

function onResize(container) {
  const width = container.clientWidth, height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function updateCamera() {
  camera.position.x = zoom * Math.sin(rotY) * Math.cos(rotX);
  camera.position.y = -zoom * Math.sin(rotX);
  camera.position.z = zoom * Math.cos(rotY) * Math.cos(rotX);
  camera.lookAt(0, 0, 0);
}

function animate() {
  requestAnimationFrame(animate);
  updateCamera();
  if (radarSweep) radarSweep.rotation.z += 0.006;
  renderer.render(scene, camera);
}