// map3d.js -- 3D patrol map using Three.js
// Renders the waypoint graph as a ground scene with a moving robot marker
// and PERSISTENT beacon markers for confirmed target matches.

let scene, camera, renderer;
let robotGroup, robotBeaconLight;
let waypointNodes = {};
let targetMarkers = [];
let rotY = 0.4, rotX = -0.55, dragging = false, lastX = 0, lastY = 0;
let zoom = 26;

const GRID_SCALE = 0.9; // maps 0-100 waypoint grid -> world units (-45..45)

function toWorld(x, y) {
  return {
    wx: (x - 50) * GRID_SCALE,
    wz: (y - 50) * GRID_SCALE,
  };
}

function makeLabelSprite(text, color = "#4ADE80") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 34px JetBrains Mono, monospace";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(8, 2, 1);
  return sprite;
}

function initMap3D(containerId, waypoints, onWaypointClick) {
  const container = document.getElementById(containerId);
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0a0d0b, 40, 110);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  // ground grid
  const grid = new THREE.GridHelper(100, 24, 0x2a3a30, 0x1b2620);
  scene.add(grid);

  const groundGeo = new THREE.CircleGeometry(60, 48);
  const groundMat = new THREE.MeshBasicMaterial({ color: 0x0e1512, transparent: true, opacity: 0.6 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.05;
  scene.add(ground);

  // ambient + accent lights
  scene.add(new THREE.AmbientLight(0x6f8b7c, 0.9));
  const key = new THREE.PointLight(0x4ade80, 1.2, 80);
  key.position.set(0, 25, 0);
  scene.add(key);

  // waypoint beacons
  waypoints.forEach((wp) => {
    const { wx, wz } = toWorld(wp.x, wp.y);
    const group = new THREE.Group();
    group.position.set(wx, 0, wz);

    const postGeo = new THREE.CylinderGeometry(0.15, 0.15, 2.4, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x223028, emissive: 0x0d1611 });
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.y = 1.2;
    group.add(post);

    const beaconGeo = new THREE.SphereGeometry(0.45, 16, 16);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0x1b2620, emissive: 0xf5a623, emissiveIntensity: 0.5,
    });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.y = 2.6;
    group.add(beacon);

    const label = makeLabelSprite(wp.id, "#6E8177");
    label.position.set(0, 3.6, 0);
    group.add(label);

    scene.add(group);
    waypointNodes[wp.id] = { group, beacon };
  });

  // connecting path loop
  const pathPoints = waypoints.map((wp) => {
    const { wx, wz } = toWorld(wp.x, wp.y);
    return new THREE.Vector3(wx, 0.05, wz);
  });
  pathPoints.push(pathPoints[0]);
  const pathGeo = new THREE.BufferGeometry().setFromPoints(pathPoints);
  const pathMat = new THREE.LineDashedMaterial({ color: 0x2a3a30, dashSize: 1, gapSize: 0.6 });
  const pathLine = new THREE.Line(pathGeo, pathMat);
  pathLine.computeLineDistances();
  scene.add(pathLine);

  // robot marker
  robotGroup = new THREE.Group();
  const bodyGeo = new THREE.BoxGeometry(1.4, 0.5, 1.8);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x0e1512, emissive: 0x1a3025 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = 0.35;
  robotGroup.add(body);

  const coneGeo = new THREE.ConeGeometry(0.35, 0.9, 4);
  const coneMat = new THREE.MeshStandardMaterial({ color: 0x4ade80, emissive: 0x2f8f5a });
  const cone = new THREE.Mesh(coneGeo, coneMat);
  cone.rotation.x = Math.PI / 2;
  cone.position.set(0, 0.4, 1.1);
  robotGroup.add(cone);

  robotBeaconLight = new THREE.PointLight(0x4ade80, 1.5, 12);
  robotBeaconLight.position.set(0, 1.5, 0);
  robotGroup.add(robotBeaconLight);

  scene.add(robotGroup);

  bindMouseControls(renderer.domElement, onWaypointClick);
  animate();

  window.addEventListener("resize", () => onResize(container));
}

function bindMouseControls(el, onWaypointClick) {
  let didDrag = false;
  el.addEventListener("mousedown", (e) => { dragging = true; didDrag = false; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => (dragging = false));
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    didDrag = true;
    rotY += (e.clientX - lastX) * 0.006;
    rotX = Math.max(-1.2, Math.min(-0.15, rotX + (e.clientY - lastY) * 0.004));
    lastX = e.clientX;
    lastY = e.clientY;
  });
  el.addEventListener("wheel", (e) => {
    zoom = Math.max(12, Math.min(50, zoom + e.deltaY * 0.02));
    e.preventDefault();
  }, { passive: false });

  el.addEventListener("click", (e) => {
    if (didDrag) return; // ignore click that ends a drag
    if (!onWaypointClick) return;
    const rect = el.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    const beacons = Object.entries(waypointNodes).map(([id, node]) => node.beacon);
    const hits = raycaster.intersectObjects(beacons);
    if (hits.length > 0) {
      const hitBeacon = hits[0].object;
      const wpId = Object.entries(waypointNodes).find(([id, n]) => n.beacon === hitBeacon)[0];
      onWaypointClick(wpId);
    }
  });
}

function spawnPingMarker3D(waypointId, color = "#38BDF8") {
  const node = waypointNodes[waypointId];
  if (!node) return;
  const ringGeo = new THREE.RingGeometry(0.3, 0.5, 24);
  const ringMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color), side: THREE.DoubleSide, transparent: true, opacity: 1 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(0, 2.6, 0);
  node.group.add(ring);

  const start = performance.now();
  function grow() {
    const elapsed = performance.now() - start;
    const t = Math.min(elapsed / 900, 1);
    ring.scale.setScalar(1 + t * 6);
    ring.material.opacity = 1 - t;
    if (t < 1) requestAnimationFrame(grow);
    else node.group.remove(ring);
  }
  grow();
}

function onResize(container) {
  const width = container.clientWidth;
  const height = container.clientHeight;
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

  const t = performance.now() * 0.003;
  Object.values(waypointNodes).forEach(({ beacon }) => {
    beacon.material.emissiveIntensity = 0.35 + Math.sin(t + beacon.id) * 0.2;
  });

  renderer.render(scene, camera);
}

function updateRobotPosition3D(x, y, headingDeg) {
  if (!robotGroup) return;
  const { wx, wz } = toWorld(x, y);
  robotGroup.position.set(wx, 0, wz);
  robotGroup.rotation.y = THREE.MathUtils.degToRad(headingDeg || 0);
}

function pingWaypoint3D(waypointId, isMatch) {
  const node = waypointNodes[waypointId];
  if (!node) return;
  node.beacon.material.emissive.setHex(isMatch ? 0x4ade80 : 0xf5a623);
  node.beacon.material.emissiveIntensity = 1.4;
}

function addTargetMarker3D(waypointId, label, confidence) {
  const node = waypointNodes[waypointId];
  if (!node) return;

  const markerGeo = new THREE.RingGeometry(0.9, 1.1, 32);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0x4ade80, side: THREE.DoubleSide, transparent: true, opacity: 0.8 });
  const ring = new THREE.Mesh(markerGeo, markerMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(0, 0.05, 0);
  node.group.add(ring);

  const tag = makeLabelSprite(`${label} ${(confidence * 100).toFixed(0)}%`, "#4ADE80");
  tag.position.set(0, 4.5, 0);
  node.group.add(tag);

  targetMarkers.push({ waypointId, label, confidence });
}