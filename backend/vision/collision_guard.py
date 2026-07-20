"""
collision_guard.py -- proximity-based emergency stop for QUARRY.

IMPORTANT HONESTY NOTE: this is NOT depth-based obstacle avoidance.
The UGV Beast kit has no lidar or depth camera (confirmed in the
master doc's hardware audit -- this SKU is vision-only), so there is
no way to get a real distance measurement from a single RGB frame.

What this DOES give you: a fast, independent loop that watches how
much of the frame the nearest detection fills, and issues an
immediate STOP the moment something gets close enough that
bbox-area-as-a-proxy-for-proximity crosses a conservative threshold.
Treat this as a coarse "something is right in front of the camera,
stop now" reflex -- not a planner that reliably steers around things.
Don't demo this as "obstacle avoidance" without this caveat; it's an
emergency brake, which is what actually protects the hardware.

Runs as its own asyncio task, independent of the candidate-raising
loop in real_feed.py (which is deliberately cooldown-limited for game
purposes -- a safety stop can never wait on that cooldown).
"""

import asyncio
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger("quarry.vision.collision_guard")

CHECK_INTERVAL_S = 0.15          # fast enough to catch the robot driving into something
PROXIMITY_AREA_THRESHOLD = 0.30  # nearest bbox fills >=30% of frame -> treat as "too close"
CLEAR_STREAK_REQUIRED = 4        # consecutive clear reads before re-arming (debounce)


class CollisionGuard:
    def __init__(self, twin, detector, on_event: Optional[Callable[[dict], None]] = None):
        self.twin = twin
        self.detector = detector
        self.on_event = on_event  # optional callback(dict) to surface alerts to the dashboard
        self.blocked = False
        self._clear_streak = 0

    @staticmethod
    def _bbox_area_fraction(bbox) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    async def run(self):
        logger.info(
            "Collision guard active (stop threshold=%.0f%% of frame, checking every %.2fs)",
            PROXIMITY_AREA_THRESHOLD * 100, CHECK_INTERVAL_S,
        )
        while True:
            await asyncio.sleep(CHECK_INTERVAL_S)
            if self.twin is None or not self.detector.available:
                continue

            try:
                frame = self.twin.get_frame('numpy', source='remote_edge')
            except Exception as exc:  # noqa: BLE001 -- a bad read must not kill this loop
                logger.debug("collision guard: frame read failed (%s)", exc)
                continue
            if frame is None:
                continue

            detections = self.detector.detect(frame)
            nearest_fraction = max(
                (self._bbox_area_fraction(d.bbox) for d in detections), default=0.0
            )

            if nearest_fraction >= PROXIMITY_AREA_THRESHOLD:
                self._clear_streak = 0
                if not self.blocked:
                    self.blocked = True
                    logger.warning(
                        "COLLISION GUARD: nearest object fills %.0f%% of frame -- stopping",
                        nearest_fraction * 100,
                    )
                    await self._emergency_stop()
            else:
                self._clear_streak += 1
                if self.blocked and self._clear_streak >= CLEAR_STREAK_REQUIRED:
                    self.blocked = False
                    logger.info("Collision guard: path clear, re-armed")
                    if self.on_event:
                        self.on_event({"type": "collision_clear", "timestamp": time.time()})

    async def _emergency_stop(self):
        # NOTE: 'stop' as a bound method does not exist on the twin object
        # (confirmed during testing -- AttributeError). The verified path
        # is publish_command, matching the 'stop' entry in the twin's own
        # MQTT command spec. Run help(t.publish_command) once on hardware
        # to confirm the exact argument shape before trusting this in a
        # real near-collision -- this is safety-critical, don't assume.
        try:
            self.twin.publish_command("stop", {})
        except Exception as exc:  # noqa: BLE001
            logger.error("EMERGENCY STOP FAILED TO SEND: %s -- robot may keep moving!", exc)
            return

        if self.on_event:
            self.on_event({"type": "collision_alert", "timestamp": time.time()})