"""
teleop.py -- gamepad bridge for QUARRY, laptop-side.

Cyberwave's dashboard "Manual Control" panel is joint-slider based
(individual wheel joint radians), not a gamepad/keyboard drive
interface -- confirmed by checking the actual UI. So this script
rebuilds the old app.py gamepad path, but routed through the
Cyberwave SDK's move_forward/turn_left/etc instead of talking to the
ESP32 directly, matching the split-compute architecture (laptop makes
driving decisions, Pi only executes + runs the safety watchdog).

RUN THIS ON THE LAPTOP, WITH THE GAMEPAD'S USB DONGLE PLUGGED INTO
THE LAPTOP -- not the Pi.

FIRST RUN: use --calibrate to find your controller's actual axis and
button numbers before trusting the drive mapping below. Every
generic 2.4G gamepad numbers its axes/buttons slightly differently;
guessing wrong here means unexpected movement on a real robot.

    python teleop.py --calibrate

Wiggle the left stick and press a few buttons, watch which axis/button
indices change, then edit AXIS_FORWARD / AXIS_TURN / DEADMAN_BUTTON
below to match what you saw before running normally:

    python teleop.py

SAFETY: movement only sends while DEADMAN_BUTTON is held down. Release
it (or let the stick return to center) and a stop() is sent within one
tick. This is on top of, not instead of, the Pi's own 0.5s movement
watchdog -- two independent layers, which is the point.
"""

import argparse
import os
import time

import pygame
from dotenv import load_dotenv

import cyberwave
from backend.vision.detector import YoloeDetector

load_dotenv()

ENVIRONMENT_ID = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
ASSET_KEY = "waveshare/ugv-beast"
TWIN_NAME = "QUARRY"

# --- EDIT THESE AFTER RUNNING --calibrate ---------------------------------
AXIS_FORWARD = 1        # left stick vertical -- confirmed: pushed up is negative
AXIS_TURN = 0            # left stick horizontal
DEADMAN_BUTTON = 4       # L1 -- must be held to allow any drive movement
# ---------------------------------------------------------------------------

# --- extra controls, confirmed mapping for this Xbox 360-reporting pad -----
AXIS_CAM_PAN = 2         # right stick horizontal -- camera left/right
AXIS_CAM_TILT = 3        # right stick vertical -- camera up/down
BUTTON_A = 0              # toggle camera light
BUTTON_B = 1              # manual emergency stop (in addition to releasing L1)
BUTTON_X = 2              # take photo
BUTTON_Y = 3              # toggle chassis lights
AXIS_R2 = 5               # right trigger, resting -1.0, pressed +1.0 -- speed boost
CAM_DEADZONE = 0.35
SPEED_BOOST_DISTANCE = 0.5   # move distance when R2 held, instead of the normal 0.3
NORMAL_DISTANCE = 0.3


DEADZONE = 0.25
TICK_HZ = 10             # how often we sample the stick and send a command
BURST_DURATION = 1.0 / TICK_HZ + 0.1  # slightly longer than tick so bursts overlap, no gaps

# Proximity-based forward-block. Same bbox-area-as-distance-proxy approach as
# CollisionGuard, and critically the SAME debounce pattern -- a single clear
# frame does NOT re-arm forward movement. A naive per-tick check (capture one
# frame, block or don't, based on that frame alone) lets exactly one flaky
# clear read slip you straight back into an obstacle that's still there --
# confirmed as a real live failure mode, not a hypothetical. Sampled every
# tick regardless of drive intent, same as CollisionGuard's independent loop,
# so the block state is never stale by the time you actually press forward.
PROXIMITY_AREA_THRESHOLD = 0.30
CLEAR_STREAK_REQUIRED = 4


def calibrate():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Is the dongle plugged into THIS machine?")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Found: {js.get_name()}  ({js.get_numaxes()} axes, {js.get_numbuttons()} buttons)")
    print("Wiggle sticks and press buttons. Ctrl+C to stop.\n")

    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            buttons = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            print(f"\raxes={axes}  buttons_held={buttons}    ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone calibrating.")


CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "captures")


def _save_photo(twin):
    """Grab the current frame directly (same call real_feed.py uses for
    detection) and save it to disk -- more reliable than the 'take_photo'
    MQTT command, which has no confirmed storage destination configured
    on the driver/platform side (checked the dashboard's Captures tab --
    empty, that tab is tied to Workflow missions, not raw teleop calls)."""
    try:
        frame = twin.get_latest_frame()
    except Exception as exc:  # noqa: BLE001
        print(f"[photo] capture failed: {exc}")
        return
    if frame is None:
        print("[photo] no frame available yet")
        return

    os.makedirs(CAPTURES_DIR, exist_ok=True)
    filename = f"quarry_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURES_DIR, filename)
    try:
        import cv2
        import numpy as np

        arr = np.array(frame) if hasattr(frame, "convert") else frame  # normalize PIL if needed
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print(f"[photo] saved: {os.path.abspath(path)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[photo] save failed: {exc}")


def run():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected -- check the dongle is in this laptop's USB port.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Gamepad connected: {js.get_name()}")

    cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
    twin = cyberwave.twin(ASSET_KEY, environment=ENVIRONMENT_ID, name=TWIN_NAME)
    print("Connected to QUARRY twin. Hold L1 to drive. A=light  B=stop  X=photo  Y=chassis lights")

    was_moving = False
    # track previous button state so single-press actions (photo, toggles,
    # manual stop) fire once per press, not once per tick for as long as
    # you hold them down
    prev_buttons = {BUTTON_A: False, BUTTON_B: False, BUTTON_X: False, BUTTON_Y: False}

    detector = YoloeDetector()
    guard_blocked = False       # persistent -- survives across ticks, not re-derived fresh each check
    guard_clear_streak = 0

    try:
        while True:
            pygame.event.pump()

            # -- proximity guard: sampled every tick, independent of drive
            # intent, so the block state is never stale by the time forward
            # is actually attempted -----------------------------------------
            try:
                frame = twin.get_frame('numpy', source='remote_edge')
                frame = detector.normalize_frame(frame)
            except Exception:  # noqa: BLE001 -- a bad read must not crash driving
                frame = None

            if frame is not None and detector.available:
                detections = detector.detect(frame)
                nearest_area = max(
                    ((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) for d in detections),
                    default=0.0,
                )
                if nearest_area >= PROXIMITY_AREA_THRESHOLD:
                    guard_clear_streak = 0
                    if not guard_blocked:
                        guard_blocked = True
                        print("[collision guard] BLOCKED -- obstacle too close, forward refused")
                else:
                    guard_clear_streak += 1
                    if guard_blocked and guard_clear_streak >= CLEAR_STREAK_REQUIRED:
                        guard_blocked = False
                        print("[collision guard] path clear, forward re-armed")

            # -- one-shot buttons: fire only on the press edge --------------
            for btn, label, command in (
                (BUTTON_A, "camera light toggle", "camera_light_toggle"),
                (BUTTON_Y, "chassis lights toggle", "chassis_light_toggle"),
            ):
                held = js.get_button(btn) == 1
                if held and not prev_buttons[btn]:
                    print(f"[{label}]")
                    try:
                        twin.publish_command(command, {})
                    except Exception as exc:  # noqa: BLE001
                        print(f"  command failed: {exc}")
                prev_buttons[btn] = held

            x_held = js.get_button(BUTTON_X) == 1
            if x_held and not prev_buttons[BUTTON_X]:
                _save_photo(twin)
            prev_buttons[BUTTON_X] = x_held

            b_held = js.get_button(BUTTON_B) == 1
            if b_held and not prev_buttons[BUTTON_B]:
                print("[manual stop]")
                try:
                    twin.publish_command("stop", {})
                except Exception as exc:  # noqa: BLE001
                    print(f"  stop failed: {exc}")
                was_moving = False
            prev_buttons[BUTTON_B] = b_held

            # -- right stick: camera pan/tilt, no deadman needed (not a motion hazard) --
            pan = js.get_axis(AXIS_CAM_PAN)
            tilt = -js.get_axis(AXIS_CAM_TILT)
            if abs(pan) >= CAM_DEADZONE:
                try:
                    twin.publish_command("camera_right" if pan > 0 else "camera_left", {})
                except Exception as exc:  # noqa: BLE001
                    print(f"  camera pan failed: {exc}")
            elif abs(tilt) >= CAM_DEADZONE:
                try:
                    twin.publish_command("camera_up" if tilt > 0 else "camera_down", {})
                except Exception as exc:  # noqa: BLE001
                    print(f"  camera tilt failed: {exc}")

            # -- left stick + L1: drive ---------------------------------------
            deadman_held = js.get_button(DEADMAN_BUTTON) == 1
            fwd = -js.get_axis(AXIS_FORWARD)
            turn = js.get_axis(AXIS_TURN)

            if not deadman_held or (abs(fwd) < DEADZONE and abs(turn) < DEADZONE):
                if was_moving:
                    twin.publish_command("stop", {})
                    was_moving = False
                time.sleep(1.0 / TICK_HZ)
                continue

            was_moving = True
            distance = SPEED_BOOST_DISTANCE if js.get_axis(AXIS_R2) > 0 else NORMAL_DISTANCE
            if abs(fwd) >= abs(turn):
                if fwd > 0:
                    if guard_blocked:
                        print("[collision guard] forward refused -- obstacle still too close")
                        twin.publish_command("stop", {})
                    else:
                        twin.move_forward(distance=distance, duration=BURST_DURATION)
                else:
                    twin.move_backward(distance=distance, duration=BURST_DURATION)  # backing away always allowed
            else:
                if turn > 0:
                    twin.turn_right(0.3)
                else:
                    twin.turn_left(0.3)

            time.sleep(1.0 / TICK_HZ)

    except KeyboardInterrupt:
        print("\nStopping...")
        try:
            twin.publish_command("stop", {})
        except Exception:  # noqa: BLE001 -- best-effort stop on exit
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="print raw axis/button values, don't drive")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    else:
        run()