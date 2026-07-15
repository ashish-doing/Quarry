"""
voice_control.py -- push-to-talk voice driving for QUARRY, laptop-side.

Uses Groq's Whisper API for transcription -- picked over offline Vosk
(no local model download/setup needed) and over Google's Web Speech API
(you mentioned Groq access already works). Hold SPACE, speak a command,
release SPACE -- audio gets transcribed and matched against a small
fixed vocabulary, then sent through the same SDK calls teleop.py uses.

VERIFY BEFORE TRUSTING: the exact Groq audio-transcription call shape
below (`groq_client.audio.transcriptions.create(...)`) is based on
Groq's documented OpenAI-compatible API surface, not something tested
live in this session -- run a single quick sanity check first (see the
bottom of this docstring) before wiring it into actual driving, the
same way every other unverified API call in this project got confirmed
before being trusted for real movement.

Recognized commands (case-insensitive, simple keyword match):
    "forward" / "go"                -> move_forward
    "back" / "backward" / "reverse" -> move_backward
    "left"                          -> turn_left
    "right"                         -> turn_right
    "stop" / "halt"                 -> stop

Requirements:
    pip install groq sounddevice soundfile keyboard numpy
    Set GROQ_API_KEY in your .env

SAFETY: like teleop.py, this drives your OWN robot only, run locally,
never routed through the multi-site Hub -- same trust boundary as
every other control path in this project. Run only one of
patrol.py / teleop.py / voice_control.py at a time.

Quick sanity check before your first real drive session:
    python -c "
    from groq import Groq
    import os
    from dotenv import load_dotenv
    load_dotenv()
    print(Groq(api_key=os.environ['GROQ_API_KEY']).models.list())
    "
This just confirms your API key and the client import work, before
trusting the transcription call live.
"""

import io
import os

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq

import cyberwave

load_dotenv()

ENVIRONMENT_ID = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
ASSET_KEY = "waveshare/ugv-beast"
TWIN_NAME = "QUARRY"
SAMPLE_RATE = 16000

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def record_while_held(key="space"):
    print("Listening... (hold SPACE, release when done)")
    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    stream.start()
    try:
        while keyboard.is_pressed(key):
            data, _ = stream.read(1024)
            frames.append(data.copy())
    finally:
        stream.stop()
        stream.close()

    if not frames:
        return None
    audio = np.concatenate(frames, axis=0)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    buf.name = "command.wav"
    return buf


def transcribe(audio_buf) -> str:
    result = groq_client.audio.transcriptions.create(
        file=audio_buf, model="whisper-large-v3", language="en",
    )
    return result.text.strip().lower()


def execute(text: str, twin):
    print(f"Heard: '{text}'")
    if "stop" in text or "halt" in text:
        twin.publish_command("stop", {})
    elif "forward" in text or text.strip() == "go":
        twin.move_forward(distance=0.3, duration=1.5)
    elif "back" in text or "reverse" in text:
        twin.move_backward(distance=0.3, duration=1.5)
    elif "left" in text:
        twin.turn_left(0.5)
    elif "right" in text:
        twin.turn_right(0.5)
    else:
        print("  (not a recognized command, ignored)")


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set -- add it to your .env first.")

    cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
    twin = cyberwave.twin(ASSET_KEY, environment=ENVIRONMENT_ID, name=TWIN_NAME)
    print("Voice control ready. Hold SPACE to speak a command, release to send.")
    print("Commands: forward, back, left, right, stop. Ctrl+C to quit.")

    try:
        while True:
            keyboard.wait("space")
            audio = record_while_held("space")
            if audio is None:
                continue
            try:
                text = transcribe(audio)
                execute(text, twin)
            except Exception as exc:  # noqa: BLE001 -- one bad transcription shouldn't kill the session
                print(f"Transcription/command failed: {exc}")
    except KeyboardInterrupt:
        print("\nExiting voice control.")
        try:
            twin.publish_command("stop", {})
        except Exception:  # noqa: BLE001 -- best-effort stop on exit
            pass


if __name__ == "__main__":
    main()