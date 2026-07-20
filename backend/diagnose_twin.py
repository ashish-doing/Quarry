"""
diagnose_twin.py -- ONE-TIME diagnostic. Run this, paste the output back,
then we wire the real digital-twin position-sync call. Not meant to be
kept in the repo long-term or imported anywhere.

Usage: python -m backend.diagnose_twin
"""

import asyncio
from dotenv import load_dotenv

from backend import real_feed

load_dotenv()


async def diagnose():
    real_feed.state.connect()
    twin = real_feed.state.twin

    print("=== dir(twin) ===")
    print([m for m in dir(twin) if not m.startswith("_")])

    print("\n=== twin.commands.get_supported_commands() ===")
    try:
        print(twin.commands.get_supported_commands())
    except Exception as exc:
        print(f"failed: {exc}")

    print("\n=== checking for twin/position-sync-sounding methods ===")
    candidates = ("update", "update_position", "set_position", "sync",
                  "publish_pose", "update_pose", "push_state", "set_pose")
    for name in candidates:
        exists = hasattr(twin, name)
        print(f"  {name:20s} -> {exists}")
        if exists:
            try:
                help(getattr(twin, name))
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(diagnose())