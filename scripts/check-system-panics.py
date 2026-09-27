#!/usr/bin/env python3
"""Check system panic containment across real cdylib boundaries (macOS/Linux).

Run: python3 scripts/check-system-panics.py. No GPU required.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    scripts = Path(__file__).resolve().parent
    probe = runpy.run_path(str(scripts / "check-io-reload.py"))
    probe["main"](scripts.parent / "ecs/tests/fixtures/system_panics")
