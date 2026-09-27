#!/usr/bin/env python3
"""Check IO across real cdylib reloads on macOS/Linux, without a GPU.

Build both images against identical Cargo artifacts, then check guest Tokio
context, poll/drop panics, worker TLS and retained wakers after dlclose.
Run from any directory: python3 scripts/check-io-reload.py
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "ecs/tests/fixtures/io_reload"


def main(fixtures=FIXTURES):
    if sys.platform not in ("darwin", "linux"):
        raise SystemExit("This dlopen probe requires macOS or Linux")
    build = subprocess.run(
        ["cargo", "build", "-p", "redlilium-ecs", "--message-format=json"],
        cwd=ROOT, stdout=subprocess.PIPE, text=True, check=True,
    )
    artifacts = [json.loads(line) for line in build.stdout.splitlines()]

    def rlib_for(name):
        return next(
            Path(path)
            for item in artifacts
            if item.get("reason") == "compiler-artifact" and item["target"]["name"] == name
            for path in item["filenames"] if path.endswith(".rlib")
        )

    ecs = rlib_for("redlilium_ecs")
    tokio = rlib_for("tokio")
    dependencies = tokio.parent
    base = [
        "rustc", "--edition=2024", "--extern", f"redlilium_ecs={ecs}",
        "--extern", f"tokio={tokio}", "-L", f"dependency={dependencies}",
    ]
    for output in (dependencies.parent / "build").glob("*/out"):
        base += ["-L", f"native={output}"]
    if sys.platform == "linux":
        base += ["-l", "dl"]
    with tempfile.TemporaryDirectory(prefix="vibe-io-reload-") as directory:
        temporary = Path(directory)
        guest = temporary / ("guest.dylib" if sys.platform == "darwin" else "guest.so")
        host = temporary / "host"
        subprocess.run(
            base + ["--crate-type=cdylib", str(fixtures / "guest.rs"), "-o", str(guest)],
            check=True, timeout=60,
        )
        subprocess.run(
            base + [str(fixtures / "host.rs"), "-o", str(host)], check=True, timeout=60,
        )
        subprocess.run([str(host), str(guest)], check=True, timeout=30)


if __name__ == "__main__":
    main()
