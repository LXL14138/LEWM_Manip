"""Offline check that the running interpreter satisfies every pin in requirements.txt (importlib.metadata only).

    python tools/check_requirements.py [requirements.txt]

Exit 0 when every pinned distribution is installed at exactly the pinned version (a local version suffix such as
torch 2.14.0+cu130 counts as satisfying 2.14.0), 1 otherwise. Used instead of `pip install -r` / `pip check` where pip
is not available.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "requirements.txt")
bad = []
rows = []
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    name, pinned = line.split("==")
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = None
    ok = installed is not None and (installed == pinned or installed.split("+", 1)[0] == pinned)
    rows.append((name, pinned, installed, ok))
    if not ok:
        bad.append(name)
width = max(len(r[0]) for r in rows)
for name, pinned, installed, ok in rows:
    print(f"{'ok ' if ok else 'BAD'} {name:<{width}}  required {pinned:<12} installed {installed}")
print(f"python {sys.version.split()[0]}: {len(rows) - len(bad)} / {len(rows)} pins satisfied" + (f"; unsatisfied: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
