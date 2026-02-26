#!/usr/bin/env python3
"""Quick local checklist for deployment/runtime essentials."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

required_files = [
    "Dockerfile",
    "render.yaml",
    "Procfile",
    "requirements.txt",
    "app.py",
    "static/admin.html",
    "static/chat-widget.html",
]

print("[check] required files")
missing = []
for rel in required_files:
    ok = (ROOT / rel).exists()
    print(f" - {rel}: {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(rel)

render_yaml = (ROOT / "render.yaml").read_text(encoding="utf-8")
dockerfile_path = re.search(r"dockerfilePath:\s*(.+)", render_yaml)
docker_context = re.search(r"dockerContext:\s*(.+)", render_yaml)

path_val = dockerfile_path.group(1).strip().strip("\"'") if dockerfile_path else "./Dockerfile"
ctx_val = docker_context.group(1).strip().strip("\"'") if docker_context else "."

print("[check] render.yaml")
print(f" - dockerfilePath: {path_val} -> {'OK' if (ROOT / path_val).exists() else 'MISSING'}")
print(f" - dockerContext: {ctx_val} -> {'OK' if (ROOT / ctx_val).exists() else 'MISSING'}")

requirements = {
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
}

print("[check] requirements")
for pkg in ("flask", "requests", "gunicorn"):
    print(f" - {pkg}: {'OK' if pkg in requirements else 'MISSING'}")

if missing:
    raise SystemExit(1)

print("[check] done")
