#!/usr/bin/env python3
"""Print the Colab URL that opens a notebook straight from GitHub.

Run: python scripts/colab_link.py [notebook]

Defaults to notebooks/02_stage1_finetune.ipynb. No runtime, no upload, no
token: Colab fetches the notebook from GitHub when you open the link.

Before printing the link it pre-flights the things that make a notebook
open blank or stale in Colab:
  - the notebook is valid JSON and every cell has non-empty source
  - every cell has an `id` (required by nbformat 4.5; without it strict
    readers can drop cell content)
  - the file is committed, and the commit is pushed to origin/<branch>
    (Colab reads the pushed branch, not your working tree)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = "notebooks/02_stage1_finetune.ipynb"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def repo_slug() -> str:
    url = git("remote", "get-url", "origin")
    m = re.search(r"github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"origin is not a GitHub remote: {url}")
    return m.group("slug")


def check_notebook(path: Path) -> list[str]:
    errors = []
    try:
        nb = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path}: not valid JSON ({e}) — Colab will refuse to open it"]

    cells = nb.get("cells")
    if not cells:
        return [f"{path}: no cells"]

    minor = nb.get("nbformat_minor", 0)
    for i, cell in enumerate(cells):
        source = cell.get("source") or []
        if not "".join(source).strip():
            errors.append(f"{path}: cell {i} ({cell.get('cell_type')}) has empty source")
        if minor >= 5 and "id" not in cell:
            errors.append(f"{path}: cell {i} has no 'id' (required by nbformat 4.{minor})")
    return errors


def check_pushed(rel: str, branch: str) -> list[str]:
    if git("status", "--porcelain", "--", rel):
        return [f"{rel} has uncommitted changes — Colab opens the pushed copy, not yours"]

    remote_ref = f"origin/{branch}"
    try:
        git("rev-parse", "--verify", "--quiet", remote_ref)
    except subprocess.CalledProcessError:
        return [f"{remote_ref} does not exist — push the branch first"]

    local_blob = git("rev-parse", f"HEAD:{rel}")
    try:
        remote_blob = git("rev-parse", f"{remote_ref}:{rel}")
    except subprocess.CalledProcessError:
        return [f"{rel} is not on {remote_ref} — push the branch first"]

    if local_blob != remote_blob:
        return [f"{rel} differs from {remote_ref} — push before opening in Colab"]
    return []


def main() -> int:
    rel = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f"no such notebook: {rel}")

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    problems = check_notebook(path) + check_pushed(rel, branch)

    if problems:
        print(f"FAILED: {len(problems)} issue(s)\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"OK: {rel} is valid and matches origin/{branch}\n")
    print(f"https://colab.research.google.com/github/{repo_slug()}/blob/{branch}/{rel}")
    print("\nOpen it, then File -> Save a copy in Drive to get an editable copy")
    print("that survives runtime shutdowns (MyDrive/Colab Notebooks/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
