#!/usr/bin/env python3
"""Export GitHub Issues of this repository to static JSON for the Pages site.

The site (assets/js/main.js) reads these files instead of calling the GitHub
API from the browser, so visitors never hit the unauthenticated rate limit.

Outputs (all minified JSON):
  data/index.json   - lightweight list: number, title, labels, dates, preview
  data/issues.json  - same plus full markdown body (loaded lazily for
                      full-text search)
  data/issues/N.json - one file per issue (loaded for the detail view)

Usage:
  python3 scripts/build_data.py            # uses `gh api` (auth via gh / GH_TOKEN)
  python3 scripts/build_data.py --state all

Requires the GitHub CLI (`gh`) on PATH.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

OWNER = "Hiroki11x"
REPO = "Papers"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREVIEW_LEN = 160


def fetch_issues(state: str) -> list[dict]:
    cmd = [
        "gh", "api", "--paginate",
        f"repos/{OWNER}/{REPO}/issues?state={state}&per_page=100&sort=created&direction=desc",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    # --paginate concatenates one JSON array per page; split them robustly.
    issues: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(out):
        while idx < len(out) and out[idx].isspace():
            idx += 1
        if idx >= len(out):
            break
        page, end = decoder.raw_decode(out, idx)
        issues.extend(page)
        idx = end
    # The issues endpoint also returns pull requests; drop them.
    return [i for i in issues if not i.get("pull_request")]


def make_preview(body: str | None) -> str:
    if not body:
        return ""
    text = body
    text = re.sub(r"```[\s\S]*?```", " ", text)            # code blocks
    text = re.sub(r"<[^>]+>", " ", text)                   # raw html tags
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)      # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # links -> text
    text = re.sub(r"https?://\S+", " ", text)              # bare urls
    text = re.sub(r"[#>*_`|]", " ", text)                  # md punctuation
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > PREVIEW_LEN:
        text = text[:PREVIEW_LEN].rstrip() + "…"
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="open", choices=["open", "closed", "all"])
    args = ap.parse_args()

    raw = fetch_issues(args.state)
    raw.sort(key=lambda i: i["number"], reverse=True)

    full: list[dict] = []
    index: list[dict] = []
    label_counts: dict[str, int] = {}
    label_colors: dict[str, str] = {}
    for i in raw:
        labels = sorted(l["name"] for l in i.get("labels", []))
        for l in i.get("labels", []):
            label_counts[l["name"]] = label_counts.get(l["name"], 0) + 1
            label_colors.setdefault(l["name"], l.get("color", ""))
        base = {
            "number": i["number"],
            "title": i["title"],
            "labels": labels,
            "state": i["state"],
            "created_at": i["created_at"],
            "updated_at": i["updated_at"],
            "html_url": i["html_url"],
        }
        index.append({**base, "preview": make_preview(i.get("body"))})
        full.append({**base, "body": i.get("body") or ""})

    # Deterministic timestamp (latest issue update) so re-runs without issue
    # changes produce byte-identical output and the workflow commits nothing.
    generated = max((i["updated_at"] for i in raw), default="")
    labels_meta = [
        {"name": n, "count": c, "color": label_colors.get(n, "")}
        for n, c in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    ]
    meta = {"generated_at": generated, "count": len(index), "labels": labels_meta,
            "repo": f"{OWNER}/{REPO}", "state": args.state}

    DATA_DIR.mkdir(exist_ok=True)
    sep = (",", ":")
    (DATA_DIR / "index.json").write_text(
        json.dumps({**meta, "issues": index}, ensure_ascii=False, separators=sep) + "\n",
        encoding="utf-8")
    (DATA_DIR / "issues.json").write_text(
        json.dumps({**meta, "issues": full}, ensure_ascii=False, separators=sep) + "\n",
        encoding="utf-8")
    per_issue_dir = DATA_DIR / "issues"
    per_issue_dir.mkdir(exist_ok=True)
    keep = set()
    for item in full:
        name = f"{item['number']}.json"
        keep.add(name)
        (per_issue_dir / name).write_text(
            json.dumps(item, ensure_ascii=False, separators=sep) + "\n", encoding="utf-8")
    # Remove files for issues that are no longer in the export (closed/deleted).
    for stale in per_issue_dir.glob("*.json"):
        if stale.name not in keep:
            stale.unlink()
    print(f"wrote {len(index)} issues ({args.state}) to {DATA_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
