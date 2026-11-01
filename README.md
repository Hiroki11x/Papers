# Papers: Reading Notes Archive


## Overview

This repository is a lightweight archive of paper reading notes, focused mainly on machine learning and optimization.

- Notes are written in **GitHub Issues**
- A GitHub Actions workflow exports the issues to static JSON under `data/`
- The website reads that JSON (no GitHub API calls from the browser, so no rate limits)
- GitHub Pages provides a simple public index with search, label filters, sorting and pagination

The goal is clarity and continuity: keep notes easy to add, easy to read, and easy to maintain.

## Repository structure

```text
.
├── index.html                # GitHub Pages entry point
├── assets/
│   ├── css/style.css         # Site styling
│   └── js/main.js            # Client-side rendering (search / filter / sort / detail)
├── data/                     # Static issue export (generated, do not edit by hand)
│   ├── index.json            # List metadata + preview for every open issue
│   ├── issues.json           # Same with full bodies (full-text search)
│   └── issues/<n>.json       # One file per issue (detail view)
├── scripts/
│   └── build_data.py         # Exports issues -> data/ (run by GitHub Actions)
├── docs/                     # Japanese surveys rebuilt from all issues
│   ├── README.md             # Index and cross-topic timeline
│   ├── 00_catalog.md         # Catalog of all notes (topic / venue / year)
│   ├── practical_optimization/   # Critical batch size, low precision + Muon, semi-synchronous training
│   └── misc/                 # 12 other topics (OOD, calibration, loss landscape, generalization, ...)
├── notes/
│   └── templates/
│       └── paper-note.md     # Recommended note format
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── paper-note.yml    # Optional issue form for consistent notes
│   └── workflows/
│       └── build-data.yml    # Rebuilds data/ on issue changes (and daily)
└── README.md
```

## How notes are organized

- **One paper = one GitHub Issue**
- Issue title should be the paper title
- Labels are used as topic tags (e.g., `optimization`, `generalization`, `sgd`)
- Notes are listed newest-first by default; sort by oldest, recently updated, or title A-Z / Z-A
- Full-text search (title, labels, body), multi-label AND filter, page-number navigation
- Only **open** issues are published; closing an issue removes it from the site

## How to add a new note

1. Open a new issue in this repository.
2. Use the **Paper Note** issue template (or `notes/templates/paper-note.md`).
3. Fill in the core sections: citation, link, date read, tags, summary, key ideas, comments.
4. Add relevant labels for discoverability.
5. Submit the issue — the `build-data` workflow regenerates `data/` and the note appears on the site within a minute or two.

To regenerate the data locally (requires the GitHub CLI):

```bash
python3 scripts/build_data.py
```

## Publishing / GitHub Pages

- The site is fully static (`index.html` + CSS + JS).
- On load, it fetches issue data from this repository through the GitHub API.
- No build step is required.
- Publish via GitHub Pages from the repository root.

## Design principles

- Keep the workflow issue-first and manual-friendly.
- Prefer simple structure over additional tooling.
- Keep presentation clean, readable, and technical.
- Avoid feature creep (no heavy framework, no complex backend).

## Future improvements (small, optional)

- Add a short “start here” note collection for recommended foundational papers.
- Add a small label naming guide if label vocabulary expands.
