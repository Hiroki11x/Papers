# Papers: Reading Notes Archive

## Overview

This repository is a lightweight archive of paper reading notes, focused mainly on machine learning and optimization.

- Notes are written in **GitHub Issues**
- The website reads those issues through the GitHub API
- GitHub Pages provides a simple public index for browsing

The goal is clarity and continuity: keep notes easy to add, easy to read, and easy to maintain.

## Repository structure

```text
.
├── index.html                # GitHub Pages entry point
├── assets/
│   ├── css/style.css         # Site styling
│   └── js/main.js            # Client-side rendering (issues list/detail)
├── notes/
│   └── templates/
│       └── paper-note.md     # Recommended note format
├── .github/
│   └── ISSUE_TEMPLATE/
│       └── paper-note.yml    # Optional issue form for consistent notes
└── README.md
```

## How notes are organized

- **One paper = one GitHub Issue**
- Issue title should be the paper title
- Labels are used as topic tags (e.g., `optimization`, `generalization`, `sgd`)
- Notes are listed in reverse chronological order by issue creation date
- The site supports browsing by label filters and pagination

## How to add a new note

1. Open a new issue in this repository.
2. Use the **Paper Note** issue template (or `notes/templates/paper-note.md`).
3. Fill in the core sections: citation, link, date read, tags, summary, key ideas, comments.
4. Add relevant labels for discoverability.
5. Submit the issue — it will automatically appear on the GitHub Pages site.

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
