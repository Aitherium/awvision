# awvision for agents

Read this if you are an agent (or a human) editing this package. Short on
purpose: the commands, the traps that cost a session, and where the rest lives.
Nothing here is read at runtime — it is for you.

## What this is

PyPI distribution **`awvision`** (version in `pyproject.toml`), import package
`awvision`, Python >= 3.10. See an image — describe it, ask it a question,
compare two. Pixels in, text out, through a vision model.

This repository is a **synced mirror** of the AitherOS monorepo (lane
`.github/workflows/sync-awvision.yml`). Hand edits made here are overwritten on
the next sync — change the source and let the lane publish.

## Build, test, verify

```bash
python -m pytest tests -q        # the suite: 44 tests, green at v0.3.2
pip install -e .                 # editable install for developing against it
```

The suite was run from a source checkout with no prior install. The publish
lane (`publish-brick.yml`) additionally builds the wheel, installs it and
imports it — a tree that tests green can still ship a broken wheel.

## Rules that keep this useful

- **Resolution and sight are separate steps, and both are tested.**
  `test_resolve.py` pins how a model/endpoint is resolved before any image is
  sent; `test_sight.py` pins the seeing itself. A change that merges them
  makes "which model answered" unanswerable — keep them separable.
- **The question shapes the answer.** Describe / ask / compare are one client
  with three verbs; never widen a describe into an unbounded caption dump that
  costs the caller a page of tokens for one fact.
- **Never present a partial read as the whole image.** A truncated or resized
  input is a fact the caller needs; an answer that silently ignored half the
  pixels is the quiet wrong this family keeps paying for.
- **The registry drives the public surface.** This repo's README header,
  `llms.txt` and `aither-manifest.json` are generated from the ecosystem
  registry (one yaml in the AitherOS monorepo) and rewritten on every sync.
  Change the registry; do not hand-edit the generated blocks.

## Read next

- `llms.txt` — the install/use card written for an agent to execute
- `README.md` — the human front door
- `docs/` — the generated docs site source
