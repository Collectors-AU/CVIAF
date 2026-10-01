# Deploying the CVIAF evidence dashboard

**Status: built and ready to deploy. Not deployed.** No hosting destination has been
chosen and nothing has been published.

The build is a single self-contained file plus one JSON bundle. There is no server, no
build step and no runtime dependency.

```
site/
  index.html          # the whole UI; bundle is inlined so it opens from file://
  data/bundle.json    # same data, versioned separately for review/diffing
  DEPLOY.md
```

## Review before publishing

1. `grep -riE "(/Users/|billasur|cviaf-analysis|BEGIN [A-Z ]*PRIVATE KEY)" site/` must return
   nothing. The bundle names repository-relative paths only.
2. Confirm the bundle commit matches the deck: `python scripts/build_site.py` regenerates
   `bundle.json` from the receipts at the checked-out commit.
3. Open `index.html` from `file://` and confirm the evidence explorer shows both linked rows
   and `local artifact` rows -- the large corpora are deliberately outside git.

## Hosting

Any static host works (GitHub Pages, an internal S3 bucket, `python -m http.server` for a
local demo). Because the page is inlined it also works from an air-gapped machine.

```bash
cd site && python3 -m http.server 8080   # local preview only
```

## Regenerating

```bash
CVIAF_REPO=<path-to-cviaf-worktree> python scripts/build_site.py
```

Commit: 5866b55f89f41e6f2c23f023f7b5e4d8201467f5 (task3-real-backbone)
