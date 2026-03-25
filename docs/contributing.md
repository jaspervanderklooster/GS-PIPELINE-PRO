# Contributing to GS_PIPELINE

Thank you for helping make GS_PIPELINE stable, human-friendly and trustworthy.

This file explains how humans and AI (vibe-coding) should contribute code, tests and documentation in a way that preserves the project’s core principle: **the technology should adapt to the human**. Stability and output quality are the top priority.

---

## Quickstarts — local dev

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
````

2. Run unit tests:

```bash
pytest -q
```

3. Run the baseline end-to-end test (the repo must contain a `data/sample_dataset` and `expected/`):

```bash
# Example: simple helper (project may provide this)
# make run-sample
python tests/run_sample_end2end.py   # or run the script the repo provides
```

4. Docker (optional, recommended for consistent environments):

```bash
docker build -t gs_pipeline:dev .
docker run --rm -v $(pwd)/data:/workspace/data gs_pipeline:dev make run-sample
```

---

## Baseline & Golden Dataset (mandatory)

Before any non-trivial change, the repository must have:

* A tagged **working baseline** (e.g. `v0.1-working`).
* A small **golden dataset** in `data/sample_dataset/` that reproduces success on that tag.
* An `expected/` folder with checksums or expected artifacts (e.g. fused.ply sha256).

Any PR that affects pipeline behavior must run against the baseline dataset and explain the result.

---

## Branches & PRs

* Create a feature branch: `feature/<short-descriptive-name>` or `fix/<short>`.
* One feature / fix per PR. Keep changes small and focused.
* Always include:

  * Plain-language description of *why* the change is needed.
  * How to test (step-by-step for non-devs).
  * Acceptance criteria (visual + metrics when relevant).
  * Note which preset(s) are affected.

**PR naming**: short, descriptive (e.g. `worker: add colmap fallback`).

---

## PR template (copy into `.github/PULL_REQUEST_TEMPLATE.md`)

```
### Summary
Short summary in plain English.

### Motivation
Why this change? What user problem is solved?

### What changed
- Bullet list of files and behaviour changes.

### How to test (non-technical steps)
1. Checkout branch
2. Activate `.venv` and run:
   - `pytest -q`
   - `python tests/run_sample_end2end.py --dataset data/sample_dataset`
3. Expected results:
   - `expected/fused.ply` checksum matches (or explain acceptable tolerance).
   - `job.json` contains `meta.colmap_preset` and `preset_used` when relevant.

### Acceptance criteria
- [ ] Unit tests pass
- [ ] End-to-end baseline run OK on `data/sample_dataset`
- [ ] CI green
- [ ] No secrets added
- [ ] Docs updated (README/OPERATIONS or CHANGELOG if applicable)

### Notes for reviewers
Any special things to check or known caveats.
```

---

## AI (vibe-coding) Guidelines — short & practical

We treat AI as a powerful assistant, not an oracle. Use the AI to generate code, tests, or docs — but follow these rules:

### Before asking the AI

* Pin the **baseline commit**, **sample dataset**, and **expected outputs** in the prompt.
* State the exact version of tools (COLMAP, CUDA/driver, Python).
* State the acceptance criteria and which preset the change affects.

### One task per prompt

Ask AI to do **one** thing at a time:

* “Write a headless dedupe script that takes `--input-dir` and outputs `analysis.json` and `summary.txt`.”
* “Write unit tests for `_ssim_gray` and include boundary cases.”

### Always request tests

Every substantive AI-generated code change must include:

* Unit tests (pytest format).
* A short end-to-end smoke test instruction (how to run on the sample dataset).

### Prompt Template (use this)

```
Context:
- Repo: GS_PIPELINE
- Baseline tag: v0.1-working
- Sample dataset: data/sample_dataset/
- Tooling: Python 3.10, COLMAP vX.Y.Z (or COLMAP_BIN env var)
Task:
- Implement <concise task description>
Constraints:
- Follow repo style and file layout
- Add unit tests under tests/
- Provide a 3-line plain-English description for reviewers explaining what changed and why
- Do not touch unrelated files
```

### After AI produces code

* Review the generated code line-by-line.
* Run unit tests and the baseline E2E test.
* Make sure the AI includes docstrings, a plain-language explanation, and a test plan.
* Check for secrets or unsafe assumptions.
* If you accept the patch, create a PR on a feature branch and include “AI-generated” in the PR body.

---

## Human review checklist (for reviewers)

When reviewing a PR (human or AI-generated), verify:

* [ ] The PR is small and focused.
* [ ] Unit tests exist and pass locally.
* [ ] The baseline end-to-end test passes: no regressions on `data/sample_dataset`.
* [ ] No unpinned dependencies. If adding a new dependency, add an entry in `requirements.txt` with a pinned version and add a rationale.
* [ ] No secrets, no hard-coded paths or credentials.
* [ ] CI definition updated if needed (GitHub Actions or other).
* [ ] The change has a plain-English explanation and a simple test plan.
* [ ] Logs are preserved and improved; do not remove existing meaningful logs.
* [ ] The change respects the “technology adapts to humans” principle (no forcing users to special input formats).

If in doubt, ask the original maintainer or open an RFC issue.

---

## Special rules: presets, COLMAP & driver upgrades

Upgrading COLMAP, CUDA, or GPU drivers is *risky*. Follow this checklist:

1. **Create a new branch**: `upgrade/colmap-<version>`.
2. **Snapshot baseline**: tag `v0.1-working` (if not already).
3. **Update Dockerfile and lockfiles** (requirements/conda/poetry).
4. **Run the baseline E2E** against `data/sample_dataset`.
5. **Compare metrics**: reprojection error, registered images, fused.ply checksum or similarity metric.
6. **If results degrade**: do a bisect-style test (old/new colmap) on the same dataset and document findings.
7. **Only merge once** the team agrees and the baseline dataset reproduces expected quality.

If a requested run is heavy, prefer a safe, documented fallback (e.g. `standard_safe`, `hq_safe`) rather than a hard failure.

---

## Tests & CI

* Unit tests: `pytest -q`
* Integration: `tests/run_sample_end2end.py` (or `make run-sample`)
* CI must run:

  * Lint (optional)
  * Unit tests
  * End-to-end on the golden dataset OR a smoke test that verifies artifacts were written and `job.json` updated.
* For GPU-bound tests, use self-hosted runners with the required drivers. CI must not silently skip integration tests — these will be marked and run on the correct runner.

---

## Dependency & versioning policy

* Pin everything in `requirements.txt` or use `poetry.lock`. No floating dependencies for core pipeline packages.
* Docker images must be tagged (`gs_pipeline:colmap-<version>-py3.11`) and Dockerfiles checked into the repo.
* Any third-party tool upgrades require a documented upgrade process (see above).

---

## Security and secrets

* Never commit private keys, tokens, or passwords.
* Use environment variables or secret management for runtime credentials.
* Run a quick scan for secrets on any PR that changes configs or CI.

---

## When to open an RFC or Issue

Open an RFC / Issue before making large changes, especially for:

* API changes to `job.json`
* Preset semantics changes
* Major rearchitectures (e.g. move from file-based job management to a DB)
* Upgrading COLMAP / CUDA / drivers
* Anything that could affect many past jobs or users

Label RFCs clearly: `RFC: <short title>`.

---

## Escalation & ownership

If a change is risky or unclear, assign it to a maintainer and add the `manual_attention` label. If you are the author and are uncertain, open a short RFC and ask for a second reviewer.

---

## Final note

This project’s goal is clear: **make reliable, high-quality Gaussian Splats from human-provided media**. Every change should support that goal by improving reliability, reproducibility, or the human experience of the pipeline.

Happy contributing — and when in doubt: **choose stability and clarity** over cleverness.
