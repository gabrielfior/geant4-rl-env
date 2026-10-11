# Setup, sandbox and grading

## Setup (all local, no downloads at import time)
    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python geant4_pybind numpy pytest
    micromamba create -y -p ./.conda -c conda-forge geant4 python=3.12   # ships the Geant4 datasets (~2.8 GB)
    source scripts/env.sh        # points G4*DATA at ./.conda datasets; needed in every shell
    python scripts/make_references.py   # only to regenerate geant4rl/references.json

(pip's own dataset downloader pulls from CERN at ~200 KB/s; the conda-forge dataset packages avoid that.)

## Use
    python -m geant4rl.harness spec  --task proton_range --seed 3 --workdir work/x
    # agent writes work/x/solution.py
    python -m geant4rl.harness grade --task proton_range --seed 3 --workdir work/x
    pytest tests

## Sandbox (model-written code runs in Docker, never on the host)
`docker build -t geant4rl-sandbox docker` (linux/amd64 image: there is no linux-arm64 geant4_pybind wheel; OrbStack runs it via Rosetta).
`geant4rl/runner.py` runs `solution.py` with: no network, read-only root fs, non-root, all caps dropped, 4 GB / 1 CPU / 256 pids, only the task workdir (rw) and the Geant4 datasets (ro) mounted, no host home/tokens/answer key, and API keys are not passed in. Geant4 `abort()` becomes exit status 6 (no macOS crash reports). The agent loop and the grader both use it. Output printed by a script still goes to the model, but the script can only see its own workdir.

## Physics list (fixed)
Every prompt tells the agent to use `FTFP_BERT` (Geant4 11.4, default cuts). Agents must not choose the list: tolerances (2–5%) cover statistics, not list-to-list EM differences. The hidden oracle for `alpha_range` and `nai_absorbed_fraction` uses `FTFP_BERT_EMZ` (EM option4) as an independent check, so its disagreement with FTFP_BERT is part of the tolerance budget; `scripts/make_references.py` should be re-run with `PL = "FTFP_BERT"` if tolerances are tightened.

## Grading
Answer must be within a task-specific relative tolerance (3–5%) of the reference, report n_events >= min_events and a sane 1-sigma uncertainty, and the run must print the Geant4 banner.
References: slab = NIST XCOM table (validated against Geant4 in tests); proton_range = checked vs NIST PSTAR CSDA (method-independent: no binning involved); alpha/nai = high-statistics Geant4 oracle with a *different* EM list (`FTFP_BERT_EMZ`, EM option4) in `references.json`.
