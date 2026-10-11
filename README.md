# geant4rl — RL environments for physics agents

Verifiable-reward tasks where an LLM agent must **write and run a Geant4 simulation** (via `geant4_pybind`) and report a physical quantity, graded against NIST tables or a hidden high-statistics oracle.

## Motivation

RL environments exist in bulk for coding and browser use; science is thin. Physics is a good fit for verifiable rewards:

- **Ground truth is physical, not human taste.** Attenuation, ranges and detector response can be checked against NIST XCOM/PSTAR or a reference simulation.
- **Simulators are the real tools of the field.** Geant4, FLUKA, OpenFOAM and friends are what physicists use, and driving them requires API knowledge, units discipline and statistical sense. Agents fail in realistic ways (wrong units, no EM physics, miscounted events).
- **Difficulty is tunable.** The same simulator supports from one-liner lookups to debugging and design tasks, which is what RL needs (a pass rate that is neither 0% nor 100%).
- **Correct physics over clever scoring.** Graded quantities must not depend on analysis choices the prompt doesn't state; a task that did (Bragg-peak binning) was replaced by a binning-free range metric.

Five task families: **A** write a simulation, **B** debug a broken one, **C** extract a coefficient or exponent from simulated data, **D** design (shield thickness, minimum crystal), **E** identify an unknown material. 11 tasks, parametrised by seed.

## Results so far (pilot, honest numbers)

Agent loop: up to 15 turns, the model writes one Python script per turn, it runs in a locked-down Docker container, the model sees the output; the last script is graded. Models: `deepseek-v4.1-flash`, `qwen3.8-27b`, `Qwen3-8B`/`14B`. Episodes are 2 seeds x 3 samples per task unless stated; with 6 episodes per task, differences of one or two are noise.

**Family A (write a simulation), `deepseek-v4.1-flash`, working example script in the prompt:** slab transmission 6/6, proton range 6/6, NaI absorbed fraction 5/6, alpha range 1/6 (grader flaw since fixed: deterministic stopping gives zero uncertainty). Without the example: 0/24, because agents spend all turns discovering the pybind API. Qwen3-8B without the example: 0/24. Family A is therefore mostly an *API-discovery* test, and too easy once the example is given.

**Families B-E, example given, 15 turns (regraded from transcripts):**

| Task | deepseek-v4.1-flash | qwen3.8-27b thinking | qwen3.8-27b non-thinking |
|---|---|---|---|
| debug_transmission | 6/6 | 4/6 | 2/6 |
| debug_nai | 5/6 | 3/6 | 5/6 |
| attenuation_coefficient | 2/6 | 1/6 | 1/6 |
| range_exponent | 6/6 | 1/6 | 0/6 |
| shield_thickness | 2/6 | 1/6 | 1/6 |
| nai_min_thickness | 4/6 | 3/6 | 4/6 |
| unknown_material | 5/6 | 2/6 | 1/6 |
| **Total** | **30/42 (71%)** | **15/42 (36%)** | **14/42 (33%)** |

Failure causes: most Qwen failures are *no result printed* (21-25 of 42) after turns spent exploring the API; 19% of Qwen thinking replies were empty (reasoning exhausted the token budget). DeepSeek's failures are mostly agents running out of turns, plus one wrong material.

A full 11-task run (3 seeds x 3 samples, 99 episodes per model) for DeepSeek, Qwen3-8B and Qwen3-14B was in progress when this was written; its numbers will be added here.

An earlier Qwen run scored 0/42 because of a bug on our side (Geant4 data paths missing in the sandbox); it is excluded.

## Caveats

- Small samples; no confidence intervals yet.
- XCOM/PSTAR values were typed from memory and cross-checked against Geant4 within tolerance; verify against NIST before relying on them.
- Anti-cheat is weak: an agent can print a correct closed-form number (slab transmission is Beer-Lambert). `geant4rl/references.json` and the tables in `tasks.py` are the answer key, so this repo is **not** a held-out benchmark.
- The oracle is Geant4 itself (different EM list), so it is not fully independent.

## Details

Four tasks with verifiable graders: `slab_transmission`, `alpha_range`, `proton_range`, `nai_absorbed_fraction`.
An agent gets `prompt.md` + `spec.json` in a workdir, writes `solution.py` (using `geant4_pybind`), the harness runs it and grades the JSON answer.

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

## Known limits
- Banner check is weak: an agent could print the banner and a closed-form number (slab transmission is Beer–Lambert). Next step: re-run the submitted code under a Geant4 hook / check event counts from a stepping-action log.
- XCOM/PSTAR numbers were typed from memory and only validated by the Geant4 cross-checks (<5%); verify against NIST before publishing.
- Geant4 allows one run manager per process: every simulation runs in its own subprocess (`refsim.isolated`).
- Task design rule: graded quantities must not depend on analysis choices (binning etc.) not stated in the prompt. The original Bragg-peak task did and was replaced.
- pybind quirks: `z` is a property (not `z()`); use `GetTouchable().GetVolume()`; `FTFP_BERT.ReplacePhysics` is not exposed (custom modular list used).
