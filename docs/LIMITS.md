# Known limits

- Small samples; no confidence intervals yet.
- XCOM/PSTAR values were typed from memory and cross-checked against Geant4 within tolerance; verify against NIST before relying on them.
- Anti-cheat is weak: an agent can print a correct closed-form number (slab transmission is Beer-Lambert). `geant4rl/references.json` and the tables in `tasks.py` are the answer key, so this repo is **not** a held-out benchmark.
- The oracle is Geant4 itself (different EM list), so it is not fully independent.

- Banner check is weak: an agent could print the banner and a closed-form number (slab transmission is Beer–Lambert). Next step: re-run the submitted code under a Geant4 hook / check event counts from a stepping-action log.
- XCOM/PSTAR numbers were typed from memory and only validated by the Geant4 cross-checks (<5%); verify against NIST before publishing.
- Geant4 allows one run manager per process: every simulation runs in its own subprocess (`refsim.isolated`).
- Task design rule: graded quantities must not depend on analysis choices (binning etc.) not stated in the prompt. The original Bragg-peak task did and was replaced.
- pybind quirks: `z` is a property (not `z()`); use `GetTouchable().GetVolume()`; `FTFP_BERT.ReplacePhysics` is not exposed (custom modular list used).
