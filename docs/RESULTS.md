# Results

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
