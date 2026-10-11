# geant4rl: RL environments for physics agents

An LLM agent must **write and run a Geant4 simulation** (`geant4_pybind`) and report a physical quantity. A grader checks it against NIST tables or a hidden high-statistics oracle. Model-written code runs in a locked-down Docker container.

## Why physics RL

- **Verifiable by physics, not taste:** attenuation, ranges and detector response have NIST or simulator ground truth.
- **Real tools:** driving Geant4 takes API knowledge, units discipline and statistical sense; agents fail in realistic ways.
- **Tunable difficulty:** from lookups to debugging and design, so pass rates can be set away from 0% and 100%.
- **Correct physics first:** graded quantities never depend on analysis choices the prompt leaves unstated.

## What is here

11 tasks in 5 families (**A** write a simulation, **B** debug one, **C** extract a coefficient, **D** design, **E** identify a material), parametrised by seed.

## Results so far (15 turns, example script given, families B-E, 42 episodes each)

| Model | Pass rate |
|---|---|
| deepseek-v4.1-flash | 30/42 (71%) |
| qwen3.8-27b, thinking | 15/42 (36%) |
| qwen3.8-27b, non-thinking | 14/42 (33%) |

Full 11-task run (99 episodes per model) with Qwen3-8B/14B: pending. Details and failure causes: [docs/RESULTS.md](docs/RESULTS.md).

## More

- [docs/SETUP.md](docs/SETUP.md): install, run, sandbox, physics list, grading
- [docs/RESULTS.md](docs/RESULTS.md): tables and failure analysis
- [docs/LIMITS.md](docs/LIMITS.md): caveats. Note: this repo contains the answer key, so it is not a held-out benchmark.
