"""Reference solution.py (uses the hidden oracle module; for grader testing only, NOT given to agents)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("~/code/geant4-rl-env").expanduser()))
from geant4rl import refsim

spec = json.load(open("spec.json"))
p = dict(spec["params"])
n = p.pop("min_events")
print(json.dumps(getattr(refsim, spec["task"])(n=n, seed=5, **p)))
