"""Precompute oracle references (high-statistics Geant4, FTFP_BERT_EMZ) for tasks without closed-form answers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from geant4rl import refsim
from geant4rl.tasks import ALPHA_E, NAI_E, NAI_T, PSTAR_CSDA, REFS_PATH

PL = "FTFP_BERT_EMZ"
refs = {"alpha_range": {}, "proton_range": {}, "nai_absorbed_fraction": {}}
for e in ALPHA_E:
    refs["alpha_range"][str(e)] = refsim.isolated("alpha_range", energy_mev=e, n=5000, seed=7, physics=PL)["answer"]
for e in PSTAR_CSDA:
    refs["proton_range"][str(e)] = refsim.isolated("proton_range", energy_mev=e, n=5000, seed=7, physics=PL)["answer"]
for e in NAI_E:
    for t in NAI_T:
        refs["nai_absorbed_fraction"][f"{e}_{t}"] = refsim.isolated("nai_absorbed_fraction", energy_kev=e, thickness_cm=t, n=100000, seed=7, physics=PL)["answer"]
# unknown-material measurements: narrow-beam transmission, 2.5 cm slab, 60k events each, oracle physics
from concurrent.futures import ThreadPoolExecutor
from geant4rl.tasks import UNKNOWN_CANDIDATES, UNKNOWN_ENERGIES, UNKNOWN_THICK
jobs = [(m, e) for m in UNKNOWN_CANDIDATES for e in UNKNOWN_ENERGIES]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lambda j: refsim.isolated("slab_transmission", material=j[0], energy_mev=j[1], thickness_cm=UNKNOWN_THICK,
                                                n=60000, seed=21, physics=PL)["answer"], jobs))
refs["unknown_material"] = {m: [r for (mm, e), r in zip(jobs, res) if mm == m] for m in UNKNOWN_CANDIDATES}
REFS_PATH.write_text(json.dumps(refs, indent=2))
print(json.dumps(refs, indent=2))
