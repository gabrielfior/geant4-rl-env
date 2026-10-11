"""Validate families B-E: broken scripts fail, fixed pass, and the references are consistent with FTFP_BERT simulations."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from geant4rl import refsim
from geant4rl.tasks import (DENSITY, UNKNOWN_CANDIDATES, UNKNOWN_ENERGIES, UNKNOWN_THICK, _fit_exponent, buggy_nai_src,
                            buggy_transmission_src, grade_answer, make_instance)

B = "Geant4 version"


def run_src(src, tmp_path):
    f = tmp_path / "s.py"
    f.write_text(src)
    out = subprocess.run([sys.executable, "-I", str(f)], capture_output=True, text=True, cwd=tmp_path).stdout
    line = [l for l in out.splitlines() if l.startswith('{"answer"')][-1]
    return json.loads(line), out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_debug_transmission(seed, tmp_path):
    inst = make_instance("debug_transmission", seed)
    bad, out = run_src(buggy_transmission_src(inst.params), tmp_path)
    good, out = run_src(buggy_transmission_src(inst.params, fixed=True), tmp_path)
    assert not grade_answer("debug_transmission", inst.params, bad, out)[0]
    assert grade_answer("debug_transmission", inst.params, good, out)[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_debug_nai(seed, tmp_path):
    inst = make_instance("debug_nai", seed)
    bad, out = run_src(buggy_nai_src(inst.params), tmp_path)
    good, out = run_src(buggy_nai_src(inst.params, fixed=True), tmp_path)
    assert not grade_answer("debug_nai", inst.params, bad, out)[0]
    assert grade_answer("debug_nai", inst.params, good, out)[0]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_attenuation(seed):
    p = make_instance("attenuation_coefficient", seed).params
    vals = []
    for tt in (0.5, 1.0, 1.5):
        t = tt / (0.07 * DENSITY[p["material"]])
        r = refsim.isolated("slab_transmission", material=p["material"], energy_mev=p["energy_mev"], thickness_cm=t, n=30000, seed=4)
        vals.append(-math.log(r["answer"]) / (DENSITY[p["material"]] * t))
    ans = {"answer": sum(vals) / 3, "uncertainty": 0.0005, "n_events": 90000}
    assert grade_answer("attenuation_coefficient", p, ans, B)[0], ans


def test_range_exponent():
    p = make_instance("range_exponent", 0).params
    rs = [refsim.isolated("proton_range", energy_mev=e, n=1500, seed=6)["answer"] for e in p["energies_mev"]]
    ans = {"answer": _fit_exponent(p["energies_mev"], rs), "uncertainty": 0.005, "n_events": 6000}
    assert grade_answer("range_exponent", p, ans, B)[0], ans


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_shield(seed):
    p = make_instance("shield_thickness", seed).params
    from geant4rl.tasks import ref_shield
    t, _ = ref_shield(p)
    r = refsim.isolated("slab_transmission", material=p["material"], energy_mev=p["energy_mev"], thickness_cm=t, n=60000, seed=8)
    assert abs(r["answer"] - p["target_transmission"]) < max(0.15 * p["target_transmission"], 4 * r["uncertainty"]), r


@pytest.mark.parametrize("seed", range(6))
def test_nai_thickness(seed):
    inst = make_instance("nai_min_thickness", seed)
    p = inst.params
    r = refsim.isolated("nai_absorbed_fraction", energy_kev=p["energy_kev"], thickness_cm=p["_t_star"], n=30000, seed=9)
    assert abs(r["answer"] - p["target_fraction"]) < 0.04 * p["target_fraction"], (r, p)


def test_unknown_material_identifiable():
    refs = json.load(open(Path(__file__).parent.parent / "geant4rl" / "references.json"))["unknown_material"]
    sims = {}
    for m in UNKNOWN_CANDIDATES:
        sims[m] = [refsim.isolated("slab_transmission", material=m, energy_mev=e, thickness_cm=UNKNOWN_THICK, n=30000, seed=31)["answer"]
                   for e in UNKNOWN_ENERGIES]
    for true in UNKNOWN_CANDIDATES:
        best = min(UNKNOWN_CANDIDATES, key=lambda m: sum((a - b) ** 2 for a, b in zip(sims[m], refs[true])))
        assert best == true
    p = make_instance("unknown_material", 0).params
    assert grade_answer("unknown_material", p, {"answer": "G4_Cu", "n_events": 99999}, B)[0]
    assert not grade_answer("unknown_material", p, {"answer": "G4_Fe", "n_events": 99999}, B)[0]
