"""Sanity: the Geant4 oracle must agree with independent tables (so the grader is not self-referential),
and the grader must accept a correct run and reject cheats."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from geant4rl import refsim
from geant4rl.tasks import DENSITY, PSTAR_CSDA, XCOM, grade_answer, make_instance

BANNER = "Geant4 version"


@pytest.mark.parametrize("mat,e", [("G4_Al", 1.0), ("G4_Fe", 0.5), ("G4_Pb", 1.0), ("G4_Pb", 2.0)])
def test_slab_matches_xcom(mat, e):
    import math
    t = 0.4 / (XCOM[mat][e] * DENSITY[mat])
    res = refsim.isolated('slab_transmission', material=mat, energy_mev=e, thickness_cm=t, n=20000, seed=3)
    expect = math.exp(-XCOM[mat][e] * DENSITY[mat] * t)
    assert abs(res["answer"] - expect) < max(0.05 * expect, 4 * res["uncertainty"])


def test_proton_range_matches_pstar():
    for e in (100, 150):
        res = refsim.isolated('proton_range', energy_mev=e, n=1500, seed=3)
        assert abs(res["answer"] - PSTAR_CSDA[e]) < 0.02 * PSTAR_CSDA[e]


def test_alpha_5mev_approx_37um():
    res = refsim.isolated('alpha_range', energy_mev=5.0, n=500, seed=3)
    assert 33 < res["answer"] < 41


def test_grader_rejects_cheats():
    inst = make_instance("slab_transmission", 1)
    good = refsim.solve(inst.public_spec())
    assert grade_answer("slab_transmission", inst.params, good, BANNER)[0]
    assert "warning" in grade_answer("slab_transmission", inst.params, good, "")[1]  # banner is now a warning only
    fake = dict(good, n_events=5)
    assert not grade_answer("slab_transmission", inst.params, fake, BANNER)[0]
    wrong = dict(good, answer=good["answer"] * 1.3)
    assert not grade_answer("slab_transmission", inst.params, wrong, BANNER)[0]
    assert not grade_answer("slab_transmission", inst.params, dict(good, uncertainty=-1), BANNER)[0]
