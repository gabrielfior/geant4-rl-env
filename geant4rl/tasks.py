"""Family A tasks. Each task samples an instance from a discrete grid (so oracle references can be precomputed),
renders a prompt, and grades the agent's JSON answer."""
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

REFS_PATH = Path(__file__).parent / "references.json"

# NIST XCOM total attenuation incl. coherent, cm^2/g (rounded). Validated against Geant4 in tests/test_tasks.py.
XCOM = {
    "G4_Al": {0.5: 0.0844, 1.0: 0.0613, 1.5: 0.0500, 2.0: 0.0432},
    "G4_Fe": {0.5: 0.0840, 1.0: 0.0599, 1.5: 0.0488, 2.0: 0.0425},
    "G4_Pb": {0.5: 0.1614, 1.0: 0.0710, 1.5: 0.0522, 2.0: 0.0457},
}
DENSITY = {"G4_Al": 2.699, "G4_Fe": 7.874, "G4_Pb": 11.35}

# NIST PSTAR CSDA range in water, g/cm^2 (Bragg peak sits at ~ this depth).
PSTAR_CSDA = {70: 4.080, 100: 7.718, 150: 15.77, 200: 25.96}

ALPHA_E = [4.0, 5.0, 6.0, 7.0]
NAI_E = [300.0, 662.0, 1000.0]
NAI_T = [2.0, 3.0, 5.0, 7.5]

ANSWER_FORMAT = (
    "Use the `FTFP_BERT` reference physics list (`geant4_pybind.FTFP_BERT`, Geant4 11.4) with default production cuts. "
    "Write `solution.py` that uses `geant4_pybind` to run the simulation and prints on its own line (the last such line is graded; Geant4 may print teardown text after it) one JSON "
    'object: {"answer": <float>, "uncertainty": <float, 1 sigma>, "n_events": <int>}. '
    "It must run in under 10 minutes on one CPU core. Use at least {min_events} events."
)


@dataclass
class Instance:
    task: str
    params: dict
    prompt: str
    unit: str
    files: dict = None

    def public_spec(self):
        pub = {k: v for k, v in self.params.items() if not k.startswith("_")}
        return {"task": self.task, "params": pub, "unit": self.unit}


def _refs():
    return json.loads(REFS_PATH.read_text()) if REFS_PATH.exists() else {}


def _mu(material, e):
    return XCOM[material][e]


# ---------- instance makers ----------

def make_slab(seed):
    r = random.Random(seed)
    mat = r.choice(list(XCOM))
    e = r.choice(list(XCOM[mat]))
    target_t = r.uniform(0.15, 0.6)
    thick = round(-math.log(target_t) / (_mu(mat, e) * DENSITY[mat]), 2)
    p = {"material": mat, "energy_mev": e, "thickness_cm": thick, "min_events": 20000}
    prompt = (f"Simulate a narrow, monoenergetic {e} MeV gamma pencil beam normally incident on a {thick} cm thick, "
              f"large-area slab of {mat} (Geant4 NIST material name). Report the fraction of primary photons that "
              f"emerge from the back face with unchanged energy AND direction (the narrow-beam transmission). "
              f"Unit: dimensionless fraction. " + ANSWER_FORMAT.replace("{min_events}", str(p["min_events"])))
    return Instance("slab_transmission", p, prompt, "fraction")


def make_alpha(seed):
    r = random.Random(seed)
    e = r.choice(ALPHA_E)
    p = {"energy_mev": e, "min_events": 2000}
    prompt = (f"Simulate {e} MeV alpha particles entering liquid water (G4_WATER) along a straight line. Report the "
              f"mean depth at which they stop, in micrometres. " + ANSWER_FORMAT.replace("{min_events}", str(p["min_events"])))
    return Instance("alpha_range", p, prompt, "um")


def make_proton(seed):
    r = random.Random(seed)
    e = r.choice(list(PSTAR_CSDA))
    p = {"energy_mev": e, "min_events": 2000}
    prompt = (f"Simulate a {e} MeV proton pencil beam entering a large water phantom (G4_WATER) at normal incidence. "
              f"Report the mean depth in cm, measured from the phantom surface, at which primary protons come to rest, "
              f"averaged over primaries that stop without undergoing a nuclear inelastic interaction. "
              + ANSWER_FORMAT.replace("{min_events}", str(p["min_events"])))
    return Instance("proton_range", p, prompt, "cm")


def make_nai(seed):
    r = random.Random(seed)
    e = r.choice(NAI_E)
    t = r.choice(NAI_T)
    p = {"energy_kev": e, "thickness_cm": t, "min_events": 20000}
    prompt = (f"A {e} keV gamma pencil beam hits the centre of the flat face of a NaI cylinder (G4_SODIUM_IODIDE, "
              f"diameter 7.62 cm, thickness {t} cm) in vacuum. Report the mean fraction of incident photon energy "
              f"deposited in the crystal. Unit: dimensionless fraction. " + ANSWER_FORMAT.replace("{min_events}", str(p["min_events"])))
    return Instance("nai_absorbed_fraction", p, prompt, "fraction")


# ---------- references: (value, relative tolerance) ----------

def ref_slab(p):
    return math.exp(-_mu(p["material"], p["energy_mev"]) * DENSITY[p["material"]] * p["thickness_cm"]), 0.05


def ref_alpha(p):
    return _refs()["alpha_range"][str(p["energy_mev"])], 0.05


def ref_proton(p):
    return PSTAR_CSDA[p["energy_mev"]] * 0.996, 0.02  # projected range ~ CSDA range (multiple scattering shortens it slightly)


def ref_nai(p):
    return _refs()["nai_absorbed_fraction"][f'{p["energy_kev"]}_{p["thickness_cm"]}'], 0.05


TASKS = {
    "slab_transmission": (make_slab, ref_slab),
    "alpha_range": (make_alpha, ref_alpha),
    "proton_range": (make_proton, ref_proton),
    "nai_absorbed_fraction": (make_nai, ref_nai),
}


def make_instance(task, seed):
    return TASKS[task][0](seed)


def grade_answer(task, params, result, stdout=""):
    """Return (passed: bool, reason: str)."""
    warn = "" if "Geant4 version" in stdout else " (warning: no Geant4 banner in final run; results may be cached)"
    if task in CATEGORICAL:
        try:
            n = int(result["n_events"])
        except (KeyError, TypeError, ValueError):
            return False, "malformed result JSON"
        if n < params["min_events"]:
            return False, f"n_events {n} < min_events {params['min_events']}"
        ref, _ = TASKS[task][1](params)
        return (str(result.get("answer")) == ref), (("ok" + warn) if str(result.get("answer")) == ref else f"wrong material {result.get('answer')!r}")
    try:
        a, u, n = float(result["answer"]), float(result["uncertainty"]), int(result["n_events"])
    except (KeyError, TypeError, ValueError):
        return False, "malformed result JSON"
    if not all(map(math.isfinite, (a, u))):
        return False, "non-finite values"
    if n < params["min_events"]:
        return False, f"n_events {n} < min_events {params['min_events']}"
    if u < 0 or u > 0.2 * abs(a):  # u == 0 is legal: alpha stopping in FTFP_BERT is deterministic (no straggling)
        return False, "uncertainty negative or too large to be informative"
    ref, rel = TASKS[task][1](params)
    if abs(a - ref) > max(rel * abs(ref), 3 * u):
        return False, f"answer {a:.5g} outside tolerance of reference (rel {rel})"
    return True, "ok" + warn


# =====================  Families B-E  =====================
UNKNOWN_CANDIDATES = ["G4_Al", "G4_Ti", "G4_Fe", "G4_Cu", "G4_Sn", "G4_W", "G4_Pb"]
UNKNOWN_ORDER = [3, 2, 4, 0, 1, 5, 6]          # seed -> candidate index (seeds 0,1 = the hard Cu/Fe pair)
UNKNOWN_ENERGIES = [0.3, 0.6, 1.0, 1.5]
UNKNOWN_THICK = 2.5


def _fmt(p):
    return ANSWER_FORMAT.replace("{min_events}", str(p["min_events"]))


def _pick_slab_params(seed):
    r = random.Random(seed)
    mat = r.choice(list(XCOM))
    e = r.choice(list(XCOM[mat]))
    t = round(-math.log(r.uniform(0.15, 0.6)) / (_mu(mat, e) * DENSITY[mat]), 2)
    return mat, e, t


def buggy_transmission_src(p, fixed=False):
    thick_unit = "cm" if fixed else "mm"
    count = ("abs(post.GetKineticEnergy() - E0) < 1e-9 and post.GetMomentumDirection().z > 1 - 1e-9"
             if fixed else "True")
    return f'''"""Narrow-beam transmission of a gamma pencil beam through a slab. Supposed to print the fraction of primary
photons leaving the back face with unchanged energy and direction. Something is wrong with it."""
import json, math
from geant4_pybind import *

MATERIAL, ENERGY_MEV, THICKNESS, N_EVENTS = "{p["material"]}", {p["energy_mev"]}, {p["thickness_cm"]}, {p["min_events"]}
E0 = ENERGY_MEV * MeV
t = THICKNESS * {thick_unit}
hits = {{"n": 0}}


class Detector(G4VUserDetectorConstruction):
    def Construct(self):
        nist = G4NistManager.Instance()
        wlv = G4LogicalVolume(G4Box("World", 1 * m, 1 * m, 1 * m), nist.FindOrBuildMaterial("G4_Galactic"), "World")
        self.wpv = G4PVPlacement(None, G4ThreeVector(), wlv, "World", None, False, 0, True)
        slv = G4LogicalVolume(G4Box("Slab", 0.9 * m, 0.9 * m, t / 2), nist.FindOrBuildMaterial(MATERIAL), "Slab")
        self.spv = G4PVPlacement(None, G4ThreeVector(), slv, "Slab", wlv, False, 0, True)
        return self.wpv


class Gen(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = G4ParticleGun(1)
        self.gun.SetParticleDefinition(G4ParticleTable.GetParticleTable().FindParticle("gamma"))
        self.gun.SetParticleEnergy(E0)
        self.gun.SetParticlePosition(G4ThreeVector(0, 0, -0.5 * m))
        self.gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1))

    def GeneratePrimaries(self, event):
        self.gun.GeneratePrimaryVertex(event)


class Step(G4UserSteppingAction):
    def UserSteppingAction(self, step):
        if step.GetTrack().GetTrackID() != 1:
            return
        pre, post = step.GetPreStepPoint(), step.GetPostStepPoint()
        a, b = pre.GetTouchable().GetVolume(), post.GetTouchable().GetVolume()
        if a is not None and a.GetName() == "Slab" and b is not None and b.GetName() == "World" and post.GetPosition().z > 0:
            if {count}:
                hits["n"] += 1


class Actions(G4VUserActionInitialization):
    def Build(self):
        self.SetUserAction(Gen())
        self.SetUserAction(Step())


rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.SerialOnly)
det, physics, actions = Detector(), FTFP_BERT(), Actions()
rm.SetUserInitialization(det)
rm.SetUserInitialization(physics)
rm.SetUserInitialization(actions)
G4Random.setTheSeed(3)
rm.Initialize()
rm.BeamOn(N_EVENTS)
p = hits["n"] / N_EVENTS
print(json.dumps({{"answer": p, "uncertainty": math.sqrt(max(p * (1 - p), 1e-12) / N_EVENTS), "n_events": N_EVENTS}}))
'''


def buggy_nai_src(p, fixed=False):
    physics = ("FTFP_BERT()" if fixed else "NoEM()")
    gun_z = "-0.3 * m" if fixed else "0.3 * m"
    return f'''"""Energy deposited by a gamma pencil beam in a NaI cylinder (diameter 7.62 cm, centred at the origin, axis z).
Supposed to print the mean fraction of incident photon energy deposited in the crystal. It prints ~0. Fix it."""
import json
from geant4_pybind import *

ENERGY_KEV, THICKNESS_CM, N_EVENTS = {p["energy_kev"]}, {p["thickness_cm"]}, {p["min_events"]}
E0 = ENERGY_KEV * keV
acc = {{"e": 0.0}}


class NoEM(G4VModularPhysicsList):
    def __init__(self):
        super().__init__()
        self.SetVerboseLevel(0)
        self.RegisterPhysics(G4DecayPhysics())


class Detector(G4VUserDetectorConstruction):
    def Construct(self):
        nist = G4NistManager.Instance()
        wlv = G4LogicalVolume(G4Box("World", 1 * m, 1 * m, 1 * m), nist.FindOrBuildMaterial("G4_Galactic"), "World")
        self.wpv = G4PVPlacement(None, G4ThreeVector(), wlv, "World", None, False, 0, True)
        clv = G4LogicalVolume(G4Tubs("NaI", 0, 3.81 * cm, THICKNESS_CM * cm / 2, 0, twopi),
                              nist.FindOrBuildMaterial("G4_SODIUM_IODIDE"), "NaI")
        self.cpv = G4PVPlacement(None, G4ThreeVector(), clv, "NaI", wlv, False, 0, True)
        return self.wpv


class Gen(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = G4ParticleGun(1)
        self.gun.SetParticleDefinition(G4ParticleTable.GetParticleTable().FindParticle("gamma"))
        self.gun.SetParticleEnergy(E0)
        self.gun.SetParticlePosition(G4ThreeVector(0, 0, {gun_z}))
        self.gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1))

    def GeneratePrimaries(self, event):
        self.gun.GeneratePrimaryVertex(event)


class Step(G4UserSteppingAction):
    def UserSteppingAction(self, step):
        v = step.GetPreStepPoint().GetTouchable().GetVolume()
        if v is not None and v.GetName() == "NaI":
            acc["e"] += step.GetTotalEnergyDeposit()


class Actions(G4VUserActionInitialization):
    def Build(self):
        self.SetUserAction(Gen())
        self.SetUserAction(Step())


rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.SerialOnly)
det, physics, actions = Detector(), {physics}, Actions()
rm.SetUserInitialization(det)
rm.SetUserInitialization(physics)
rm.SetUserInitialization(actions)
G4Random.setTheSeed(3)
rm.Initialize()
rm.BeamOn(N_EVENTS)
f = acc["e"] / (N_EVENTS * E0)
print(json.dumps({{"answer": f, "uncertainty": 0.01 * f + 1e-4, "n_events": N_EVENTS}}))
'''


def make_debug_transmission(seed):
    mat, e, t = _pick_slab_params(seed)
    p = {"material": mat, "energy_mev": e, "thickness_cm": t, "min_events": 20000}
    prompt = (f"`buggy_transmission.py` in the current directory is meant to compute the narrow-beam transmission (fraction of "
              f"primary photons leaving the back face with unchanged energy AND direction) of a {e} MeV gamma pencil beam through "
              f"a {t} cm thick {mat} slab, using the FTFP_BERT list, but its result is physically wrong. Find and fix the bugs "
              f"(there may be more than one) and write the corrected script as `solution.py`. " + _fmt(p))
    return Instance("debug_transmission", p, prompt, "fraction", {"buggy_transmission.py": buggy_transmission_src(p)})


def make_debug_nai(seed):
    r = random.Random(seed)
    p = {"energy_kev": r.choice(NAI_E), "thickness_cm": r.choice(NAI_T), "min_events": 20000}
    prompt = (f"`buggy_nai.py` in the current directory should compute the mean fraction of a {p['energy_kev']} keV gamma pencil "
              f"beam's energy deposited in a {p['thickness_cm']} cm thick NaI cylinder (the beam enters the flat face centre), but "
              f"it prints ~0. Find and fix the bugs (there may be more than one) and write the corrected script as `solution.py`. "
              + _fmt(p))
    return Instance("debug_nai", p, prompt, "fraction", {"buggy_nai.py": buggy_nai_src(p)})


def make_attenuation(seed):
    mat, e, _ = _pick_slab_params(seed)
    p = {"material": mat, "energy_mev": e, "min_events": 30000}
    prompt = (f"Using Geant4 simulations only (no tables), determine the narrow-beam mass attenuation coefficient mu/rho of {mat} "
              f"for {e} MeV photons, in cm^2/g (narrow-beam = only primary photons that leave the slab with unchanged energy and "
              f"direction count as transmitted; no coherent-vs-incoherent exclusions). You choose the slab thicknesses and how to "
              f"combine them; `min_events` is the total over all your runs. Unit: cm^2/g. " + _fmt(p))
    return Instance("attenuation_coefficient", p, prompt, "cm2/g")


def make_range_exponent(seed):
    p = {"energies_mev": [70, 100, 150, 200], "min_events": 4000}
    prompt = ("Protons of 70, 100, 150 and 200 MeV enter a large water phantom (G4_WATER) at normal incidence. For each energy "
              "compute the mean stopping depth of primary protons that stop without a nuclear inelastic interaction, then fit "
              "R = a * E^p by least squares in log-log space and report the exponent p (dimensionless). `min_events` is the "
              "total over all four energies. " + _fmt(p))
    return Instance("range_exponent", p, prompt, "dimensionless")


def make_shield(seed):
    r = random.Random(seed)
    mat = r.choice(list(XCOM))
    e = r.choice(list(XCOM[mat]))
    f = r.choice([0.1, 0.05, 0.01])
    p = {"material": mat, "energy_mev": e, "target_transmission": f, "min_events": 30000}
    prompt = (f"A monoenergetic {e} MeV gamma pencil beam hits a large {mat} slab at normal incidence. Find the minimum slab "
              f"thickness (cm) such that the narrow-beam transmission (primary photons leaving the back face with unchanged energy "
              f"and direction) is at most {f}. `min_events` is the total over all your runs. Unit: cm. " + _fmt(p))
    return Instance("shield_thickness", p, prompt, "cm")


def make_nai_thickness(seed):
    r = random.Random(seed)
    e = r.choice([662.0, 1000.0])
    t_star = r.choice([3.0, 5.0, 7.5])
    target = round(_refs()["nai_absorbed_fraction"][f"{e}_{t_star}"], 2)
    p = {"energy_kev": e, "target_fraction": target, "min_events": 30000, "_t_star": t_star}
    pub = {k: v for k, v in p.items() if not k.startswith("_")}
    prompt = (f"A {e} keV gamma pencil beam hits the centre of the flat face of a NaI cylinder (G4_SODIUM_IODIDE, diameter 7.62 cm) "
              f"in vacuum. Find the smallest crystal thickness (cm) for which the mean fraction of incident photon energy deposited "
              f"in the crystal is at least {target}. `min_events` is the total over all your runs. Unit: cm. " + _fmt(p))
    return Instance("nai_min_thickness", p, prompt, "cm")


def make_unknown_material(seed):
    mat = UNKNOWN_CANDIDATES[UNKNOWN_ORDER[seed % 7]]
    meas = _refs()["unknown_material"][mat]
    p = {"candidates": UNKNOWN_CANDIDATES, "thickness_cm": UNKNOWN_THICK, "min_events": 30000, "_material": mat}
    files = {"measurements.json": json.dumps({"slab_thickness_cm": UNKNOWN_THICK, "beam": "narrow-beam gamma pencil",
                                              "measurements": [{"energy_mev": e, "transmission": t}
                                                               for e, t in zip(UNKNOWN_ENERGIES, meas)]}, indent=1)}
    prompt = (f"`measurements.json` holds narrow-beam transmission measurements (fraction of primary photons leaving the slab with "
              f"unchanged energy and direction) of a {UNKNOWN_THICK} cm slab of an unknown material for several gamma energies. "
              f"The material is one of {UNKNOWN_CANDIDATES} (Geant4 NIST names). Use Geant4 (FTFP_BERT) simulations of the "
              f"candidates to identify it. The JSON `answer` must be the material name string (e.g. \"G4_Al\"); `uncertainty` may be "
              f"0. `min_events` is the total over all your runs. " + _fmt(p))
    return Instance("unknown_material", p, prompt, "material", files)


def _fit_exponent(es, rs):
    x = [math.log(e) for e in es]
    y = [math.log(r) for r in rs]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def ref_attenuation(p):
    return _mu(p["material"], p["energy_mev"]), 0.05


def ref_exponent(p):
    return _fit_exponent(p["energies_mev"], [PSTAR_CSDA[e] for e in p["energies_mev"]]), 0.03


def ref_shield(p):
    return -math.log(p["target_transmission"]) / (_mu(p["material"], p["energy_mev"]) * DENSITY[p["material"]]), 0.06


def ref_nai_thickness(p):
    return p["_t_star"], 0.10


def ref_unknown(p):
    return p["_material"], 0


TASKS.update({
    "debug_transmission": (make_debug_transmission, ref_slab),
    "debug_nai": (make_debug_nai, ref_nai),
    "attenuation_coefficient": (make_attenuation, ref_attenuation),
    "range_exponent": (make_range_exponent, ref_exponent),
    "shield_thickness": (make_shield, ref_shield),
    "nai_min_thickness": (make_nai_thickness, ref_nai_thickness),
    "unknown_material": (make_unknown_material, ref_unknown),
})
FAMILY = {"A": ["slab_transmission", "alpha_range", "proton_range", "nai_absorbed_fraction"],
          "B": ["debug_transmission", "debug_nai"], "C": ["attenuation_coefficient", "range_exponent"],
          "D": ["shield_thickness", "nai_min_thickness"], "E": ["unknown_material"]}
CATEGORICAL = {"unknown_material"}
