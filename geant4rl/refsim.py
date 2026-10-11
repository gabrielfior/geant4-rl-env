"""Hidden reference simulations (the "oracle"). Never shipped to the agent's workdir."""
import math

from geant4_pybind import *

_STATE = {}


def _physics(name):
    if name == "FTFP_BERT":
        return FTFP_BERT()
    if name == "FTFP_BERT_EMZ":  # oracle list: high-precision EM option4 instead of option0
        class EMZ(G4VModularPhysicsList):
            def __init__(self):
                super().__init__()
                self.SetVerboseLevel(0)
                for ph in (G4EmStandardPhysics_option4(), G4DecayPhysics(), G4HadronElasticPhysics(),
                           G4HadronPhysicsFTFP_BERT(), G4StoppingPhysics(), G4IonPhysics()):
                    self.RegisterPhysics(ph)
        return EMZ()
    raise ValueError(name)


def _run(construct, make_gun, on_step, n, seed, physics="FTFP_BERT"):
    class Detector(G4VUserDetectorConstruction):
        def Construct(self):
            return construct()

    class Gen(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = make_gun()

        def GeneratePrimaries(self, event):
            self.gun.GeneratePrimaryVertex(event)

    class Step(G4UserSteppingAction):
        def UserSteppingAction(self, step):
            on_step(step)

    class Actions(G4VUserActionInitialization):
        def Build(self):
            self.SetUserAction(Gen())
            self.SetUserAction(Step())

        def BuildForMaster(self):
            pass

    G4Random.setTheSeed(seed)
    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.SerialOnly)
    det = Detector()
    rm.SetUserInitialization(det)
    pl = _physics(physics)
    rm.SetUserInitialization(pl)
    act = Actions()
    rm.SetUserInitialization(act)
    ui = G4UImanager.GetUIpointer()
    for c in ("/run/verbose 1", "/event/verbose 0", "/tracking/verbose 0"):
        ui.ApplyCommand(c)
    rm.Initialize()
    rm.BeamOn(n)
    return rm, det, act, pl  # keep alive until return


def _world(half, mat="G4_Galactic"):
    nist = G4NistManager.Instance()
    solid = G4Box("World", half, half, half)
    lv = G4LogicalVolume(solid, nist.FindOrBuildMaterial(mat), "World")
    pv = G4PVPlacement(None, G4ThreeVector(), lv, "World", None, False, 0, True)
    return lv, pv


def _gun(particle, energy, z0):
    g = G4ParticleGun(1)
    g.SetParticleDefinition(G4ParticleTable.GetParticleTable().FindParticle(particle))
    g.SetParticleEnergy(energy)
    g.SetParticlePosition(G4ThreeVector(0, 0, z0))
    g.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1))
    return g


def slab_transmission(material, energy_mev, thickness_cm, n=20000, seed=1, physics="FTFP_BERT"):
    """Narrow-beam transmission: fraction of primary gammas leaving the slab undeflected, unchanged in energy."""
    keep = []
    hits = {"n": 0}
    E0 = energy_mev * MeV
    t = thickness_cm * cm

    def construct():
        wlv, pv = _world(1 * m)
        slab = G4LogicalVolume(
            G4Box("Slab", 0.9 * m, 0.9 * m, t / 2),
            G4NistManager.Instance().FindOrBuildMaterial(material), "Slab")
        keep.append(G4PVPlacement(None, G4ThreeVector(), slab, "Slab", wlv, False, 0, True))
        return pv

    def on_step(step):
        tr = step.GetTrack()
        if tr.GetTrackID() != 1:
            return
        post = step.GetPostStepPoint()
        pre = step.GetPreStepPoint()
        if pre.GetTouchable().GetVolume().GetName() == "Slab" and post.GetTouchable().GetVolume() is not None \
                and post.GetTouchable().GetVolume().GetName() == "World" and post.GetPosition().z > 0:
            if abs(post.GetKineticEnergy() - E0) < 1e-9 and post.GetMomentumDirection().z > 1 - 1e-9:
                hits["n"] += 1

    r = _run(construct, lambda: _gun("gamma", E0, -0.5 * m), on_step, n, seed, physics)
    p = hits["n"] / n
    return {"answer": p, "uncertainty": math.sqrt(max(p * (1 - p), 1e-12) / n), "n_events": n}


def alpha_range(energy_mev, n=2000, seed=1, physics="FTFP_BERT"):
    """Mean depth (um) at which primary alphas stop in liquid water."""
    keep = []
    zs = []
    E0 = energy_mev * MeV

    def construct():
        wlv, pv = _world(5 * mm, "G4_WATER")
        return pv

    def on_step(step):
        tr = step.GetTrack()
        if tr.GetTrackID() == 1 and step.GetPostStepPoint().GetKineticEnergy() <= 0:
            zs.append(step.GetPostStepPoint().GetPosition().z / um + 5000.0)

    r = _run(construct, lambda: _gun("alpha", E0, -4.999 * mm), on_step, n, seed, physics)
    m = sum(zs) / len(zs)
    sd = math.sqrt(sum((z - m) ** 2 for z in zs) / (len(zs) - 1))
    return {"answer": m - 1.0, "uncertainty": sd / math.sqrt(len(zs)), "n_events": n}


def proton_range(energy_mev, n=2000, seed=1, physics="FTFP_BERT"):
    """Mean stopping depth (cm) in a water phantom of primary protons that stop without a nuclear inelastic interaction."""
    keep = []
    zs = []
    E0 = energy_mev * MeV

    def construct():
        wlv, pv = _world(1 * m)
        ph = G4LogicalVolume(G4Box("Phantom", 0.3 * m, 0.3 * m, 0.2 * m),
                             G4NistManager.Instance().FindOrBuildMaterial("G4_WATER"), "Phantom")
        keep.append(G4PVPlacement(None, G4ThreeVector(0, 0, 0.2 * m), ph, "Phantom", wlv, False, 0, True))
        return pv

    def on_step(step):
        if step.GetTrack().GetTrackID() != 1:
            return
        post = step.GetPostStepPoint()
        if post.GetKineticEnergy() > 0:
            return
        proc = post.GetProcessDefinedStep()
        if proc is not None and proc.GetProcessName() == "protonInelastic":
            return
        zs.append(post.GetPosition().z / cm)

    r = _run(construct, lambda: _gun("proton", E0, -0.01 * m), on_step, n, seed, physics)
    mean = sum(zs) / len(zs)
    sd = math.sqrt(sum((z - mean) ** 2 for z in zs) / (len(zs) - 1))
    return {"answer": mean, "uncertainty": sd / math.sqrt(len(zs)), "n_events": n}


def nai_absorbed_fraction(energy_kev, thickness_cm, n=20000, seed=1, physics="FTFP_BERT"):
    """Mean fraction of incident gamma energy deposited in a 3-inch-diameter NaI cylinder."""
    keep = []
    acc = {"e": 0.0, "e2": 0.0, "cur": 0.0}
    E0 = energy_kev * keV

    def construct():
        wlv, pv = _world(1 * m)
        nai = G4LogicalVolume(G4Tubs("NaI", 0, 3.81 * cm, thickness_cm * cm / 2, 0, twopi),
                              G4NistManager.Instance().FindOrBuildMaterial("G4_SODIUM_IODIDE"), "NaI")
        keep.append(G4PVPlacement(None, G4ThreeVector(), nai, "NaI", wlv, False, 0, True))
        return pv

    def on_step(step):
        if step.GetPreStepPoint().GetTouchable().GetVolume().GetName() == "NaI":
            acc["e"] += step.GetTotalEnergyDeposit()

    r = _run(construct, lambda: _gun("gamma", E0, -0.3 * m), on_step, n, seed, physics)
    f = acc["e"] / (n * E0)
    return {"answer": f, "uncertainty": 0.01 * f + 1e-4, "n_events": n}


def solve(spec):
    """Reference solution used for testing graders: runs the right simulation for a public spec."""
    t = spec["task"]
    p = spec["params"]
    if t == "slab_transmission":
        return isolated("slab_transmission", material=p["material"], energy_mev=p["energy_mev"], thickness_cm=p["thickness_cm"], n=p["min_events"], seed=11)
    if t == "alpha_range":
        return isolated("alpha_range", energy_mev=p["energy_mev"], n=p["min_events"], seed=11)
    if t == "proton_range":
        return isolated("proton_range", energy_mev=p["energy_mev"], n=p["min_events"], seed=11)
    if t == "nai_absorbed_fraction":
        return isolated("nai_absorbed_fraction", energy_kev=p["energy_kev"], thickness_cm=p["thickness_cm"], n=p["min_events"], seed=11)
    raise ValueError(t)


def isolated(name, **kw):
    """Run one simulation in a fresh process (Geant4 allows one run manager per process)."""
    import json, subprocess, sys
    code = ("import sys, json; sys.path.insert(0, %r); from geant4rl import refsim; "
            "print('@@RESULT@@' + json.dumps(getattr(refsim, %r)(**json.loads(sys.argv[1]))))"
            % (str(__import__('pathlib').Path(__file__).parent.parent), name))
    out = subprocess.run([sys.executable, "-I", "-u", "-c", "import ctypes; _l=ctypes.CDLL(None); _l.signal.argtypes=[ctypes.c_int, ctypes.c_void_p]; _l.signal(6, ctypes.cast(_l._exit, ctypes.c_void_p).value)\n" + code, json.dumps(kw)], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    raise RuntimeError(f"{name} failed (exit {out.returncode}): {out.stdout[-500:]}\n{out.stderr[-500:]}")
