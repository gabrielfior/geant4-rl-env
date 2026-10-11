"""Minimal working geant4_pybind program (generic; unrelated to the graded tasks).
Fires 2 MeV electrons into a 1 cm aluminium block and prints the mean energy deposited in it.
Notes: one run manager per process; geometry/physics/actions are registered before Initialize()."""
from geant4_pybind import *

edep = {"sum": 0.0}


class Detector(G4VUserDetectorConstruction):
    def Construct(self):
        nist = G4NistManager.Instance()
        world_lv = G4LogicalVolume(G4Box("World", 0.5 * m, 0.5 * m, 0.5 * m), nist.FindOrBuildMaterial("G4_Galactic"), "World")
        self.world_pv = G4PVPlacement(None, G4ThreeVector(), world_lv, "World", None, False, 0, True)
        block_lv = G4LogicalVolume(G4Box("Block", 0.2 * m, 0.2 * m, 0.5 * cm), nist.FindOrBuildMaterial("G4_Al"), "Block")
        self.block_pv = G4PVPlacement(None, G4ThreeVector(), block_lv, "Block", world_lv, False, 0, True)
        return self.world_pv


class Generator(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = G4ParticleGun(1)
        self.gun.SetParticleDefinition(G4ParticleTable.GetParticleTable().FindParticle("e-"))
        self.gun.SetParticleEnergy(2 * MeV)
        self.gun.SetParticlePosition(G4ThreeVector(0, 0, -0.2 * m))
        self.gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1))

    def GeneratePrimaries(self, event):
        self.gun.GeneratePrimaryVertex(event)


class Stepping(G4UserSteppingAction):
    def UserSteppingAction(self, step):
        vol = step.GetPreStepPoint().GetTouchable().GetVolume()   # may be None outside the world
        if vol is not None and vol.GetName() == "Block":
            edep["sum"] += step.GetTotalEnergyDeposit()
        z = step.GetPostStepPoint().GetPosition().z               # vector components are properties: .x .y .z


class Actions(G4VUserActionInitialization):
    def Build(self):
        self.SetUserAction(Generator())
        self.SetUserAction(Stepping())


run_manager = G4RunManagerFactory.CreateRunManager(G4RunManagerType.SerialOnly)
det, actions, physics = Detector(), Actions(), FTFP_BERT()   # keep references alive
run_manager.SetUserInitialization(det)
run_manager.SetUserInitialization(physics)
run_manager.SetUserInitialization(actions)
G4Random.setTheSeed(1)
run_manager.Initialize()
n = 1000
run_manager.BeamOn(n)
print({"mean_edep_MeV": edep["sum"] / n / MeV})
