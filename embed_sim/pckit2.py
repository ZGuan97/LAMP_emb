# Second Generation of pckit
# Using code of Huaiyang Sun

from pyscf import gto
import numpy as np
from pyscf.scf import hf
from pyscf.dft import rks
from src.AIMP3 import AIMPEnvLoader
Ang2Bohr = 1.8897259886

# energies
def get_elecclus_nucenv_pc(mol1:gto.Mole, coords, charges): 
    # coord is set to Bohr
    nuc = 0
    for i in range(len(coords)):
        coord = coords[i]
        charge = charges[i]
        mol1.set_rinv_origin(coord)
        nuc += charge * mol1.intor('int1e_rinv')
    return -nuc

def get_nucclus_nucenv_pc(mol1:gto.Mole, coords, charges):
    # coords: all the coordinate of environment
    drvec = mol1.atom_coords()[:,None] - coords
    drinv = 1. / np.sqrt(np.einsum('ijk->ij', drvec ** 2))
    nuc_energy = np.einsum('i,ij,j->', mol1.atom_charges(), drinv, charges)
    return float(nuc_energy)


# gradients
def ip_nucenv_pc(mol1:gto.Mole, coords, charges):
    nuc = 0
    for i in range(len(coords)):
        coord = coords[i]
        charge = charges[i]
        mol1.set_rinv_origin(coord)
        nuc += charge * mol1.intor('int1e_iprinv')
    return nuc

def grad_nucclus_nucenv_pc(mol1:gto.Mole, coords, charges):
    drvec = mol1.atom_coords()[:,None] - coords
    drinvcube = np.einsum('ijx->ij', drvec**2) ** (-1.5)
    rinvgrad = - np.einsum('ijx,ij->ijx', drvec, drinvcube)
    grad = np.einsum('ijx,i,j->ix', rinvgrad, mol1.atom_charges(), charges)
    return grad

def grad_elecclus_nucenv_pc(mol1:gto.Mole, coords, charges, dm1=None):
    from src import AIMP_grad
    gradmat = ip_nucenv_pc(mol1, coords, charges)
    return AIMP_grad._grad_elecclus_nucenv(mol1, gradmat, dm1)

# Point-charge parameter class
class PointChargeParams:
    def __init__(self, coordf=None, chargef=None):
        # Read from xyz and charge files
        from src.EnvGenerator import xyz2coords
        coords = None
        charges = None
        if coordf is not None: coords = xyz2coords(coordf)
        if chargef is not None: charges = np.loadtxt(chargef)
        self.coords = coords
        self.charges = charges

    def __add__(self, other): # we can mingle two pcParams by '+'.
        coords = np.vstack((self.coords, other.coords))
        charges = np.hstack((self.charges, other.charges))

        newpc = PointChargeParams()
        newpc.coords = coords
        newpc.charges = charges
        return newpc

# Organic-Inorganic Point Charge Generator
class OrganicPCLoader(AIMPEnvLoader):
    def __init__(self, inputf):
        super().__init__(inputf)
        self.chglst = []        # charge list for equivalent molecules
        self.totchglst = None   # total charge list

        self.get_chg_list()
        self.save_chg_list()

    def get_chg_list(self):
        nsur = len(self.dm2lst)
        for j in range(nsur):
            mol2 = self.mollst[j][0]
            dm2 = self.dm2lst[j]
            chg = hf.mulliken_pop(mol2, dm2, verbose=0)[1]
            self.chglst.append(chg)

    def save_chg_list(self):
        totchglst = []
        nsur = len(self.dm2lst)
        for j in range(nsur):
            chglst = self.chglst[j]
            mol2_lst = self.mollst[j]
            for mol2 in mol2_lst: totchglst.extend(chglst)
        self.totchglst = np.array(totchglst)

    def write_chg_list(self, fileo):
        np.savetxt(self.workdir + fileo, self.totchglst)

    def make_param(self):
        pcparam = PointChargeParams()
        pcparam.charges = self.totchglst
        pcparam.coords = self.totcoordlst
        self.write_chg_list("_temp_chglst.dat")
        self.write_env_xyz("_temp_realxyz.xyz")
        return pcparam
