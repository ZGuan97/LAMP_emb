# AIMP environment read from input file (using AIMPEnvLoader)
# Author: Shuoxue Li <lishuoxue@pku.edu.cn>
# Author: Teng Zhang <lishuoxue@pku.edu.cn>

import numpy as np
from pyscf import gto, lib
from pyscf.scf import _vhf, hf, rohf, uhf
from pyscf.dft import uks, rks, roks, gks
from src.EnvGenerator import xyz_parser
from io import StringIO
import basis_set_exchange as bse
from socutils.scf import spinor_hf
from socutils.somf import amf

def load_coeff(file_name_mo_coeff):
    # First read the first few lines to check whether complex numbers are present
    with open(file_name_mo_coeff) as f:
        head = "".join([next(f) for _ in range(5)])  # Read the first 5 lines
    if "(" in head or "j" in head or "J" in head:  #  Contains complex numbers
        with open(file_name_mo_coeff) as f:
            data = [line.replace("(", "").replace(")", "") for line in f]
        return np.loadtxt(StringIO("".join(data)), dtype=complex)
    else:  # Real numbers only
        return np.loadtxt(file_name_mo_coeff)
    
def get_mol_name(mol):
    name1 = mol.splitlines()[0]
    name2 = name1.split()[0]
    return name2

def make_rdm1(mo_coeff, nelec):
    if mo_coeff.ndim == 1:
        mo_coeff = mo_coeff[:nelec]
    elif mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[:,:nelec]

    return 2. * np.einsum('ik,jk->ij', mo_coeff, mo_coeff)

def make_rdm1e(mo_energy, mo_coeff, nelec):
    mo_energy = mo_energy[:nelec]
    if mo_coeff.ndim == 1:
        mo_coeff = mo_coeff[:nelec]
    elif mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[:,:nelec]
    return 2. * np.einsum('ik,k,jk->ij', mo_coeff, mo_energy, mo_coeff)

#Add by ZhangTeng 2023.07.21

def make_gdm1(mo_coeff, nelectron):
    mo_coeff = mo_coeff[:,:nelectron]
    return np.einsum('ik,jk->ij', mo_coeff, np.conjugate(mo_coeff))

def make_gdm1e(mo_energy, mo_coeff, nelectron):
    mo_energy = mo_energy[:nelectron]
    mo_coeff = mo_coeff[:,:nelectron]
    return np.einsum('ik,k,jk->ij', mo_coeff, mo_energy, np.conjugate(mo_coeff))


# mol1 : cluster molecule
# mol2 : environment molecule
def _get_jk(mol1:gto.Mole, mol2:gto.Mole, dm2):
    # return matrix of j and k
    mol12 = mol1 + mol2
    nb1 = mol1.nbas
    nb2 = mol2.nbas

    intor = mol12._add_suffix('int2e')

    slice_j = (0, nb1, 0, nb1, nb1, nb1+nb2, nb1, nb1+nb2)
    slice_k = (0, nb1, nb1, nb1+nb2, nb1, nb1+nb2, 0, nb1)

    j = _vhf.direct_mapdm(intor, 's4', 'lk->s1ij', dm2, 1,
    mol12._atm, mol12._bas, mol12._env, shls_slice=slice_j)
    k = _vhf.direct_mapdm(intor, 's1', 'jk->s1il', dm2, 1,
    mol12._atm, mol12._bas, mol12._env, shls_slice=slice_k)
    #print(j.shape)
    #print(k.shape)
    return (j, k)

#Add by ZhangTeng 2023.07.18
def _get_jk_GKS(mol1:gto.Mole, mol2:gto.Mole, dm2):
    # return matrix of j and k
    mol12 = mol1 + mol2
    nb1 = mol1.nbas
    nb2 = mol2.nbas
    na1 = mol1.nao
    na2 = mol2.nao
    
    slice_j = (0, nb1, 0, nb1, nb1, nb1+nb2, nb1, nb1+nb2)

    slice_k = (0, nb1, nb1, nb1+nb2, nb1, nb1+nb2, 0, nb1)

    oj = mol12.intor('int2e', shls_slice=slice_j)
    eye2 = np.eye(2)
    oj_m = np.einsum('ab,ijkl->ijakbl', eye2, oj).reshape(na1, na1, na2*2, na2*2).astype(np.complex128)

    oj_GKS = np.zeros((na1*2,na1*2,na2*2,na2*2)).astype(np.complex128)
    oj_m = oj_m.astype(np.complex128)
    oj_GKS[0:na1,0:na1,:,:] = oj_m
    oj_GKS[na1:na1*2,na1:na1*2,:,:] = oj_m
    
    
    ok = mol12.intor('int2e', shls_slice=slice_k)
    ok_m = np.einsum('ab,ijkl->ijakbl', eye2, ok)
    ok_m = ok_m.reshape(na1, na2, na2*2, na1*2).astype(np.complex128)
    
    ok_GKS = np.zeros((na1*2,na2*2,na2*2,na1*2)).astype(np.complex128)
    ok_m = ok_m.astype(np.complex128)
    ok_GKS[0:na1,0:na2,:,:] = ok_m
    ok_GKS[na1:na1*2,na2:na2*2,:,:] = ok_m
    
    
    j = np.einsum("kl,ijlk->ij", dm2, oj_GKS)
    k = np.einsum("kl,iklj->ij", dm2, ok_GKS)
    
    return (j, k)


def _get_proj(mol1:gto.Mole, mol2:gto.Mole, dme):
    # return - | \psi+i \rangle \langle \psi_i | 
    S = gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
    projmat = - np.einsum('ij,jk,lk->il', S, dme, S)
    return projmat


#Add by ZhangTeng 2023.07.18
def _get_proj_GKS(mol1:gto.Mole, mol2:gto.Mole, dme):
    # return - | \psi+i \rangle \langle \psi_i | 
    S = gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
    S_gks = np.kron(np.array([[1, 0], [0, 1]]), S)
    projmat = - np.einsum('ij,jk,lk->il', S_gks, dme, S_gks)
    return projmat



def _get_elecclus_nucenv(mol1:gto.Mole, mol2:gto.Mole):
    nuc = 0.0
    for i in range(mol2.natm):
        mol1.set_rinv_origin(mol2.atom_coords()[i])
        nuc += mol2.atom_charges()[i] * mol1.intor('int1e_rinv')
    return -nuc

#Add by ZhangTeng 2023.07.18
def _get_elecclus_nucenv_GKS(mol1:gto.Mole, mol2:gto.Mole):
    nuc = 0.0
    for i in range(mol2.natm):
        mol1.set_rinv_origin(mol2.atom_coords()[i])
        nuc += mol2.atom_charges()[i] * mol1.intor('int1e_rinv')
    nuc_GKS = np.kron(np.array([[1, 0], [0, 1]]), nuc)
    return -nuc_GKS


def _get_nucclus_nucenv(mol1:gto.Mole, mol2:gto.Mole):
    drvec = mol1.atom_coords()[:,None] - mol2.atom_coords()
    drinv = 1. / np.sqrt(np.einsum('ijk->ij', drvec**2))
    nuc_energy = np.einsum('i,ij,j->', mol1.atom_charges(), drinv, mol2.atom_charges())
    return float(nuc_energy)

def _get_nucclus_elecenv(mol1:gto.Mole, mol2:gto.Mole):
    # return matrix of environment 
    rinvmat = 0.0
    for i in range(mol1.natm):
        mol2.set_rinv_origin(mol1.atom_coords()[i])
        rinvmat += mol1.atom_charges()[i] * mol2.intor('int1e_rinv')

    return rinvmat

# Batch operations

## return matrices
def get_proj(mol1:gto.Mole, nsur, mol2_lists, dme2_list):
    f = 0
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dme2 = dme2_list[i]
        for mol2 in mol2_list:
            f += _get_proj(mol1, mol2, dme2)
    return f

#Add by ZhangTeng 2023.07.18
def get_proj_GKS(mol1:gto.Mole, nsur, mol2_lists, dme2_list):
    f = 0 + 0j
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dme2 = dme2_list[i]
        for mol2 in mol2_list:
            f += _get_proj_GKS(mol1, mol2, dme2)
    return f

def get_jk(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    j = 0.0
    k = 0.0
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]
        for mol2 in mol2_list:
            dj, dk = _get_jk(mol1, mol2, dm2)
            j += dj; k += dk
    return (j, k)

#Add by ZhangTeng 2023.07.18
def get_jk_GKS(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    j = 0.0 + 0.0j
    k = 0.0 + 0.0j
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]
        for mol2 in mol2_list:
            dj, dk = _get_jk_GKS(mol1, mol2, dm2)
            j += dj; k += dk
    return (j, k)

def get_elecclus_nucenv(mol1:gto.Mole, mol2_lists):
    f = 0
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            f += _get_elecclus_nucenv(mol1, mol2)
    return f

#Add by ZhangTeng 2023.07.18
def get_elecclus_nucenv_GKS(mol1:gto.Mole, mol2_lists):
    f = 0
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            f += _get_elecclus_nucenv_GKS(mol1, mol2)
    return f

## return scalars
def get_nucclus_nucenv(mol1:gto.Mole, mol2_lists):
    e = 0
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            e += _get_nucclus_nucenv(mol1, mol2)
    return e

def get_nucclus_elecenv(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    e = 0.0
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]
        rinvmat = 0.0
        for mol2 in mol2_list:
            rinvmat += _get_nucclus_elecenv(mol1, mol2)
        e += np.einsum('ij,ji->', dm2, rinvmat)
    return - e

#Add by ZhangTeng 2023.7.21
def get_nucclus_elecenv_GKS(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    e = 0.0 + 0.0j
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]
        rinvmat = 0.0
        for mol2 in mol2_list:
            rinvmat += _get_nucclus_elecenv(mol1, mol2)
        rinvmat_GKS = np.kron(np.array([[1, 0], [0, 1]]), rinvmat)
        e += np.einsum('ij,ji->', dm2, rinvmat_GKS)
    return - e










#######################################
# Added by Shuoxue Li (Mar 6, 2022)
# Orthogonality regularization

def _get_ortho_reg(mol1:gto.Mole, mol2:gto.Mole, dm2):
    S = gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
    regmat = np.einsum('ij,jk,lk->il', S, dm2, S)
    return regmat

#Add by ZhangTeng 2023.07.18
def _get_ortho_reg_GKS(mol1:gto.Mole, mol2:gto.Mole, dm2):
    S = gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
    S_gks = np.kron(np.array([[1, 0], [0, 1]]), S)
    regmat = np.einsum('ij,jk,lk->il', S_gks, dm2, S_gks)
    return regmat


def get_ortho_reg(mol1:gto.Mole, nsur, mol2_lists, dm2_list, coef=0.5):
    f = 0
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]

        # the coefficient sequence equals to charge dictionary in `load_env_from_xyz`
        if isinstance(coef, list) or isinstance(coef, tuple):
            coefi = coef[i]
        elif isinstance(coef, float) or isinstance(coef, int):
            coefi = coef

        for mol2 in mol2_list:
            f += coefi * _get_ortho_reg(mol1, mol2, dm2)
    return f

#Add by ZhangTeng 2023.07.18

def get_ortho_reg_GKS(mol1:gto.Mole, nsur, mol2_lists, dm2_list, coef=0.5):
    f = 0 + 0j
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]

        # the coefficient sequence equals to charge dictionary in `load_env_from_xyz`
        if isinstance(coef, list) or isinstance(coef, tuple):
            coefi = coef[i]
        elif isinstance(coef, float) or isinstance(coef, int):
            coefi = coef

        for mol2 in mol2_list:
            f += coefi * _get_ortho_reg_GKS(mol1, mol2, dm2)
    return f



# 
################# ECP Contributions ###################

def get_elecclus_ecpenv(mol1:gto.Mole, mol2_lists):
    # shape (mol1.nao, mol1.nao)
    mol12 = mol1
    nb1 = mol1.nbas
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            mol12 += mol2
    
    ecp1, ecptotal = 0., 0.
    if mol1.has_ecp():
        ecp1 = mol1.intor_symmetric("ECPscalar")
    if mol12.has_ecp():
        ecptotal = mol12.intor("ECPscalar", shls_slice=(0, nb1, 0, nb1))
    return ecptotal - ecp1 # reserve all the extra contributions to cluster electrons

def _get_ecpclus_elecenv(mol1:gto.Mole, mol2:gto.Mole):
    # shape (mol2.nao, mol2.nao)
    mol21 = mol2 + mol1
    nb2 = mol2.nbas
    ecp2 = 0
    if mol2.has_ecp():
        ecp2 = mol2.intor_symmetric("ECPscalar")
    ecptotal = mol21.intor("ECPscalar", shls_slice=(0, nb2, 0, nb2))
    return ecptotal - ecp2

def get_ecpclus_elecenv(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    # Scalar, energy contribution from cluster ECP potential to electrons in environments.
    e = 0.
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]
        f = 0.
        for mol2 in mol2_list:
            f += _get_ecpclus_elecenv(mol1, mol2)
        e += np.einsum("ij,ji->", dm2, f)
    return e

#Add By ZhangTeng 2023.7.21
def get_ecpclus_elecenv_GKS(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    # Scalar, energy contribution from cluster ECP potential to electrons in environments.
    e = 0.0 + 0.0j
    for i in range(nsur):
        mol2_list = mol2_lists[i]
        dm2 = dm2_list[i]
        f = 0.0
        for mol2 in mol2_list:
            f += _get_ecpclus_elecenv(mol1, mol2)
        f_GKS = np.kron(np.array([[1, 0], [0, 1]]), f)
        e += np.einsum("ij,ji->", dm2, f_GKS)
    return e


###################### Mar 23rd #######################

def get_hcore(mf, mol=None):
    if mol is None: mol = mf.mol
    h0 = hf.get_hcore(mol = mol)
    j, k = get_jk(mol, mf.nsur, mf.mol2_lists, mf.dm2_list)
    proj = get_proj(mol, mf.nsur, mf.mol2_lists, mf.dme2_list)
    nuc = get_elecclus_nucenv(mol, mf.mol2_lists)
    orthoreg = get_ortho_reg(mol, mf.nsur,
    mf.mol2_lists, mf.dm2_list, mf.orthoreg_param)
    h = h0 + proj + nuc + j - .5 * k + orthoreg

    # Adding ECP supportings Mar 23
    if mol.has_ecp():
        h += get_elecclus_ecpenv(mol, mf.mol2_lists)

    # Adding Point Charge ...
    if mf.with_pc2:
        from src import pckit2
        hpc = pckit2.get_elecclus_nucenv_pc(
            mol, mf.pcparams.coords, mf.pcparams.charges)
        h += hpc

    return h

#Add By ZhangTeng 2023.7.21
def get_hcore_GKS(mf, mol=None):
    if mol is None: mol = mf.mol
    #h0_RKS = hf.get_hcore(mol = mol)
    #h0 = np.kron(np.array([[1, 0], [0, 1]]), h0_RKS)
    h0=mf.mf_2.get_hcore()
    j, k = get_jk_GKS(mol, mf.nsur, mf.mol2_lists, mf.dm2_list)
    proj = get_proj_GKS(mol, mf.nsur, mf.mol2_lists, mf.dme2_list)
    nuc = get_elecclus_nucenv_GKS(mol, mf.mol2_lists)
    orthoreg = get_ortho_reg_GKS(mol, mf.nsur,
    mf.mol2_lists, mf.dm2_list, mf.orthoreg_param)
    h = h0 + 2 * proj + nuc + j - k + 2 * orthoreg

    # Adding ECP supportings Mar 23
    if mol.has_ecp():
        elecclus_ecpenv = get_elecclus_ecpenv(mol, mf.mol2_lists)
        elecclus_ecpenv_GKS = np.kron(np.array([[1, 0], [0, 1]]), elecclus_ecpenv)
        h += elecclus_ecpenv_GKS

    # Adding Point Charge ...
    if mf.with_pc2:
        from src import pckit2
        hpc = pckit2.get_elecclus_nucenv_pc(
            mol, mf.pcparams.coords, mf.pcparams.charges)
        hpc_GKS = np.kron(np.array([[1, 0], [0, 1]]), hpc)
        h += hpc_GKS

    return h


def energy_nuc(mf):
    enuc = mf.mol.energy_nuc()
    enuc += get_nucclus_elecenv(mf.mol, mf.nsur, mf.mol2_lists, mf.dm2_list)

    # Add ECP supportings Mar 23
    if mf.mol.has_ecp():
        enuc += get_ecpclus_elecenv(mf.mol, mf.nsur, mf.mol2_lists, mf.dm2_list)

    nuc_nuc = get_nucclus_nucenv(mf.mol, mf.mol2_lists)
    enuc += nuc_nuc

    if mf.with_pc2:
        from src import pckit2
        enuc += pckit2.get_nucclus_nucenv_pc(mf.mol, mf.pcparams.coords, mf.pcparams.charges)
    return enuc

#Add by ZhangTeng 2023.7.21
def energy_nuc_GKS(mf):
    enuc = mf.mol.energy_nuc()
    enuc = np.complex128(enuc)
    enuc += get_nucclus_elecenv_GKS(mf.mol, mf.nsur, mf.mol2_lists, mf.dm2_list)

    # Add ECP supportings Mar 23
    if mf.mol.has_ecp():
        enuc += get_ecpclus_elecenv_GKS(mf.mol, mf.nsur, mf.mol2_lists, mf.dm2_list)

    nuc_nuc = get_nucclus_nucenv(mf.mol, mf.mol2_lists)
    enuc += nuc_nuc

    if mf.with_pc2:
        from src import pckit2
        enuc += pckit2.get_nucclus_nucenv_pc(mf.mol, mf.pcparams.coords, mf.pcparams.charges)
    return np.real(enuc)









# AIMP Environment Loader
class AIMPEnvLoader:
    '''
    A loader for AIMP environments.
    '''
    def __init__(self, inputf, socdict=False):
        if isinstance(inputf, str):
            if inputf[-5:] == ".json":
                import json
                with open(inputf, "r") as f:
                    inputdict = json.load(f)
            elif inputf[-5:] == ".yaml":
                import yaml
                with open(inputf, "r") as f:
                    inputdict = yaml.safe_load(f)
            else:
                raise NotImplementedError(
                    "AIMPEnvLoader: Haven't supported other formats yet!"
                    )
        elif isinstance(inputf, dict): inputdict = inputf
        else: 
            raise TypeError(
                "AIMPEnvLoader: Only support string or dictionary input!"
                )

        self.envfile = inputdict["dir"]     # directory of environment file
        self.workdir = inputdict['workdir'] # working directory
        map = {}
        try:
            map_prim = inputdict["mapping"]
            for equatm, orgf, coord in map_prim:  #equatim means fake atom orgf is file path 
                map[equatm] = {
                    "mapfile": orgf,
                    "equcoord": np.array(coord)
                }
        except: pass
        self.map = map

        self.chargedict = inputdict["charge"]

        try: self.basis = inputdict["basis"]
        except KeyError: self.basis = "sto-3g"

        try: self.ecp = inputdict["ecp"]
        except KeyError: self.ecp = {}

        try: self.scf = inputdict["scf"]
        except KeyError: self.scf = {"calc": "hf"}
        
        #Add by ZhangT 2023.7.12
        try: self.soc = socdict
        except KeyError: self.soc = False

        self.mollst = []
        self.mollst_for_SCEI = [] #To save fake name
        self.dm2lst = []
        self.dme2lst = []
        self.xyzstr = ""

        self.totcoordlst = None

        self.get_mol_list()
        self.get_dm_list()
        self.save_env_xyz()
    
    def get_mol_list(self):
        natm, dicts, lines = xyz_parser(self.envfile)
        count = 0
        #scan atom in charges in input file
        text_newcoord_allmol2 = []
        for atm in self.chargedict.keys():
            #mollst form [[],[],[]...] the number of sublists is the number of atoms in charges in input file
            self.mollst.append([])
            self.mollst_for_SCEI.append([])

            charge = self.chargedict[atm]
            #scan atom in aimp.xyz
            for i in range(natm):
                atmstr = dicts[i]['atom']
                coord = dicts[i]['coordinate']
                line = lines[i]
                #The following if decides the difference between line and atomline
                #atm from charge, atmstr from xyz file, run following code only if atmstr is atm
                if atmstr == atm:
                    # read basis set see if it's str (from all atom) or dict (from specific atom)
                    if isinstance(self.basis, str): bs = self.basis
                    elif isinstance(self.basis, dict): bs = self.basis[atm]

                    # read ecp
                    if isinstance(self.ecp, str): 
                        if self.ecp[-4:] in [".dat", ".txt"]:
                            with open(self.ecp, 'r') as f:
                                ecp = f.read()
                        else: ecp = self.ecp
                    #ZhangTeng fix this bug 2024_9_12
                    elif isinstance(self.ecp, dict):
                        try: 
                            ecp = self.ecp[atm]
                            if isinstance(ecp, str) and ecp[-4:] in [".dat", ".txt"]:
                                with open(ecp, 'r') as f:
                                    ecp = f.read()
                            elif isinstance(ecp, dict):
                                ecp = ecp
                            else: 
                                ecp = ecp
                        except KeyError: ecp = {}

                    # read atom
                    if atmstr in self.map.keys():
                        fake_name = atmstr
                        equcoord = self.map[atmstr]["equcoord"]
                        mapfile = self.workdir + self.map[atmstr]["mapfile"]
                        _natm, _dicts, _lines = xyz_parser(mapfile)

                        atomline = ""
                        for j in range(_natm):
                            _atmstr = _dicts[j]['atom']
                            _coord = _dicts[j]['coordinate']
                            newcoord = _coord - equcoord + coord
                            _x, _y, _z = newcoord
                            atomline += "{} {:.8f} {:.8f} {:.8f} \n".format(
                                _atmstr, _x, _y, _z
                            )
                    else:
                        atomline = line
                        fake_name = atmstr

                    text_newcoord_allmol2.append(atomline.strip())

                    # Added for custom basis sets in the NWChem format
                    if "parse" in bs or 'load' in bs:
                        bs = eval(bs)

                    mol = gto.M(atom=atomline, basis=bs, ecp=ecp, charge=charge)
                        #mol.verbose = 0
                    #We need to save fakename of mapping atoms
                    self.mollst[count].append(mol)
                    self.mollst_for_SCEI[count].append({'mol': mol, 'fake_name': fake_name})
                    '''
                    So, after first cycle, all same kind mols correspoding to 
                    the first atom in charge are add to the first sublist in
                    mollist
                    '''
            count += 1
        np.savetxt("newcoord_aimp.xyz", text_newcoord_allmol2, fmt='%s')


    #This is how mollst become mol2list
    def save_env_xyz(self):
        """
        Output xyz file for the AIMP environment,
        help double check the correctness of environement building.
        """
        natm = 0
        xyzstr = ""
        for mol2_list in self.mollst:
            #mol2_list is the list that contain all the same mols in aimp.xyz
            for mol2 in mol2_list:
                natm += mol2.natm
                molstr = mol2.atom
                if molstr[-1] != "\n": molstr += "\n"
                xyzstr += molstr
        xyzstr = "{}\n{}\n".format(natm, "Generated from Python") + xyzstr

        from src.EnvGenerator import xyz2coords
        self.totcoordlst = xyz2coords(xyzstr)
        self.xyzstr = xyzstr

    def write_env_xyz(self, fileo):
        with open(fileo, "w") as f:
            f.write(self.workdir + self.xyzstr)

    def get_dm_list(self): 
        for mol2_list in self.mollst_for_SCEI:
            mol2_info = mol2_list[0] #mol2_info is dict
            mol2 = mol2_info['mol']
            fake_name = mol2_info.get('fake_name')
            #We only need the first one of mol2list since all mols in it are same
            nelec = mol2.nelec[0]
            #In order to get mol2 name, we need to check if mol2 is from mapping
            mol2_atom = mol2.atom
            mol2_atom_lines = mol2_atom.splitlines()
            len_lines = len(mol2_atom_lines)
            if len_lines==1:
                mol2name = get_mol_name(mol2.atom)
                print(mol2name + " is load")
            else:
                mol2name = fake_name
                print(mol2name + " is load")
            nelectron = mol2.nelectron

            file_name_mo_coeff = 'MO_COEFF_' + mol2name + '.txt'
            file_name_mo_energy = 'MO_ENERGY_' + mol2name + '.txt'

            mo_energy = np.loadtxt(file_name_mo_energy)
            mo_coeff = load_coeff(file_name_mo_coeff)

            if self.soc:
                dm = make_gdm1(mo_coeff, nelectron)
                dme = make_gdm1e(mo_energy, mo_coeff, nelectron)
            else:
                dm = make_rdm1(mo_coeff, nelec)
                dme = make_rdm1e(mo_energy, mo_coeff, nelec)
            self.dm2lst.append(dm)
            self.dme2lst.append(dme)

# General AIMP Class
class AIMPMixin:
    def __init__(self, mol1:gto.Mole, loader:AIMPEnvLoader):
        
        self.mol = mol1
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst

        # orthogonality regularization parameter
        self.orthoreg_param = 0.0

        self.with_pc2 = False       
        self.pcparams = None

    def addPCParam2(self, pcParam):
        self.with_pc2 = True
        self.pcparams = pcParam

    def set_orthoreg_param(self, coef=0.5): # orthogonality
        self.orthoreg_param = coef

# AIMP branches (RHF/ROHF/UHF/RKS)
class AIMP_RHF(hf.RHF, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader): 
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        hf.RHF.__init__(self, mol1)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore
    energy_nuc = energy_nuc

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPRHF
        return GradientAIMPRHF(self)

class AIMP_ROHF(rohf.ROHF, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader):
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        hf.RHF.__init__(self, mol1)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore
    energy_nuc = energy_nuc

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPROHF
        return GradientAIMPROHF(self)

class AIMP_UHF(uhf.UHF, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader):
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        hf.RHF.__init__(self, mol1)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore
    energy_nuc = energy_nuc

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPUHF
        return GradientAIMPUHF(self)

class AIMP_RKS(rks.RKS, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader, xc="B3LYP"):
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        rks.RKS.__init__(self, mol1, xc=xc)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore
    energy_nuc = energy_nuc

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPRKS
        return GradientAIMPRKS(self)

#Add by TengZ 2023.06.01
class AIMP_UKS(uks.UKS, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader, xc="B3LYP"):
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        uks.UKS.__init__(self, mol1, xc=xc)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore
    energy_nuc = energy_nuc

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPUKS
        return GradientAIMPUKS(self)

#Add by TengZ 2023.07.12
class AIMP_GKS(gks.GKS, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader, xc="B3LYP"):
        mf_2 = gks.GKS(mol1, xc=xc).x2c1e().density_fit()
        self.__dict__.update(mf_2.__dict__)
        self.mf_2=mf_2
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        gks.GKS.__init__(self, mol1, xc=xc)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore_GKS
    energy_nuc = energy_nuc_GKS

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPUKS
        return GradientAIMPUKS(self)


class AIMP_ROKS(roks.ROKS, AIMPMixin):
    def __init__(self, mol1:gto.Mole, loader, xc='B3LYP'):
        self.pcparams = None
        self.mol2_lists = loader.mollst
        self.nsur = len(self.mol2_lists)
        self.dm2_list = loader.dm2lst
        self.dme2_list = loader.dme2lst
        self.with_pc2 = False
        self.orthoreg_param = 0.5
        roks.ROKS.__init__(self, mol1, xc=xc)
        AIMPMixin.__init__(self, mol1, loader)

    get_hcore = get_hcore
    energy_nuc = energy_nuc

    def nuc_grad_method(self):
        from src.AIMP_grad import GradientAIMPROKS
        return GradientAIMPROKS(self)
    