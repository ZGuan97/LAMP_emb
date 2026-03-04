# AIMP_grad.py
# gradient toolkits for general AIMP
# Now compatible with AIMP2.0 and AIMP3.0
# Author: Shuoxue Li <lishuoxue@pku.edu.cn>

import pyscf.grad.rhf as rhf_grad
import pyscf.grad.rohf as rohf_grad
import pyscf.grad.roks as roks_grad
import pyscf.grad.uhf as uhf_grad
import pyscf.grad.rks as rks_grad
#Add by TengZ 2023.06.01
import pyscf.grad.uks as uks_grad
import numpy as np
from pyscf import gto
from pyscf.scf import hf, _vhf
from src import pckit2

def _ip_nucenv(mol1:gto.Mole, mol2:gto.Mole):

    nuc = 0
    for i in range(mol2.natm):
        mol1.set_rinv_origin(mol2.atom_coords()[i])
        nuc += mol2.atom_charges()[i] * mol1.intor('int1e_iprinv')
    return nuc

def _grad_elecclus_nucenv(mol1:gto.Mole, grad_mat, dm1=None):
    # Shape of gradmat: (3, nao, nao)
    
    if dm1 is None: 
        mf = hf.RHF(mol1); mf.kernel()
        dm1 = mf.make_rdm1()

    natm = mol1.natm
    aoslices = mol1.aoslice_by_atom()

    grad = np.zeros_like(mol1.atom_coords())
    for atm_id in range(natm):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        gm_cut = grad_mat[:,p0:p1,:]
        dm_cut = dm1[:,p0:p1]
        grad[atm_id] += 2. * np.einsum('xij,ji->x', gm_cut, dm_cut)
    return grad

def _grad_nucclus_elecenv(mol1:gto.Mole, mol2:gto.Mole, dm2=None):

    if dm2 is None:
        mf = hf.RHF(mol2); mf.kernel()
        dm2 = mf.make_rdm1()

    natm = mol1.natm
    grad = np.zeros_like(mol1.atom_coords())

    for atm_id in range(natm):
        mol2.set_rinv_origin(mol1.atom_coords()[atm_id])
        gradmat = - mol1.atom_charges()[atm_id] * mol2.intor('int1e_iprinv')
        grad[atm_id] += 2 * np.einsum('xij,ji->x', gradmat, dm2)

    return grad

def _get_jk(mol1:gto.Mole, mol2:gto.Mole, dm2=None):
    if dm2 is None:
        mf = hf.RHF(mol2); mf.kernel()
        dm2 = mf.make_rdm1()

    mol12 = mol1 + mol2
    nb1 = mol1.nbas
    nb2 = mol2.nbas

    intor = mol12._add_suffix('int2e_ip1')

    slice_j = (0, nb1, 0, nb1, nb1, nb1+nb2, nb1, nb1+nb2)
    slice_k = (0, nb1, nb1, nb1+nb2, nb1, nb1+nb2, 0, nb1)

    j = - _vhf.direct_mapdm(intor, 's2kl', 'lk->s1ij', dm2, 3, 
    mol12._atm, mol12._bas, mol12._env, shls_slice=slice_j)

    k = - _vhf.direct_mapdm(intor, 's1', 'jk->s1il', dm2, 3,
    mol12._atm, mol12._bas, mol12._env, shls_slice=slice_k)

    return j, k

def _grad_elecclus_elecenv(mol1:gto.Mole, mol2:gto.Mole, dm1=None, dm2=None):
    j, k = _get_jk(mol1, mol2, dm2)
    if dm1 is None: 
        mf1 = hf.RHF(mol1); mf1.kernel()
        dm1 = mf1.make_rdm1()

    grad = np.zeros_like(mol1.atom_coords())

    aoslices = mol1.aoslice_by_atom()
    for atm_id in range(mol1.natm):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        grad[atm_id] = \
            np.einsum('xij,ji->x', (2. * j - k)[:,p0:p1,:], dm1[:,p0:p1])

    return grad

def _grad_nucclus_nucenv(mol1:gto.Mole, mol2:gto.Mole):

    mol12 = mol1 + mol2
    return rhf_grad.grad_nuc(mol12, atmlst=range(mol1.natm)) \
        - rhf_grad.grad_nuc(mol1)

def _grad_proj_mat(mol1:gto.Mole, mol2:gto.Mole, dme):
    # use energy_weighted density matrix...
    # get gradient of projection matrix
    S = gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
    dS = gto.mole.intor_cross('int1e_ipovlp', mol1, mol2)
    dP = np.einsum('xmk,kl,nl->xmn', dS, dme, S)
    return dP

_grad_or = _grad_proj = _grad_elecclus_nucenv

# Gradient tools for batch

def grad_elecclus_nucenv(mol1:gto.Mole, mol2_lists, dm1=None):
    gradmat = 0
    
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            gradmat += _ip_nucenv(mol1, mol2)
    return _grad_elecclus_nucenv(mol1, gradmat, dm1)

def grad_proj(mol1:gto.Mole, nsur, mol2_lists, dme2_list, dm1=None):
    gradmat = 0
    for i in range(nsur):
        for mol2 in mol2_lists[i]:
            gradmat += _grad_proj_mat(mol1, mol2, dme=dme2_list[i])
    return _grad_proj(mol1, gradmat, dm1) # Has multiplied prefactor (-2)!

def grad_nucclus_elecenv(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    grad = 0
    for i in range(nsur):
        for mol2 in mol2_lists[i]:
            grad += _grad_nucclus_elecenv(mol1, mol2, dm2_list[i])
    return grad

def grad_elecclus_elecenv(mol1:gto.Mole, nsur, mol2_lists, dm2_list, dm1=None):
    grad = 0
    for i in range(nsur):
        for mol2 in mol2_lists[i]:
            grad += _grad_elecclus_elecenv(
                mol1, mol2, dm1=dm1, dm2=dm2_list[i]
                )
    return grad

def grad_nucclus_nucenv(mol1:gto.Mole, mol2_lists):
    grad = 0
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            grad += _grad_nucclus_nucenv(mol1, mol2)
    return grad

######## For regularization #####

def _grad_or_mat(mol1:gto.Mole, mol2:gto.Mole, dm2):
    S = gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
    dS = gto.mole.intor_cross('int1e_ipovlp', mol1, mol2)
    dP = -np.einsum('xmk,kl,nl->xmn', dS, dm2, S)
    return dP

def grad_or(mol1:gto.Mole, nsur, mol2_lists, dm2_list, dm1=None):
    gradmat = 0
    for i in range(nsur):
        for mol2 in mol2_lists[i]:
            gradmat += _grad_or_mat(mol1, mol2, dm2=dm2_list[i])
    return _grad_or(mol1, gradmat, dm1)

#################################

################# ECP Contributions ###################

def _ip_elecclus_ecpenv(mol1:gto.Mole, mol2_lists):
    # Notice that this is +h, but we will return (-h) in hcore!
    mol12 = mol1
    nb1 = mol1.nbas
    for mol2_list in mol2_lists:
        for mol2 in mol2_list:
            mol12 += mol2

    ecp1, ecptotal = 0., 0.
    if mol1.has_ecp():
        ecp1 = mol1.intor("ECPscalar_ipnuc", comp=3)
    if mol12.has_ecp():
        ecptotal = mol12.intor("ECPscalar_ipnuc", 
        shls_slice=(0, nb1, 0, nb1), comp=3)
    return ecptotal - ecp1

grad_elecclus_ecpenv = _grad_elecclus_nucenv

def _grad_ecpclus_elecenv(mol1:gto.Mole, mol2:gto.Mole, dm2):
    # Default: mol1 has ECP
    natm = mol1.natm
    natm2 = mol2.natm
    grad = np.zeros_like(mol1.atom_coords())
    nb2 = mol2.nbas
    mol21 = mol2 + mol1 # Notice the sequence!

    ecp_atoms_1 = set(mol1._ecpbas[:,gto.ATOM_OF])

    for atm_id in range(natm):
        with mol21.with_rinv_at_nucleus(atm_id=natm2+atm_id):
            if natm2+atm_id in ecp_atoms_1:
            # since the position of mol1 starts after mol2.
                gradmat = mol21.intor("ECPscalar_iprinv", comp=3,
                shls_slice=(0, nb2, 0, nb2))
                #print("Shape of gradient matrix: ", gradmat.shape)
                #print("Shape of dm2: ", dm2.shape)
                grad[atm_id] += 2. * np.einsum('xij,ji->x', gradmat, dm2)
    return grad

def grad_ecpclus_elecenv(mol1:gto.Mole, nsur, mol2_lists, dm2_list):
    # Default: mol1 has ECP
    grad = 0
    for i in range(nsur):
        for mol2 in mol2_lists[i]:
            dm2 = dm2_list[i]
            grad += _grad_ecpclus_elecenv(mol1, mol2, dm2)
    return grad

###################### Mar 23rd #######################

def get_hcore(mf, mol1=None):
    # mf: gradient
    # mf.base: scf
    if mol1 is None: mol1 = mf.mol
    hcore = rhf_grad.get_hcore(mol1)
    orparam = mf.base.orthoreg_param

    for i in range(mf.base.nsur):
        dm2 = mf.base.dm2_list[i]
        dme2 = mf.base.dme2_list[i]
        if isinstance(orparam, float) or isinstance(orparam, int):
            param = orparam
        elif isinstance(orparam, list) or isinstance(orparam, tuple):
            param = orparam[i]
        else:
            raise TypeError(
                "Orthogonal Parameter is not a number or a sequence!"
                )
        for mol2 in mf.base.mol2_lists[i]:
            j, k = _get_jk(mol1, mol2, dm2)
            proj = _grad_proj_mat(mol1, mol2, dme2)
            ortho_reg = _grad_or_mat(mol1, mol2, dm2)
            ecne = _ip_nucenv(mol1, mol2) #nucenv-elecclus
            hcore += j - 0.5 * k + proj + ecne + param * ortho_reg

    # Add ECP supportings Mar 23
    if mol1.has_ecp():
        hcore -= _ip_elecclus_ecpenv(mol1, mf.base.mol2_lists)

    if mf.base.with_pc2:
        hcore += pckit2.ip_nucenv_pc(
            mol1, mf.base.pcparams.coords, mf.base.pcparams.charges
            )
    return hcore

def grad_nuc(mf, mol=None, atmlst=None):
    if mol is None: mol = mf.mol
    grad = rhf_grad.grad_nuc(mol, atmlst)
    grad += grad_nucclus_nucenv(mol, mf.base.mol2_lists)
    grad += grad_nucclus_elecenv(
        mol, mf.base.nsur, mf.base.mol2_lists, mf.base.dm2_list
        )

    # Add ECP supportings <Mar 23, 2022>
    if mol.has_ecp():
        grad += grad_ecpclus_elecenv(
            mol, mf.base.nsur, mf.base.mol2_lists, mf.base.dm2_list
            )

    if mf.base.with_pc2:
        grad += pckit2.grad_nucclus_nucenv_pc(
            mol, mf.base.pcparams.coords, mf.base.pcparams.charges
            )
    return grad

# Gradient Classes


class GradientAIMPRHF(rhf_grad.Gradients):
    get_hcore = get_hcore
    grad_nuc = grad_nuc

class GradientAIMPRKS(rks_grad.Gradients):
    get_hcore = get_hcore
    grad_nuc = grad_nuc

#Add by TengZ 2023.06.01
class GradientAIMPUKS(uks_grad.Gradients):
    get_hcore = get_hcore
    grad_nuc = grad_nuc

class GradientAIMPROHF(rohf_grad.Gradients):
    get_hcore = get_hcore
    grad_nuc = grad_nuc

class GradientAIMPUHF(uhf_grad.Gradients):
    get_hcore = get_hcore
    grad_nuc = grad_nuc

class GradientAIMPROKS(roks_grad.Gradients):
    get_hcore = get_hcore
    grad_nuc = grad_nuc