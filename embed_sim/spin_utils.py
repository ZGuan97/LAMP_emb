import numpy as np
from math import comb

def Weyl_nstate(ncas, nelecas, spin): # definition of S is the same as that in pyscf.mol, equal to 2S actually
    if (nelecas - spin)%2 == 0: 
        nstate = (spin+1)/(ncas+1) * comb(ncas+1, round(nelecas/2 - spin/2)) * comb(ncas+1, round(nelecas/2 + spin/2 + 1))
    else:
        nstate = 0
    return round(nstate)

def gen_statelis(ncas, nelecas):
    # ncas: number of active orbitals
    # nelecas: number of electron in active space
    nelecas = np.sum(nelecas)
    Smax = min(nelecas, 2*ncas-nelecas) # definition of S is same as that in molecule, equal to 2S actually
    statelis = np.array([Weyl_nstate(ncas, nelecas, ispin) for ispin in range(0, Smax+1)])
    return statelis

def spin_operator(spin, direction):
    return 

def ZFS_Hamiltonian(D_mat, spin):
    return 

def Zeeman_Hamiltonian(g_mat, mag_field, spin):
    return
