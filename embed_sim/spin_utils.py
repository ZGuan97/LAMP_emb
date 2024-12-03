import numpy as np
from scipy.special import comb

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

def unpack_nelec(nelec, spin=None):
    # from pyscf/fci/addons.py
    if spin is None:
        spin = 0
    else:
        nelec = int(np.sum(nelec))
    if isinstance(nelec, (int, np.number)):
        nelecb = (nelec-spin)//2
        neleca = nelec - nelecb
        nelec = neleca, nelecb
    return nelec

# For useful spin operators and Hamiltonians, visit documentation of EasySpin
# https://easyspin.org/easyspin/documentation/index.html
# Or sympy.physics.quantum.spin
# https://docs.sympy.org/latest/modules/physics/quantum/spin.html#module-sympy.physics.quantum.spin

_spin_operator = []
# S=0
_spin_operator.append(np.zeros((3, 1, 1)))
# S=1/2
S_1_2_op = np.array([[[0, 1/2],
                      [1/2, 0]],
                     [[0, 1/2j],
                      [-1/2j, 0]],
                     [[1/2, 0],
                      [0, -1/2]],
                     ])
_spin_operator.append(S_1_2_op)

# S=1
S_1_op = np.array([[[0, 1/np.sqrt(2), 0],
                    [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                    [0, 1/np.sqrt(2), 0]],
                   [[0, 1/(np.sqrt(2)*1j), 0],
                    [-1/(np.sqrt(2)*1j), 0, 1/(np.sqrt(2)*1j)],
                    [0, -1/(np.sqrt(2)*1j), 0]],
                   [[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1]],
                   ])
_spin_operator.append(S_1_op)

# S=3/2
S_3_2_op = np.array([[[0, np.sqrt(3)/2, 0, 0],
                      [np.sqrt(3)/2, 0, 1, 0],
                      [0, 1, 0, np.sqrt(3)/2], 
                      [0, 0, np.sqrt(3)/2, 0]],
                     [[0, np.sqrt(3)/2j, 0, 0],
                      [-np.sqrt(3)/2j, 0, -1j, 0],
                      [0, 1j, 0, np.sqrt(3)/2j], 
                      [0, 0, -np.sqrt(3)/2j, 0]],
                     [[3/2, 0, 0, 0],
                      [0, 1/2, 0, 0],
                      [0, 0, -1/2, 0],
                      [0, 0, 0, -3/2]],
                     ])
_spin_operator.append(S_3_2_op)

# S=2
S_2_op = np.array([[[0, 1, 0, 0, 0],
                    [1, 0, np.sqrt(6)/2, 0, 0],
                    [0, np.sqrt(6)/2, 0, np.sqrt(6)/2, 0],
                    [0, 0, np.sqrt(6)/2, 0, 1],
                    [0, 0, 0, 1, 0]],
                   [[0, -1j, 0, 0, 0],
                    [1j, 0, np.sqrt(6)/2j, 0, 0],
                    [0, -np.sqrt(6)/2j, 0, np.sqrt(6)/2j, 0],
                    [0, 0, -np.sqrt(6)/2j, 0, -1j],
                    [0, 0, 0, 1j, 0]],
                   [[2, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0],
                    [0, 0, 0, 0, -2]],
                   ])
_spin_operator.append(S_2_op)

# S=5/2
S_5_2_op = np.array([[[0, np.sqrt(5)/2, 0, 0, 0, 0],
                      [np.sqrt(5)/2, 0, np.sqrt(2), 0, 0, 0],
                      [0, np.sqrt(2), 0, 3/2, 0, 0],
                      [0, 0, 3/2, 0, np.sqrt(2), 0],
                      [0, 0, 0, np.sqrt(2), 0, np.sqrt(5)/2],
                      [0, 0, 0, 0, np.sqrt(5)/2, 0]],
                     [[0, np.sqrt(5)/2j, 0, 0, 0, 0],
                      [-np.sqrt(5)/2j, 0, -np.sqrt(2)*1j, 0, 0, 0],
                      [0, np.sqrt(2)*1j, 0, -3/2*1j, 0, 0],
                      [0, 0, 3/2*1j, 0, -np.sqrt(2)*1j, 0],
                      [0, 0, 0, np.sqrt(2)*1j, 0, np.sqrt(5)/2j],
                      [0, 0, 0, 0, -np.sqrt(5)/2j, 0]],
                     [[5/2, 0, 0, 0, 0, 0],
                      [0, 3/2, 0, 0, 0, 0],
                      [0, 0, 1/2, 0, 0, 0],
                      [0, 0, 0, -1/2, 0, 0],
                      [0, 0, 0, 0, -3/2, 0],
                      [0, 0, 0, 0, 0, -5/2]],
                     ])
_spin_operator.append(S_5_2_op)

def _high_order_spin_operator(spin):
    S_op = np.zeros((3, spin+1, spin+1), dtype=complex)
    # S_op[2] = np.diag(np.arange(spin/2, -spin/2-1, -1))
    for im1 in range(0, spin+1):
        for im2 in range(0, spin+1):
            m1 = -im1 + spin/2 
            m2 = -im2 + spin/2 
            if m1 == m2:
                S_op[2, im1, im2] = m1
            elif m1 == m2 + 1:
                S_op[0, im1, im2] = 1/2 * np.sqrt(spin/2*(spin/2+1) - m1 * m2)
                S_op[1, im1, im2] = 1/2j * np.sqrt(spin/2*(spin/2+1) - m1 * m2)
            elif m1 == m2 - 1:
                S_op[0, im1, im2] = 1/2 * np.sqrt(spin/2*(spin/2+1) - m1 * m2)
                S_op[1, im1, im2] = - 1/2j * np.sqrt(spin/2*(spin/2+1) - m1 * m2)
    return S_op

def spin_operator(spin, direction=None):
    # here spin=2S as in PySCF
    # at most S=5/2
    if direction is not None:
        try:
            return _spin_operator[spin, direction]
        except IndexError:
            return _high_order_spin_operator(spin)[direction]
    else:
        try:
            return _spin_operator[spin]
        except IndexError:
            return _high_order_spin_operator(spin)

def ZFS_Hamiltonian(D_mat, spin):
    # For explicit matrices, see Appendix C of
    # C. de Graaf and R. Broer, Magnetic Interactions in Molecules and Solids (Springer, Cham Heidelberg New York Dordrecht London, 2016).
    # S=1 part is wrong
    # all the Hamiltonians are up to a complex conjugate(the spin operate in Appendix B is also wrong!)
    return np.einsum('amp, ab, bpn->mn', spin_operator(spin), D_mat, spin_operator(spin))

def Zeeman_Hamiltonian(g_mat, mag_field, spin):
    return np.einsum('a, ab, bmn->mn', mag_field, g_mat, spin_operator(spin))
