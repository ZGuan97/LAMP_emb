# checking gradients by comparing with numerical value
# Author: Shuoxue Li < 1800011839@pku.edu.cn >

from pyscf import gto
import numpy as np


def _grad_num(mol_clus:gto.Mole, dim, n_atm, f, dr=1E-4, *args, **kwargs):
    ptr = mol_clus._atm[n_atm,1] + dim
    mol_clus._atom[n_atm][1][dim] += dr
    mol_clus._env[ptr] += dr

    Ep = f(mol_clus, *args, **kwargs)
    mol_clus._atom[n_atm][1][dim] -= dr
    mol_clus._env[ptr] -= 2. * dr

    En = f(mol_clus, *args, **kwargs)

    mol_clus._atom[n_atm][1][dim] += dr
    mol_clus._env[ptr] += dr

    return (Ep - En) / (2 * dr)

def grad_check(mol_clus:gto.Mole, func, grad_anal, dr=1E-4, verbose=False, tol=1E-5, *args, **kwargs):
    # Func must have the type of f(mol_clus, *args, **kwargs)

    grad_num = np.zeros_like(grad_anal)
    for n_atm in range(mol_clus.natm):
        for dim in range(3):
            grad_num[n_atm, dim] = _grad_num(mol_clus, dim, n_atm, func, dr, *args, **kwargs)
    grad = grad_num - grad_anal
    dgradnorm = np.linalg.norm(grad)
    print("Autotest {} ; |dgrad| = {}".format(np.abs(dgradnorm) < tol, dgradnorm))
    if verbose:
        print("numerical gradient: ", grad_num)
        print("analytical gradient: ", grad_anal)
