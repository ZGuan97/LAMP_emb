#!/usr/bin/env python3

"""
Generate given number of random sites in the unit cell for the electrostatic
potential fit. For sites that are too close to any atoms, we discard them.
The sites are given as fractional coordinates.
"""

import numpy as np
from itertools import product
from ase.io import read
from .neighborTools import neighbors

def distance_from_fracs(x1, x2, cell):
    """
    Calculate minimum distance (i.e., including images) from two fractional coordinates x1 and x2.
    :param cell: 3x3 2d array. Cell vectors. Each raw is a lattice vector.
    """
    d_min = np.inf
    Lmax = 2
    d_frac = (x2 - x1) % 1.0    # first we reduce the fractional coordinate distance to the unit cell at origin
    for a, b, c in product(range(-Lmax, Lmax+1), repeat=3):
        disp = np.matmul(d_frac + np.array([a, b, c], dtype=np.float64), cell)
        d_min = min(d_min, np.linalg.norm(disp))
    return d_min

def gen_random_sites(poscarfile, num = 100, deps = 0.02, atom = None, cAtom = None, cAtomIndex = None):
    """
    Generate given number (num) of random sites in the unit cell (poscarfile).
    Sites that are too close to already generated sites and atoms (threshold: deps Angstrom) will be discarded.
    Sites are returned as a 2D array, each raw of which is a fractional coordinate.
    
    in:
        atom: int
            index of the central atom (starts from 0)
        cAtom: str
            central atom label
        cAtomIndex: int
            central atom index
        For example, we can generate random sites in a unit cell centered at the 1st La atom
    
    Note that still two ways of specifying the central atom is provided. atom or cAtom + cAtomIndex
    """
    mol = read(poscarfile)
    cell = mol.get_cell()
    fracs = mol.get_scaled_positions(wrap=True)
    n_pass, n_fail = 0, 0
    sites = []
    # calculate the fractional coordinates of the center
    if not (atom is None and cAtom is None and cAtomIndex is None):
        nbs = neighbors(poscarfile, atom=atom, cAtom=cAtom, cAtomIndex=cAtomIndex, rCut=5.0, sort=False)
        frac_center = fracs[nbs.center_index]
    else:
        frac_center = np.ones(3,dtype=np.float64) * 0.50
    while n_pass != num:
        trial_x = np.random.random(3) - np.ones(3, dtype=np.float64) * 0.50 + frac_center
        bad = False
        # check atom sites first
        for frac in fracs:
            if distance_from_fracs(frac, trial_x, cell) < deps:
                bad = True
                break
        # and check the already generated sites
        if not bad:
            for site in sites:
                if distance_from_fracs(site, trial_x, cell) < deps:
                    bad = True
                    break
        if bad:
            n_fail += 1
        else:
            n_pass += 1
            sites.append(trial_x)
    print("n_pass =", n_pass, "n_fail =", n_fail)
    return np.array(sites)