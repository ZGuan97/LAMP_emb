#!/usr/bin/env python3

import numpy as np
from ase.io import read
from itertools import product

from .exact_potential import formal_charges

class neighbors(object):
    """
    Calculate neighbors of a given atom within a cutoff
    """
    
    def __init__(self, structFile, atom = 0, rCut = 10.0, cAtom = None, cAtomIndex = None, rCore = -1.0e0, sort = False) -> None:
        """
        Specification of the system.
        structFile: poscarfile
        atom: the atom index of the central atom
        rCut, rCore: atoms within rCut and out of rCore is treated as neighbors (unit: A)
        cAtom: str
        cAtomIndex: another way to specify the central atom. For example, we can say the 1-th La atom
                    cAtomIndex starts from 1.
        sort: sorting the neighbor according to distance
        """
        super().__init__()
        mol = read(structFile)
        self.mol = mol
        self.rCut = rCut
        self.rCore = rCore
        self.center_index = atom
        # update center_index if cAtom and cAtomIndex are present
        if not (cAtom is None or cAtomIndex is None):
            symbols = mol.get_chemical_symbols()
            count = 0
            for i, symb in enumerate(symbols):
                if symb == cAtom:
                    count += 1
                if count == cAtomIndex:
                    self.center_index = i
                    break
        # perform the actual calculation when initialization
        cell = mol.get_cell()
        fracs = mol.get_scaled_positions()
        cross_products = np.cross(cell[[1,2,0],:], cell[[2,0,1],:])
        surface_distances = cell.volume / np.linalg.norm(cross_products, axis=1)
        Lmax = [int(x) for x in np.ceil(self.rCut / surface_distances) + 2]
        self.indices = []
        self.offsets = []
        self.distances = []
        self.cart_x = []
        for ia, ib, ic in product(range(-Lmax[0], Lmax[0]+1), range(-Lmax[1], Lmax[1]+1), range(-Lmax[2], Lmax[2]+1)):
            offset = np.array([ia, ib, ic], dtype=np.float64)
            for i in range(len(mol)):
                disp = np.matmul(fracs[i] - fracs[self.center_index] + offset, cell)
                d = np.linalg.norm(disp)
                if d >= self.rCore and d < self.rCut:
                    self.indices.append(i)
                    self.offsets.append(offset)
                    self.distances.append(d)
                    self.cart_x.append(disp)
        # sort
        if sort:
            ascending_index = np.argsort(self.distances)
            self.indices = np.array(self.indices)[ascending_index]
            self.offsets = np.array(self.offsets)[ascending_index]
            self.distances = np.array(self.distances)[ascending_index]
            self.cart_x = np.array(self.cart_x)[ascending_index]
    
    def get_neighbors(self):
        """
        Return copies of internal indices and offsets
        """
        return np.array(self.indices), np.array(self.offsets)
    
    def get_distances(self):
        """
        Compute distances to neighboring atoms
        """
        return np.array(self.distances)
    
    def get_charge_list(self):
        """
        Return the 1D array of charges of neighbors
        """
        indices, offsets = self.get_neighbors()
        symbols = self.mol.get_chemical_symbols()
        charges = np.zeros(indices.shape, dtype=np.float64)
        for i, index in enumerate(indices):
            charges[i] = formal_charges[symbols[index]]
        return charges
    
    def get_total_charge(self):
        """
        Return the total charge of the neighbors
        """
        charges = self.get_charge_list()
        return charges.sum()
    
    def get_number_of_neighbors(self):
        """
        Total number of neighboring atoms.
        """
        return len(self.indices)
    
    def get_cartesian_coordinates(self, origin_shifted = True):
        """
        Return the Cartesian coordinates of neighbors.
        
        in:
            origin_shifted: bool
                whether to use the central atom as origin.
        """
        # Note that self.cart_x is already shifted to use central atom as origin
        if origin_shifted:
            origin = np.zeros(3, dtype=np.float64)
        else:
            cell = self.mol.get_cell()
            frac_center = self.mol.get_scaled_positions()[self.center_index]
            origin = np.matmul(frac_center, cell)
        return np.array(self.cart_x) + origin