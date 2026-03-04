#!/usr/bin/env python3
import os
from ase.io import read
from ase.units import Bohr
import subprocess
import numpy as np

# Hf Ta W Re Os Ir are for mapping
formal_charges = {
    "Na": 1.0, "K": 1.0, "Rb": 1.0, "Cs": 1.0,
    "Mg": 2.0, "Ca": 2.0, "Sr": 2.0, "Ba": 2.0,
    "Cu": 1.0, "Ti": 1.0,
    "Al": 3.0, "Ga": 1.0, "In": 3.0, "Tl": 1.0,
    "O": -2.0, "S": -2.0,
    "F": -1.0, "Cl": -1.0, "Br": -1.0, "I": -1.0,
    "Sn": 2.0,
    "La": 3.0, "He": 0.0, 
    "Ag": 1.0, "Hf": -4.0, "Ta": -4.0, "W": -4.0,
    "Re": -4.0, "Os": -4.0, "Ir": -4.0, "Y": 3.0, 
    "N": -3.0, "Si": 4.0, "Ce":3.0

}

def get_exact_potential(poscarfile, frac_sites):
    """
    Caculate the electrostaic potential at given sites of a
    given structure.
    
    :param poscarfile: str, name of the structure file in POSCAR format
    :param frac_sites: 2d array, each raw is a fractional coordinate at which potential is to be calculated.
    :return: an array of electrostatic potentials (V)
    
    This program invokes the calcmad executable and calculates the exact electrostaic potential on 
    given points (frac_sites) of a given crystal stucture (poscarfile).
    """
    
    mol = read(poscarfile)
    natom = len(mol)
    cell = mol.cell
    lengths = cell.lengths() / Bohr # lengths returned by ASE are in Angstroms, we convert it into Bohr
    angles = cell.angles()
    symbols = mol.get_chemical_symbols()
    fracs = mol.get_scaled_positions(wrap=True)
    # Name of the input and output files for calcmad
    infile, outfile = "calcmad.in", "calcmad.log"
    with open(infile, "w") as fin:
        print(" 3D1     Ewald ", file=fin)
        print(" 1  0.0  1.0E-12  1", file=fin)    # Factor, alpha, convergence, verbose level (Factor not important here.)
        # alpgha = 0 will force the code to generate a proper one; verbose level can be 0 to 3 (most detailed)
        print("%20.10f %20.10f %20.10f" % tuple(lengths), file=fin)
        print("%10.4f %10.4f %10.4f" % tuple(angles), file=fin)
        print(natom, file=fin)
        for i in range(natom):
            atom_name = symbols[i]
            atom_charge = formal_charges[atom_name]
            print(f"{atom_name:4}", "%20.10f %20.10f %20.10f" % tuple(fracs[i]), f"{atom_charge:8.3f}", " 0  0", file=fin)
        # non-atom positions to calculate Madelung potential 
        nSites = len(frac_sites)
        print(nSites, file=fin)
        for i in range(nSites):
            print("%20.10f %20.10f %20.10f" % tuple(frac_sites[i]), file=fin)
        print(" ", file=fin, flush=True)
    # construction of input file in done. We now need to run calcmad and extract the results
    calcmad_path = os.path.join(os.path.dirname(__file__), "../src_calcmad/calcmad")
    subprocess.run([calcmad_path, infile, outfile], check=True)
    with open(outfile, "r") as fout:
        potential_lines = fout.readlines()[3:]
        potential = []
        for line in potential_lines:
            potential.append(float(line.split()[0]))
    potential = np.array(potential)
    return potential
