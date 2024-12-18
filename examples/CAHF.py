import numpy as np
from pyscf import gto, scf

def get_mol(dihedral):
     mol = gto.M(atom = '''
                Co             
                S                  1            2.30186590
                S                  1            2.30186590    2            109.47122060
                S                  1            2.30186590    3            109.47122065    2            -120.00000001                  0
                S                  1            2.30186590    4            109.47122060    3            120.00000001                   0
                H                  2            1.30714645    1            109.47121982    4            '''+str(-60-dihedral)+'''      0
                H                  4            1.30714645    1            109.47121982    3            '''+str(60+dihedral)+'''       0
                H                  5            1.30714645    1            109.47121982    4            '''+str(-180+dihedral)+'''     0
                H                  3            1.30714645    1            109.47121982    4            '''+str(60-dihedral)+'''       0
     ''',
     basis={'default':'def2tzvp','s':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 3,charge = -2,verbose= 4)

     return mol

mol = get_mol(0)

from embed_sim import cahf, rdiis
mf = cahf.CAHF(mol, ncas=5, nelecas=7, spin=3).x2c()
mf.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=mol.search_ao_label(['Co.*d']), power=0.2)

mf.max_cycle=200
mf.level_shift = 0.5 # larger level shift should be used to improve convergence, especially when dealing with lanthanide systems
mf.kernel()
