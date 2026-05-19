import numpy as np
from pyscf import gto, scf
from embed_sim import cahf, myavas

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

mf1 = scf.rohf.ROHF(mol).x2c()
mf1.init_guess = 'atom'
mf1.max_cycle=0
mf1.kernel()

ncas, nelec, mo = myavas.avas(mf1, ['Co 3d'], threshold=0.5)
ncas, nelec = 5, 7
occ = cahf.CAHF_get_occ(ncas, nelec)(mf1)

mf2 = cahf.CAHF(mol, ncas=5, nelecas=7, spin=3).x2c().newton()
mf2.max_cycle=200
mf2.conv_tol = 1e-9
mf2.kernel(mo, occ)
