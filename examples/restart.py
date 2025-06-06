import numpy as np
from pyscf import scf
from embed_sim import ssdmet, myavas, sacasscf_mixer, siso

title = 'CoSH4'

from pyscf import gto
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

mf = scf.rohf.ROHF(mol).x2c()

chk_fname = title + '_rohf.chk'

mf.chkfile = chk_fname
mf.init_guess = 'chk'
mf.level_shift = .1
mf.max_cycle = 1
mf.max_memory = 100000
mf.kernel()

mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Co *')
mydmet.build(save_chk=True) 
# save_chk=True by default, but the dmet_chk file can be large

mydmet = ssdmet.SSDMET(mf, title=title, threshold=1e-5)
mydmet.build(chk_fname_load=title)
# chk will be used only if density matrix, impurity and threshold check is passed so that embedding can be reproduced exactly