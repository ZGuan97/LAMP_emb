import numpy as np
from pyscf import scf
from embed_sim import myavas, sacasscf_mixer

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
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()

ncas, nelec, mo = myavas.avas(mf, 'Co 3d', minao='def2tzvp', openshell_option=3, threshold=0.5)

mycas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec, statelis=[0, 40, 0, 10])
cas_result = mycas.kernel(mo)
# CASSCF energy = -2986.16051679323
# CASCI E = -2986.16051679323  E(CI) = -19.6474414129580  S^2 = 1.3500000
# CASCI state-averaged energy = -2986.16051679322

e_corr_casci = sacasscf_mixer.sacasscf_nevpt2_casci_ver(mycas)
e_corr_undo = sacasscf_mixer.sacasscf_nevpt2_undo_ver(mycas)

print(np.allclose(e_corr_casci, e_corr_undo))

mycas.e_states += e_corr_undo