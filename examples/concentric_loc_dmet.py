import sys
sys.path.append('/Users/cuihengxin/Desktop/2025-2030phd/8dmet4reac/2de/LAMP_emb')
import numpy as np
from pyscf import gto, scf
from embed_sim import ssdmet, sacasscf_mixer, siso, concentric_loc

title = 'CoSH4'

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

mf = scf.rohf.ROHF(mol).x2c().density_fit()

chk_fname = title + '_rohf.chk'

mf.chkfile = chk_fname
mf.init_guess = 'chk'
mf.level_shift = .1
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()
imp = mol.search_ao_label(['Co.*'])
mydmet = ssdmet.SSDMET(mf, title=title, imp_idx=imp, es_natorb=False).density_fit()
# if impurity is not assigned, the orbitals on the first atom is chosen as impurity
mydmet.build()

concentric_loc.concentric_localization(mydmet, proj_bas = 'ccpvdz', n_shell =1, atoms_A=[0],couple_op = 'fock',ele_density=True)
concentric_loc.concentric_occ_localization(mydmet, proj_bas = 'ccpvdz', n_shell =1, atoms_A=[0], couple_op = 'fock', ele_density=True)
ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='def2tzvp', threshold=0.5)

es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)

es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
mysiso = siso.SISO(title, total_cas).density_fit()
mysiso.kernel()
