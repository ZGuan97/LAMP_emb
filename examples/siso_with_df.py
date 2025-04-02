import os
import numpy as np
from pyscf import gto, scf, df
from embed_sim import ssdmet, sacasscf_mixer, siso, rdiis

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

'''
Density fitting and scalar relativistic effects can be applied together,
regardless to the order you apply the decoration.
'''
mf = scf.rohf.ROHF(mol).density_fit().x2c()

'''
Sometimes it is useful to save it to disk for re-use in later calculations.
This can be achieved by specifying a HDF5 file by setting _cderi_to_save.
The saved DF tensor can be used later by setting _cderi to the HDF5 file.
'''
cderi_fname = title + '_cderi.h5'
if not os.path.exists(cderi_fname):
    mydf = df.df.DF(mol)
    mydf.auxbasis = mf.with_df.auxbasis
    mydf._cderi_to_save = cderi_fname
    mydf.build()
    mf.with_df = mydf
else:
    mf.with_df._cderi = cderi_fname
    mf.with_df.auxmol = df.addons.make_auxmol(mol)

chk_fname = title + '_rohf.chk'
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Co.*d']),power=0.2)
mf.chkfile = chk_fname
mf.init_guess = 'chk'
mf.level_shift = .1
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()

mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Co.*')
# if impurity is not assigned, the orbitals on the first atom is chosen as impurity
mydmet.build()

ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='def2tzvp', threshold=0.5, openshell_option=3)

es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)

es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
'''
Density fitting can be used to accelerate the calculation of SOC 2e integrals
by setting a DF object to the with_df attribute.
'''
mysiso = siso.SISO(title, total_cas, with_df=mf.with_df)
mysiso.kernel()
