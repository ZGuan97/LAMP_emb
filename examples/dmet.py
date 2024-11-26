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
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()

mydmet = ssdmet.SSDMET(mf, title=title)
mydmet.imp_idx = mol.search_ao_label('Co *')
mydmet.build()

es_mf = mydmet.ROHF()

ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='def2tzvp', threshold=0.5, canonicalize=False)
# a strange thing: using def2tzvp give smaller avas eigenvalues
# and may involve virtual orbitals if we don't increase the threshold

es_cas = sacasscf_mixer.sacasscf_mixer(es_mf, ncas, nelec)
es_cas.kernel(es_mo)

total_cas = mydmet.total_cas(es_cas)
mysiso = siso.SISO(title, total_cas)
mysiso.kernel()