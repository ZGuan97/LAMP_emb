'''
Calculate oscillator strength fosc and transition dipole moment D between SOC states
'''

import numpy as np
from pyscf import gto, scf
from embed_sim import ssdmet, sacasscf_mixer, siso, rdiis

title = 'CoSH4'
mol = gto.M(atom = '''
        Co          0.00000000        0.00000000        0.00000000
        S           2.30186590        0.00000000        0.00000000
        S          -0.76728863        0.00000000        2.17021998
        S          -0.76728863        1.87946564       -1.08510999
        S          -0.76728863       -1.87946564       -1.08510999
        H           2.73758137       -0.00000000       -1.23238950
        H          -0.33157314        2.94674623       -0.46891523
        H          -0.33157314       -2.94674623       -0.46891523
        H          -2.07443508       -0.00000000        2.17021997
    ''',
    basis='x2c-SVPall', symmetry=0 ,spin = 3,charge = -2, verbose= 4, nucmod='G')
mf = scf.ROHF(mol).x2c()
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Co.*d']),power=0.2)
mf.init_guess = 'atom'
mf.max_cycle = 100
mf.kernel()

mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Co.*', threshold=1e-12)
mydmet.build(save_chk=False)
ncas, nelec, es_mo = mydmet.avas(['Co 3d'], minao=mol._basis['Co'], openshell_option=2, threshold=0.5)
es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)
mycas = mydmet.total_cas(es_cas)

mysiso = siso.SISO(title, mycas, amfi=True) # Use one-center approximation to accelerate SOC calculation
mysiso.kernel()

mysiso.analyze() # The ground state is analyzed by default

# Note that if the molecule has odd unpaired electrons, all the states are Kramers degenerate
mysiso.analyze(states=2) # Analyze the first excited state, i.e. KD1

# The following ones are equivalent
mysiso.analyze(states=[0,1,2,3])
mysiso.analyze(states=(0,1,2,3))
mysiso.analyze(states=range(4))
mysiso.analyze(states=np.arange(4))