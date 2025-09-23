import os
import numpy as np
from pyscf import gto, scf, df
from embed_sim import aodmet, myavas, sacasscf_mixer, siso, rdiis

'''
PySCF only supports strongly contracted NEVPT2 (SC-NEVPT2).
Prism (https://github.com/sokolov-group/prism) supports both partially contracted NEVPT2 (PC-NEVPT2)
(also known as full internal contraction NEVPT2 (FIC-NEVPT2)) and quasidegenerate NEVPT2 (QD-NEVPT2).
'''

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
    basis='ANO-R1', symmetry=0, spin = 3, charge = -2, nucmod='G', verbose= 4)

    return mol

mol = get_mol(0)

auxbasis = df.autoaux(mol)
mf = scf.ROHF(mol).density_fit(auxbasis=auxbasis).x2c()
folder = './'
if os.path.exists(folder+title+'_df.h5'):
    mf.with_df._cderi = folder+title+'_df.h5'
    mf.with_df.auxmol = df.make_auxmol(mol, auxbasis=auxbasis)
else:
    mf.with_df._cderi_to_save = folder+title+'_df.h5'

chk_fname = title + '_rohf.chk'
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Co.*d']),power=0.2)
mf.chkfile = chk_fname
mf.init_guess = 'atom'
mf.level_shift = 2.0
mf.max_cycle = 1000
mf.kernel()

ncas, nelec, mo = myavas.avas(mf, 'Co 3d', minao=mol._basis['Co'], openshell_option=2, threshold=0.5)
mycas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec)
cas_result = mycas.kernel(mo)

'''
SC-NEVPT2
'''
e_corr = sacasscf_mixer.sacasscf_nevpt2(mycas)
e_corr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='SC')
'''
PC(FIC)-NEVPT2
'''
e_corr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='PC')
e_corr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='FIC')
'''
QD-PC-NEVPT2
'''
e_corr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='QD')

'''
Expert options
See https://github.com/sokolov-group/prism for more details.
'''
expert_options = {'compute_singles_amplitudes': True,
                  's_thresh_singles': 1e-10,
                  's_thresh_doubles': 1e-10}
e_corr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='PC', expert_options=expert_options)

mycas.fcisolver.e_states = mycas.fcisolver.e_states + e_corr

mysiso = siso.SISO(title, mycas, amfi=True, save_mag=False)
mysiso.kernel()

'''
DMET-(QD-)PC-NEVPT2
'''
mydmet = aodmet.AODMET(mf, title=title, imp_idx='Co.*', es_natorb=False, bath_option={'MP2':1e-3}).density_fit()
mydmet.build(save_chk=False)

ncas, nelec, es_mo = mydmet.avas('Co 3d', minao=mol._basis['Co'], threshold=0.5, openshell_option=2)
es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)

'''
If the oscillator strengths of PC- or QD-NEVPT2 are needed to be calculate, you must input dmet=mydmet.
Otherwise, the transition dipole moments cannot be calculated (so that they are set to zeros), and the 
oscillator strengths in the output are all zeros, which are meaningless.
'''
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas, method='PC', dmet=mydmet)
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas, method='QD', dmet=mydmet, expert_options=expert_options)

es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
Ha2cm = 219474.63
np.savetxt(mydmet.title+'_opt.txt',(es_cas.fcisolver.e_states-np.min(es_cas.fcisolver.e_states))*Ha2cm,fmt='%.6f')

'''
The amfi means one-center approximated SOMF Hamiltonian (SOMF(1X)), which strictly speaking
is not AMFI. The notation here is just for the consistence with Block2.
'''
mysiso = siso.SISO(title, total_cas, amfi=True).density_fit()

mysiso.kernel()
