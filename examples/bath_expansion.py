import os
import numpy as np
from pyscf import gto, scf, df
from embed_sim import aodmet, sacasscf_mixer, siso, rdiis

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

'''
DMET with density fitting is highly recommanded when performing bath expansion,
as DF reduces the formal scaling of local MP2 from O(N^4) to O(N^3),
see Nusspickel, M.; Booth, G. H. Phys. Rev. X. 2022, 12, 011046. for more details.
'''
'''
The bath_option have to be a dictionary, with the key representing the method for bath expansion,
and the value representing the threshold (eta in the reference).
When ROHF is the low-level solver, MP2, RMP2, ROMP2 and UMP2 are all available for the key.
In this case, both MP2 and RMP2 will be automatically switched to ROMP2. UMP2 may give very similar
results to ROMP2 when the spin-contamination is small, but you should note that applying UMP2 to ROHF
is not theoretically rigorous.
'''
mydmet = aodmet.AODMET(mf, title=title, imp_idx='Co.*', es_natorb=False, bath_option={'MP2':1e-3}).density_fit()
mydmet.build(save_chk=False)

ncas, nelec, es_mo = mydmet.avas('Co 3d', minao=mol._basis['Co'], threshold=0.5, openshell_option=2)
es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)

es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
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
