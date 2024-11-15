from embed_sim import myavas, sacasscf_mixer, siso

import numpy as np
from pyscf import scf

title = 'freeion'

from pyscf import gto
mol = gto.M(atom = '''
                Co             
     ''',
     basis={'default':'MINAO'}, symmetry=0 ,spin = 3,charge = 2,verbose= 4)

mf = scf.rohf.ROHF(mol).x2c()
mf_smear = scf.addons.smearing_(mf, sigma=.1, method='fermi')

chk_fname = title + '_rohf.chk'

mf_smear.chkfile = chk_fname
mf_smear.init_guess = 'chk'
mf_smear.level_shift = .2
mf_smear.max_cycle = 1000
mf_smear.max_memory = 100000
mf_smear.kernel()

mf = mf_smear.undo_smearing()
mf.max_cycle = 1
mf.kernel()

ncas, nelec, mo = myavas.avas(mf, 'Co 3d', threshold=0.5, minao='def2tzvp' ,canonicalize=False)

mycas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec, statelis=[0, 40, 0, 10])
mycas.kernel(mo)

mysiso = siso.SISO(title, mycas)
mysiso.kernel()

orb_ang_mom = mysiso.orbital_ang_mom()
spin_ang_mom = mysiso.spin_ang_mom()

tot_ang_mom = spin_ang_mom*0.5 + orb_ang_mom

eigval, eigvec = np.linalg.eigh(mysiso.SOC_Hamiltonian)
eig_state_tot_ang_mom = np.einsum('pm, mni, nq->pqi', eigvec.conj().T, tot_ang_mom, eigvec)

J_sq = np.einsum('pqi, qpi->p', eig_state_tot_ang_mom, eig_state_tot_ang_mom)
print('check J_tot')
print(J_sq[:10])
print(9/2*(9/2+1))

np.random.seed(0)
Nsample = 80
s = np.random.normal(0, 1, size=3)
norm=np.sqrt(np.sum(s*s))
direction=1/norm * s
magnitude = 1e-5

H_zee = np.zeros((mysiso.nstates, mysiso.nstates), dtype = complex)
H_zee += np.einsum('ija, a->ij', spin_ang_mom, direction) * magnitude
H_zee += np.einsum('ija, a->ij', orb_ang_mom, direction) * magnitude

tot_H = mysiso.SOC_Hamiltonian + H_zee

mag_eigval, mag_eigvec = np.linalg.eigh(tot_H)
mag_ene_zero = np.min(mag_eigval)

for i in range(1, 10):
    print(i, 'th excited state', 'energy', mag_eigval[i] - mag_ene_zero, 'Lande g-factor', (mag_eigval[i] - mag_ene_zero)/i/magnitude)

S_num = 3/2
L_num = 3
J_num = 9/2
gJ = 1 + (J_num*(J_num+1) + S_num*(S_num+1) - L_num*(L_num+1)) / (2*J_num*(J_num+1))
print('gJ', gJ)