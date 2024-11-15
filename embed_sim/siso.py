import numpy as np
from math import comb
import itertools 
from sympy.physics.wigner import wigner_3j

from pyscf import scf
from pyscf.scf import jk
from pyscf.data import nist
from functools import reduce
from pyscf.fci import cistring

def Weyl_nstate(ncas, nelecas, spin): # definition of S is the same as that in pyscf.mol, equal to 2S actually
    if (nelecas - spin)%2 == 0: 
        nstate = (spin+1)/(ncas+1) * comb(ncas+1, round(nelecas/2 - spin/2)) * comb(ncas+1, round(nelecas/2 + spin/2 + 1))
    else:
        nstate = 0
    return round(nstate)

def gen_statelis(ncas, nelecas):
    # ncas: number of active orbitals
    # nelecas: number of electron in active space
    nelecas = np.sum(nelecas)
    Smax = min(nelecas, 2*ncas-nelecas) # definition of S is same as that in molecule, equal to 2S actually
    statelis = np.array([Weyl_nstate(ncas, nelecas, ispin) for ispin in range(0, Smax+1)])
    return statelis

def unpack_nelec(nelec, spin=None):
    # from pyscf/fci/addons.py
    if spin is None:
        spin = 0
    else:
        nelec = int(np.sum(nelec))
    if isinstance(nelec, (int, np.number)):
        nelecb = (nelec-spin)//2
        neleca = nelec - nelecb
        nelec = neleca, nelecb
    return nelec

def make_rdm1_splus(bra, ket, norb, nelec, spin=None): # increase M_S of ket by 1
    # <bra|i_alpha^+ j_beta|ket> 
    # from examples/fci/35-transition_density_matrix.py
    neleca, nelecb = unpack_nelec(nelec, spin=spin) # na, nb of ket
    ades_index = cistring.gen_des_str_index(range(norb), neleca+1)
    bdes_index = cistring.gen_des_str_index(range(norb), nelecb)
    na_bra = cistring.num_strings(norb, neleca+1)
    nb_bra = cistring.num_strings(norb, nelecb-1)
    na_ket = cistring.num_strings(norb, neleca)
    nb_ket = cistring.num_strings(norb, nelecb)
    assert bra.shape == (na_bra, nb_bra)
    assert ket.shape == (na_ket, nb_ket)

    t1bra = np.zeros((na_ket,nb_bra,norb))
    t1ket = np.zeros((na_ket,nb_bra,norb))
    # bra and ket after performing creation and annilation operators
    for str0, tab in enumerate(bdes_index): # str0: ket for beta spin, str1: bra for beta spin 
        for _, i, str1, sign in tab: # i: orbital index in a_{i\beta}, sign: sign of matrix element
            t1ket[:,str1,i] += sign * ket[:,str0]
    for str0, tab in enumerate(ades_index):
        for _, i, str1, sign in tab:
            t1bra[str1,:,i] += sign * bra[str0,:]
    dm1 = np.einsum('abp,abq->pq', t1bra, t1ket)
    return dm1

# SISO object for SOC calculation, based on multi-configuration calculation 
class SISO():
    def __init__(self, title, mc, statelis=None):
        self.title = title
        self.mol = mc.mol
        self.mc = mc

        if statelis is None:
            statelis = gen_statelis(self.mc.ncas, self.mc.nelecas)
        self.statelis = np.asarray(statelis, dtype=int)
        self.Smax = np.shape(self.statelis)[0]
        self.Slis = np.nonzero(self.statelis)[0]

        self.casscf_state_idx = [np.arange(np.sum(self.statelis[0: S]),
                                           np.sum(self.statelis[0: S+1])) for S in range(0, self.Smax)]
        
        self.accu_statelis_mul = np.fromiter(itertools.accumulate(self.statelis * (np.arange(1, self.Smax+1)), initial=0), dtype=int) # acumulated statelis with respect to spin multiplicity

        self.siso_state_idx = {}
        for S in range(0, self.Smax):
            for MS in range(-S, S+1):
                self.siso_state_idx[S, MS] = self.state_idx(S, MS)

        self.nstates = np.sum([(i+1)*(x) for i,x in enumerate(self.statelis)])

        self.z = None
        self.Y = None
        # self.Y = np.zeros((np.sum(self.statelis), np.sum(self.statelis), 3), dtype = complex)
        self.SOC_Hamiltonian = np.zeros((self.nstates, self.nstates), dtype = complex)
        self.full_trans_dm = np.zeros((self.nstates, self.nstates, self.mc.ncas, self.mc.ncas), dtype = complex)

    def state_idx(self, S=None, MS=None, alpha=None): 
        if alpha is not None:
            return self.accu_statelis_mul[S] + (MS+S)//2*self.statelis[S] + alpha
        elif MS is not None:
            return np.arange(self.accu_statelis_mul[S] + (MS+S)//2*self.statelis[S],
                             self.accu_statelis_mul[S] + ((MS+S)//2+1)*self.statelis[S])
        elif S is not None:
            return np.arange(self.accu_statelis_mul[S], self.accu_statelis_mul[S+1])
        else:
            return np.arange(self.nstates)
        
    def idx2state(self, idx):
        S = np.arange(self.Smax+1)[self.accu_statelis_mul>idx][0]-1
        MS = (idx - self.accu_statelis_mul[S])//self.statelis[S] * 2 - S
        alpha = (idx - self.accu_statelis_mul[S])%self.statelis[S]
        return S, MS, alpha
    
    def make_full_trans_dm(self):
        np.save('mo_coeff', self.mc.mo_coeff)
        np.save('e_states', self.mc.e_states)
        mc = self.mc
        unmixed_fcisolver = mc.fcisolver.undo_state_average() # unmix for calling trans_rdm1
        for S in self.Slis:
            unmixed_fcisolver.spin = S
            t_dm1 = np.array([[unmixed_fcisolver.trans_rdm1(mc.ci[i], mc.ci[j], mc.ncas, mc.nelecas) 
                                    for i in self.casscf_state_idx[S]]
                                    for j in self.casscf_state_idx[S]])
            for MS in range(-S, S+1, 2):
                self.full_trans_dm[np.ix_(self.siso_state_idx[S, MS], self.siso_state_idx[S, MS])] = t_dm1
        return self.full_trans_dm
    
    def orbital_ang_mom(self):
        mol = self.mol
        origin = mol.atom_coord(0)
        mol.set_common_origin(origin)
        ang_mom_1e = mol.intor('int1e_cg_irxp') / 1j # (direction, ao_bra, ao_ket)

        mocore = self.mc.mo_coeff[:,:self.mc.ncore]
        dm1core = np.dot(mocore, mocore.conj().T)

        ang_mom_core =np.einsum('ijk, jk->i', ang_mom_1e, dm1core)
        as_mo_tdm1 = self.make_full_trans_dm()
        mocas = self.mc.mo_coeff[:, self.mc.ncore: self.mc.ncore+self.mc.ncas]
        ang_mom_act_mo = np.einsum('ijk, ja, bk->abi', ang_mom_1e, mocas, mocas.conj().T)

        ang_mom_act = np.einsum('abi, mnab->mni', ang_mom_act_mo, as_mo_tdm1)
        ang_mom = ang_mom_core + ang_mom_act

        return ang_mom

    def orbital_ang_mom_old(self):
        mol = self.mol
        origin = mol.atom_coord(0)
        mol.set_common_origin(origin)
        ang_mom_1e = mol.intor('int1e_cg_irxp') / 1j # (direction, ao_bra, ao_ket)

        as_mo_tdm1 = self.make_full_trans_dm()

        mocore = self.mc.mo_coeff[:,:self.mc.ncore]
        mocas = self.mc.mo_coeff[:, self.mc.ncore: self.mc.ncore+self.mc.ncas]
        dm1b = np.dot(mocore, mocore.conj().T)

        ao_tdm1 = np.einsum('ia, mnab, bj->mnij', mocas, as_mo_tdm1, mocas.conj().T)
        tdm1 = dm1b + ao_tdm1

        ang_mom = np.einsum('ijk, mnjk->mni', ang_mom_1e, tdm1)
        return ang_mom

    def calc_z(self):
        # 1e SOC integrals
        hso1e = self.mol.intor('int1e_pnucxp',3)

        # All electron SISO
        mo_cas = self.mc.mo_coeff[:,self.mc.ncore:self.mc.ncore+self.mc.ncas]
        sodm1 = self.mc.make_rdm1()

        # 2e SOC J/K1/K2 integrals
        # SOC_2e integrals are anti-symmetric towards exchange (ij|kl) -> (ji|kl) TODO
        vj,vk,vk2 = jk.get_jk(self.mol,[sodm1,sodm1,sodm1],['ijkl,kl->ij','ijkl,jk->il','ijkl,li->kj'],intor='int2e_p1vxp1', comp=3)

        #vj,vk,vk2 = mpi_jk.get_jk(mol,np.asarray([sodm1]),hermi=0)
        hso2e = vj - 1.5 * vk - 1.5 * vk2
        
        alpha = nist.ALPHA
        hso = 1.j*(alpha**2/2)*(hso1e+hso2e)

        # from AO matrix element to MO matrix element
        h1 = np.asarray([reduce(np.dot, (mo_cas.T, x.T, mo_cas)) for x in hso])
        z = np.asarray([1/np.sqrt(2)*(h1[0]-1.j*h1[1]),h1[2],-1/np.sqrt(2)*(h1[0]+1.j*h1[1])]) # m= -1, 0, 1
        self.z = z
        # np.save(self.title+'_siso_z', z)
        return z

    def calc_Y(self):
        # TODO check spin states through mc.ci CI coefficients shape
        # using MC density matrix
        # <S2| H_SOMF |S1> 
        Y = np.zeros((3, np.sum(self.statelis), np.sum(self.statelis)), dtype = complex)

        mc = self.mc
        for S1, S2 in itertools.product(range(0, self.Smax), range(0, self.Smax)):
            if S1 == S2: # z_0 s_0
                for I1, I2 in itertools.product(self.casscf_state_idx[S1],
                                                self.casscf_state_idx[S2]):
                    mc.fcisolver.spin = S1 # state-averaged fcisolver does not have definite spin, and wrong spin may be used to unpack nelecas
                    t_dm1 = mc.fcisolver.trans_rdm1s(mc.ci[I2], mc.ci[I1], mc.ncas, mc.nelecas)
                    # shape (2, ncas, ncas)

                    for m in range(0, 3): # -1, 0, 1
                        Y[m, I1, I2] = 1 / wigner_3j(S2/2, 1, S1/2, -S2/2, 0, S1/2) * np.einsum('ij,ij->', self.z[m], 1/2 * (t_dm1[0] - t_dm1[1]))
        
            elif S1 - S2 == -2: # z_-1 s_+1
                for I1, I2 in itertools.product(self.casscf_state_idx[S1],
                                                self.casscf_state_idx[S2]):
                    t_dm1 = make_rdm1_splus(mc.ci[I2], mc.ci[I1], mc.ncas, mc.nelecas, spin = S1) # shape (ncas, ncas)
                    
                    for m in range(0, 3): # -1, 0, 1
                        Y[m, I1, I2] = 1 / wigner_3j(S2/2, 1, S1/2, -S2/2, 1, S1/2) * np.einsum('ij,ij->', self.z[m], - 1 / np.sqrt(2) * t_dm1)

            elif S1 - S2 == 2: # z_+1 s_-1
                for I1, I2 in itertools.product(self.casscf_state_idx[S1],
                                                self.casscf_state_idx[S2]):
                    t_dm1 = make_rdm1_splus(mc.ci[I1], mc.ci[I2], mc.ncas, mc.nelecas, spin = S2).conj().T # shape (ncas, ncas), hermitian conjugate of splus matrix element
                    
                    for m in range(0, 3): # -1, 0, 1
                        Y[m, I1, I2] = 1 / wigner_3j(S2/2, 1, S1/2, -S2/2, -1, S1/2) * np.einsum('ij,ij->', self.z[m], 1 / np.sqrt(2) * t_dm1)
        self.Y = Y
    
    def calc_h(self):
        for S1, S2 in itertools.product(self.Slis, self.Slis):
            # if Stot1 <= Stot2:
            for MS1, MS2 in itertools.product(range(-S1, S1+1, 2),
                                              range(-S2, S2+1, 2)):
                if np.abs(MS1 - MS2) <= 2:
                    ratio = (-1.0) ** (MS2/2 - MS1/2) * (-1.0)**(S2/2 - MS2/2) * wigner_3j(S2/2, 1, S1/2, -MS2/2, MS2/2-MS1/2, MS1/2)

                    self.SOC_Hamiltonian[np.ix_(self.siso_state_idx[S1, MS1], self.siso_state_idx[S2, MS2])] = ratio * self.Y[1 - (MS2-MS1)//2][np.ix_(self.casscf_state_idx[S1], self.casscf_state_idx[S2])] # m=-1->im=0, m=0->im=1, m=1->im=0

                    if S1 == S2 and MS1 == MS2:
                        e_states = np.asarray(self.mc.e_states) # TODO: Fix this in MC module
                        self.SOC_Hamiltonian[np.ix_(self.state_idx(S1, MS1), self.state_idx(S2, MS2))] += np.diag(e_states[self.casscf_state_idx[S1]])
        np.savetxt('myHmat', self.SOC_Hamiltonian)
    
    def reshape_old(self): # for check with Yuhang's code
        # print('reshape_old')
        accu_range = np.fromiter(itertools.accumulate(self.statelis * (np.arange(1, self.Smax+1)), initial=0), dtype=int)
        arg = []
        for spin in range(0, self.Smax):
            # print('spin', spin)
            for MS in np.arange(0, spin + 1):
                # print('MS', MS)
                for istate in range(0, self.statelis[spin]):
                    # print(istate * (spin+1) + MS + accu_range[spin])
                    arg.append(istate * (spin+1) + MS + accu_range[spin])
        arg = np.array(arg,dtype=int)
        # print(arg)

        inv = np.argsort(arg)

        old_Hal = self.SOC_Hamiltonian[inv][:, inv]
        np.savetxt('old_Hal', old_Hal)

    def solve(self):
        myeigval, myeigvec = np.linalg.eigh(self.SOC_Hamiltonian)

        print('mag energy', (myeigval[:10]-min(myeigval))*219474.63)

        for i in range(0, np.min((10, self.nstates))): # print 10 biggest coefficients and corresponding spin states
            coeff = myeigvec[:, i]
            arg_sort_coeff = np.argsort(-np.abs(coeff))
            print('state', i)
            for j in range(0, 10):
                print('(S, MS, I)', self.idx2state(arg_sort_coeff[j]), 'coeff', coeff[arg_sort_coeff[j]], 'coeff ** 2', np.linalg.norm(coeff[arg_sort_coeff[j]])**2)
        return 
            
    def kernel(self):
        self.calc_z()
        self.calc_Y()
        self.calc_h()
        self.solve()
        return 
    

if __name__ == '__main__':
    title = 'Darmanovic2019_1'
    
    from pyscf import gto
    mol = gto.M(atom='Darmanovic2019_1.xyz',
        basis={'default':'def2tzvp','C':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 3,charge = 2,verbose= 4)
    
    mf = scf.rohf.ROHF(mol).x2c()
    chk_fname = title + '_rohf.chk'

    mf.chkfile = chk_fname
    mf.init_guess = 'chk'
    mf.level_shift = .1
    mf.max_cycle = 0
    mf.max_memory = 100000
    mf.kernel()
    
    from pyscf.mcscf import avas 
    from liblan.solver import mcscfsol_pt2
    aslabel = ['Co 3d']
    statelis = [0, 40, 0, 10]

    import os
    from pyscf.lib import chkfile

    caschk_fname = 'sacasscf.chk'
    if not os.path.isfile(caschk_fname):
        ncasorb,ncaselec,casorbind = avas.avas(mf, aslabel, canonicalize=False)
    else:
        ncasorb,ncaselec = 5, 7
        casorbind = chkfile.load(caschk_fname, 'mcscf/mo_coeff')

    # from pyscf import lib
    # mo = lib.chkfile.load("sacasscf.chk", 'mcscf/mo_coeff')

    mc = mcscfsol_pt2.sacasscf_solve_imp(mf,mol,ncasorb,ncaselec,casorbind,statelis,avas=True,nevpt2_ene=False)
    
    mysiso = SISO(title, mol, mc)


    import h5py
    z_fname = 'Darmanovic2019_1_siso_z.h5'
    with h5py.File(z_fname, 'r') as fh5:
        old_z = fh5['z'][:]

    mysiso.z = old_z
    
    # mysiso.calc_z()
    mysiso.kernel()
    ang_mom = mysiso.orbital_ang_mom()
    # mysiso.reshape_old()