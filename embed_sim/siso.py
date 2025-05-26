import numpy as np
from scipy.special import comb
import itertools 
from sympy.physics.wigner import wigner_3j
import os
from collections.abc import Iterable
import prettytable

from pyscf import scf
from pyscf.scf import jk
from pyscf.data import nist
from functools import reduce
from pyscf.fci import cistring
from pyscf import df, gto, lib
from pyscf.lib import logger
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf import __config__

from embed_sim.spin_utils import gen_statelis, unpack_nelec
from embed_sim.sacasscf_mixer import read_statelis

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
    dm1 = lib.einsum('abp,abq->pq', t1bra, t1ket)
    return dm1

# SISO object for SOC calculation, based on multi-configuration calculation 
class SISO():
    def __init__(self, title, mc, statelis=None, amfi=False, save_mag=True, save_Hmat=False, save_old_Hal=False, verbose=5):        
        self.title = title
        self.mol = mc.mol
        self.mc = mc

        # if statelis is None:
        #     statelis = gen_statelis(self.mc.ncas, self.mc.nelecas)
        # self.statelis = np.asarray(statelis, dtype=int)
        self.statelis = read_statelis(mc)
        self.Smax = np.shape(self.statelis)[0]
        self.Slis = np.nonzero(self.statelis)[0]

        self.casscf_state_idx = [np.arange(np.sum(self.statelis[0: S]),
                                           np.sum(self.statelis[0: S+1])) for S in range(0, self.Smax)]
        
        self.accu_statelis_mul = np.concatenate((np.zeros(1, dtype=int), np.fromiter(itertools.accumulate(self.statelis * (np.arange(1, self.Smax+1))), dtype=int))) # acumulated statelis with respect to spin multiplicity)

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

        self.amfi = amfi
        self.with_df = False
        self.save_mag = save_mag
        self.save_Hmat = save_Hmat
        self.save_old_Hal = save_old_Hal
        self.verbose = verbose

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
        return np.array([S, MS, alpha], dtype=int)
    
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

        # dm[p,q] = <|q^+ p|>

        ang_mom_core = lib.einsum('ijk, jk->i', ang_mom_1e, dm1core)
        as_mo_tdm1 = self.make_full_trans_dm()
        mocas = self.mc.mo_coeff[:, self.mc.ncore: self.mc.ncore+self.mc.ncas]
        ang_mom_act_mo = np.einsum('ijk, ja, bk->abi', ang_mom_1e, mocas, mocas.conj().T)

        ang_mom_act = lib.einsum('abi, mnba->mni', ang_mom_act_mo, as_mo_tdm1)
        ang_mom = ang_mom_core + ang_mom_act

        return ang_mom
    
    def spin_ang_mom(self):
        spin_ang_mom_ = np.zeros((3, self.nstates, self.nstates), dtype = complex)
        for S in self.Slis:
            S = int(S) # to avoid problem of wigner_3j function on some machine(macos)
            for MS1 in range(-S, S+1, 2):
                for MS2 in range(-S, S+1, 2):
                    spin_ang_mom_[2][np.ix_(self.siso_state_idx[S, MS2], self.siso_state_idx[S, MS1])] = np.eye(len(self.siso_state_idx[S, MS1])) * S * (-1.0)**(S/2 - MS2/2) * wigner_3j(S/2, 1, S/2, -MS2/2, 0, MS1/2) / wigner_3j(S/2, 1, S/2, -S/2, 0, S/2)

                    spin_ang_mom_[0][np.ix_(self.siso_state_idx[S, MS2], self.siso_state_idx[S, MS1])] = np.eye(len(self.siso_state_idx[S, MS1])) * S * (-1.0)**(S/2 - MS2/2) * wigner_3j(S/2, 1, S/2, -MS2/2, -1, MS1/2) / wigner_3j(S/2, 1, S/2, -S/2, 0, S/2) # m=-1

                    spin_ang_mom_[1][np.ix_(self.siso_state_idx[S, MS2], self.siso_state_idx[S, MS1])] = np.eye(len(self.siso_state_idx[S, MS1])) * S * (-1.0)**(S/2 - MS2/2) * wigner_3j(S/2, 1, S/2, -MS2/2, 1, MS1/2) / wigner_3j(S/2, 1, S/2, -S/2, 0, S/2) # m=1

        spin_ang_mom = np.zeros((3,self.nstates, self.nstates), dtype = complex)
        spin_ang_mom[0,:,:] = (spin_ang_mom_[0,:,:] - spin_ang_mom_[1,:,:]) / np.sqrt(2)
        spin_ang_mom[1,:,:] = -1j * (spin_ang_mom_[0,:,:] + spin_ang_mom_[1,:,:]) / np.sqrt(2)
        # here y component should have additional minus sign, but I don't know why
        spin_ang_mom[2,:,:] = spin_ang_mom_[2,:,:]

        spin_ang_mom = np.transpose(spin_ang_mom, axes=(1,2,0))
        return spin_ang_mom

    def orbital_ang_mom_old(self):
        mol = self.mol
        origin = mol.atom_coord(0)
        mol.set_common_origin(origin)
        ang_mom_1e = mol.intor('int1e_cg_irxp') / 1j # (direction, ao_bra, ao_ket)

        as_mo_tdm1 = self.make_full_trans_dm()

        mocore = self.mc.mo_coeff[:,:self.mc.ncore]
        mocas = self.mc.mo_coeff[:, self.mc.ncore: self.mc.ncore+self.mc.ncas]
        dm1b = np.dot(mocore, mocore.conj().T)

        ao_tdm1 = lib.einsum('ia, mnab, bj->mnij', mocas, as_mo_tdm1, mocas.conj().T)
        tdm1 = dm1b + ao_tdm1

        ang_mom = lib.einsum('ijk, mnkj->mni', ang_mom_1e, tdm1)
        return ang_mom

    def calc_z(self):
        # 1e SOC integrals
        if self.amfi:
            nao = self.mol.nao
            hso1e = np.zeros((3,nao,nao))
            aoslices = self.mol.aoslice_by_atom()
            natm = self.mol.natm
            for atm_id in range(natm):
                bas_start, bas_end, ao_start, ao_end = aoslices[atm_id]
                nao_atm = ao_end - ao_start
                shls_slice = (bas_start,bas_end,bas_start,bas_end)
                with self.mol.with_rinv_as_nucleus(atm_id):
                    z = -self.mol.atom_charge(atm_id)
                    hso1e[:,ao_start:ao_end,ao_start:ao_end] = z * self.mol.intor('int1e_prinvxp', comp=3, shls_slice=shls_slice)
        else:
            hso1e = self.mol.intor('int1e_pnucxp',3)
        if self.mol.has_ecp_soc():
            hso1e -= 0.5*self.mol.intor('ECPso')

        # All electron SISO
        mo_cas = self.mc.mo_coeff[:,self.mc.ncore:self.mc.ncore+self.mc.ncas]
        sodm1 = self.mc.make_rdm1()

        # 2e SOC J/K1/K2 integrals
        # SOC_2e integrals are anti-symmetric towards exchange (ij|kl) -> (ji|kl) TODO
        log = logger.Logger(self.mol.stdout, self.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        if self.amfi:
            nao = self.mol.nao
            vj = np.zeros((3,nao,nao))
            vk = np.zeros((3,nao,nao))
            vk2 = np.zeros((3,nao,nao))
            aoslices = self.mol.aoslice_by_atom()
            natm = self.mol.natm
            for atm_id in range(natm):
                bas_start, bas_end, ao_start, ao_end = aoslices[atm_id]
                nao_atm = ao_end - ao_start
                shls_slice = (bas_start,bas_end,bas_start,bas_end,bas_start,bas_end,bas_start,bas_end)
                p1vxp1 = self.mol.intor('int2e_p1vxp1', comp=3, aosym='s2kl', shls_slice=shls_slice)
                p1vxp1 = lib.unpack_tril(p1vxp1.reshape(3*nao_atm**2,-1)).reshape(3,nao_atm,nao_atm,nao_atm,nao_atm)
                vj[:,ao_start:ao_end,ao_start:ao_end] = lib.einsum('xijkl,kl->xij',p1vxp1,sodm1[ao_start:ao_end,ao_start:ao_end])
                vk[:,ao_start:ao_end,ao_start:ao_end] = lib.einsum('xijkl,jk->xil',p1vxp1,sodm1[ao_start:ao_end,ao_start:ao_end])
                vk2[:,ao_start:ao_end,ao_start:ao_end] = lib.einsum('xijkl,li->xkj',p1vxp1,sodm1[ao_start:ao_end,ao_start:ao_end])
        else:
            vj,vk,vk2 = jk.get_jk(self.mol,[sodm1,sodm1,sodm1],['ijkl,kl->ij','ijkl,jk->il','ijkl,li->kj'],intor='int2e_p1vxp1', comp=3)

        #vj,vk,vk2 = mpi_jk.get_jk(mol,np.asarray([sodm1]),hermi=0)
        t0 = log.timer('2e SOC J/K1/K2 integrals', *t0)
            
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
            S1 = int(S1)
            S2 = int(S2)
            if S1 == S2: # z_0 s_0
                for I1, I2 in itertools.product(self.casscf_state_idx[S1],
                                                self.casscf_state_idx[S2]):
                    mc.fcisolver.spin = S1 # state-averaged fcisolver does not have definite spin, and wrong spin may be used to unpack nelecas
                    mc.nelecas = unpack_nelec(mc.nelecas, spin=mc.fcisolver.spin)

                    t_dm1 = mc.fcisolver.trans_rdm1s(mc.ci[I2], mc.ci[I1], mc.ncas, mc.nelecas)
                    # shape (2, ncas, ncas)

                    for m in range(0, 3): # -1, 0, 1
                        if np.abs(wigner_3j(S2/2, 1, S1/2, -S2/2, 0, S1/2)) > 1e-8:
                            Y[m, I1, I2] = 1 / wigner_3j(S2/2, 1, S1/2, -S2/2, 0, S1/2) * lib.einsum('ij,ij->', self.z[m], 1/2 * (t_dm1[0] - t_dm1[1]))
                        else:
                            Y[m, I1, I2] = 0
        
            elif S1 - S2 == -2: # z_-1 s_+1
                for I1, I2 in itertools.product(self.casscf_state_idx[S1],
                                                self.casscf_state_idx[S2]):
                    t_dm1 = make_rdm1_splus(mc.ci[I2], mc.ci[I1], mc.ncas, mc.nelecas, spin = S1) # shape (ncas, ncas)
                    
                    for m in range(0, 3): # -1, 0, 1
                        Y[m, I1, I2] = 1 / wigner_3j(S2/2, 1, S1/2, -S2/2, 1, S1/2) * lib.einsum('ij,ij->', self.z[m], - 1 / np.sqrt(2) * t_dm1)

            elif S1 - S2 == 2: # z_+1 s_-1
                for I1, I2 in itertools.product(self.casscf_state_idx[S1],
                                                self.casscf_state_idx[S2]):
                    t_dm1 = make_rdm1_splus(mc.ci[I1], mc.ci[I2], mc.ncas, mc.nelecas, spin = S2).conj().T # shape (ncas, ncas), hermitian conjugate of splus matrix element
                    
                    for m in range(0, 3): # -1, 0, 1
                        Y[m, I1, I2] = 1 / wigner_3j(S2/2, 1, S1/2, -S2/2, -1, S1/2) * lib.einsum('ij,ij->', self.z[m], 1 / np.sqrt(2) * t_dm1)
        self.Y = Y
    
    def calc_h(self):
        for S1, S2 in itertools.product(self.Slis, self.Slis):
            S1 = int(S1)
            S2 = int(S2)
            # if Stot1 <= Stot2:
            for MS1, MS2 in itertools.product(range(-S1, S1+1, 2),
                                              range(-S2, S2+1, 2)):
                if np.abs(MS1 - MS2) <= 2:
                    ratio = (-1.0) ** (MS2/2 - MS1/2) * (-1.0)**(S2/2 - MS2/2) * wigner_3j(S2/2, 1, S1/2, -MS2/2, MS2/2-MS1/2, MS1/2)

                    self.SOC_Hamiltonian[np.ix_(self.siso_state_idx[S1, MS1], self.siso_state_idx[S2, MS2])] = ratio * self.Y[1 - (MS2-MS1)//2][np.ix_(self.casscf_state_idx[S1], self.casscf_state_idx[S2])] # m=-1->im=0, m=0->im=1, m=1->im=0

                    if S1 == S2 and MS1 == MS2:
                        e_states = np.asarray(self.mc.e_states) # TODO: Fix this in MC module
                        self.SOC_Hamiltonian[np.ix_(self.state_idx(S1, MS1), self.state_idx(S2, MS2))] += np.diag(e_states[self.casscf_state_idx[S1]])
        if self.save_Hmat:
            np.savetxt('myHmat', self.SOC_Hamiltonian)
    
    def reshape_old(self): # for check with Yuhang's code
        # print('reshape_old')
        accu_range = np.concatenate((np.zeros(1, dtype=int), np.fromiter(itertools.accumulate(self.statelis * (np.arange(1, self.Smax+1))), dtype=int))) # acumulated statelis with respect to spin multiplicity)
        # accu_range = np.fromiter(itertools.accumulate(self.statelis * (np.arange(1, self.Smax+1)), initial=0), dtype=int)
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
        if self.save_old_Hal:
            np.savetxt('old_Hal', old_Hal)

    def solve(self, nprint=4, ncomp=10):
        myeigval, myeigvec = np.linalg.eigh(self.SOC_Hamiltonian)

        mag_ene =  (myeigval-min(myeigval))*219474.63
        if self.save_mag:
            np.savetxt(self.title+'_mag.txt',mag_ene,fmt='%.6f')
            self.mag_ene = mag_ene
        
        for i in range(0, np.min((nprint, self.nstates))): # print 10 biggest coefficients and corresponding spin states
            coeff = myeigvec[:, i]
            arg_sort_coeff = np.argsort(-np.abs(coeff))
            print('state', i, 'energy', mag_ene[i])
            for j in range(0, ncomp):
                with np.printoptions(precision=3, suppress=True):
                    print(f'(S, MS, I), {self.idx2state(arg_sort_coeff[j])}\t coeff\t {coeff[arg_sort_coeff[j]]:.3f}\t |coeff| ** 2\t {np.linalg.norm(coeff[arg_sort_coeff[j]])**2:.3f}')
        print(f'mag energy {mag_ene[:20]}')
        return 
            
    def kernel(self):
        self.calc_z()
        self.calc_Y()
        self.calc_h()
        self.solve()
        return
    
    def density_fit(self, with_df=None):
        from embed_sim.df import DFSISO
        if with_df is None:
            if not getattr(self.mc, 'with_df', False):
                raise NotImplementedError
            else:
                with_df = self.mc.with_df
        return DFSISO(self.title, self.mc, self.statelis, self.save_mag, self.save_Hmat, self.save_old_Hal, self.verbose, with_df)
    
    def analyze(self, states=0, picture_change=True, gauge='length', order=0):
        '''
        Calculate oscillator strength and transition dipole moment between SOC states

        Args:
            states: integer or a list
                The index of states for analysis
            picture_change: Boolean
                If scalar relativistic effect is considered via X2C,
                picture_change can set to be True to eliminate
                picture change error (can be neglected since PCE is
                small for properties related with valence electrons).
            gauge: string
                The gauge for transition dipole moment
                gauge = 'length':  <i|r|j>
                gauge = 'velocity: TODO
            order: integer
                The order of multipole expansion when using velocity gauge, TODO
        '''

        def _charge_center(mol):
            charges = mol.atom_charges()
            coords  = mol.atom_coords()
            return np.einsum('z,zr->r', charges, coords)/charges.sum()
        myeigval, myeigvec = np.linalg.eigh(self.SOC_Hamiltonian)
        mol = self.mol
        mc = self.mc
        with_x2c = getattr(self.mc._scf, 'with_x2c', None)

        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', 'siso.analyze')
        if isinstance(states, int):
            if states >= self.nstates:
                raise IndexError('states index out of range')
        elif isinstance(states, Iterable):
            states = np.asarray(states, dtype=int)
            if states.max() >= self.nstates:
                raise IndexError('states index out of range')
        else:
            raise NotImplementedError
        
        if picture_change and with_x2c is None:
            picture_change = False
            log.warn('Picture change is only needed when X2C is applied.')
            
        if gauge.lower() != 'length':
            raise NotImplementedError
        else:
            gauge = gauge.lower()
        
        log.info('states = %s', states)
        log.info('with_x2c = %s', with_x2c)
        log.info('picture_change = %s', picture_change)
        log.info('gauge = %s', gauge)
        if gauge != 'length':
            if order > 0:
                raise NotImplementedError
            else:
                log.info('order = %s', order)

        if isinstance(states, int):
            states = [states]

        with mol.with_common_orig(_charge_center(self.mol)):
            if picture_change:
                xmol = with_x2c.get_xmol()[0]
                nao = xmol.nao
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)[:,3]
                c1 = 0.5/lib.param.LIGHT_SPEED
                ao_dip = with_x2c.picture_change(('int1e_r', prp*c1**2))
            else:
                if gauge == 'length':
                    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
                else:
                    ao_dip = mol.intor('int1e_ipovlp', comp=3, hermi=2)
        mo_cas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
        z = np.asarray([reduce(np.dot, (mo_cas.T, x.T, mo_cas)) for x in ao_dip])
        tdm = np.zeros((3, self.nstates, self.nstates))
        for idx1, idx2 in itertools.product(range(self.nstates), range(self.nstates)):
            S1, MS1, I1 = self.idx2state(idx1)
            S2, MS2, I2 = self.idx2state(idx2)
            if S1 == S2 and MS1 == MS2:
                mc.fcisolver.spin = S1
                mc.nelecas = unpack_nelec(mc.nelecas, spin=mc.fcisolver.spin)
                t_dm1 = mc.fcisolver.trans_rdm1(mc.ci[self.casscf_state_idx[S2][I2]], mc.ci[self.casscf_state_idx[S1][I1]], mc.ncas, mc.nelecas)
                tdm[:,idx2,idx1] = lib.einsum('xij,ji->x',z,t_dm1)
        
        for state in states:
            tdm_so = np.abs(np.asarray([reduce(np.dot,(myeigvec.conj().T, x, myeigvec)) for x in tdm])[:, state])
            mag_ene = myeigval - myeigval[state]
            log.info('')
            log.info('The oscillator strength and transition dipole moment for state %s', state)
            tb = prettytable.PrettyTable(['','Energy (cm-1)','Energy (nm)','Energy (eV)',
                                          'fosc','D**2 (a.u.**2)','|Dx| (a.u.)','|Dy| (a.u.)','|Dz| (a.u.)'])
            tb.align = 'c'
            tb.hrules = 3
            tb.vrules = 2
            for name in tb.field_names[:3]:
                tb.align[name] = 'r'
            for i in range(self.nstates):
                D2 = np.linalg.norm(tdm_so[:,i])**2
                tb.add_row(['%s'%i,
                            '%.6f'%(mag_ene[i]*nist.HARTREE2WAVENUMBER),
                            '%.6f'%(mag_ene[i]*1e7/nist.HARTREE2WAVENUMBER),
                            '%.6f'%(mag_ene[i]*nist.HARTREE2EV),
                            '%.6f'%np.abs(2/3*mag_ene[i]*D2),
                            '%.6f'%D2,
                            '%.6f'%tdm_so[0,i],'%.6f'%tdm_so[1,i],'%.6f'%tdm_so[2,i]])
            log.info('%s', tb)

            if self.mol.spin%2 != 0:
                log.info('')
                log.info('The oscillator strength and transition dipole moment for state %s after summing the degenerate Kramers doublets', state)
                tb = prettytable.PrettyTable(['','Energy (cm-1)','Energy (nm)','Energy (eV)','fosc','D**2 (a.u.**2)'])
                tb.align = 'c'
                tb.hrules = 3
                tb.vrules = 2
                for name in tb.field_names[:3]:
                    tb.align[name] = 'r'
                for i in range(0, self.nstates, 2):
                    D2 = np.linalg.norm(tdm_so[:,i])**2+np.linalg.norm(tdm_so[:,i+1])**2
                    tb.add_row(['%s'%(i//2),
                                '%.6f'%(mag_ene[i]*nist.HARTREE2WAVENUMBER),
                                '%.6f'%(mag_ene[i]*1e7/nist.HARTREE2WAVENUMBER),
                                '%.6f'%(mag_ene[i]*nist.HARTREE2EV),
                                '%.6f'%np.abs(2/3*mag_ene[i]*D2),
                                '%.6f'%D2])
                log.info('%s', tb)
    