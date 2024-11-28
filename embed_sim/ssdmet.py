import numpy as np
from functools import reduce
from scipy.linalg import block_diag
import h5py

from pyscf.lo.orth import lowdin
from pyscf import gto, scf, mcscf, ao2mo

import os

def compare_imp_idx(imp_idx1, imp_idx2):
    imp_idx1 = np.array(imp_idx1)
    imp_idx2 = np.array(imp_idx2)
    try:
        return np.all(imp_idx1 == imp_idx2)
    except ValueError:
        return False

def mf_or_cas_make_rdm1s(mf_or_cas):
    from pyscf.scf.hf import RHF
    from pyscf.scf.rohf import ROHF
    from pyscf.mcscf.mc1step import CASSCF
    # I don't know whether there is a general way to calculate rdm1s
    # If there is, better to use that function
    if isinstance(mf_or_cas, CASSCF): 
        print('DMET from CASSCF')
        dma, dmb = mf_or_cas.make_rdm1s()
    elif isinstance(mf_or_cas, ROHF):
        print('DMET from ROHF')
        dma, dmb = mf_or_cas.make_rdm1()
    elif isinstance(mf_or_cas, RHF):
        print('DMET from RHF')
        dma = mf_or_cas.make_rdm1()
        dmb = mf_or_cas.make_rdm1()
    else:
        raise TypeError('starting point not supported',  mf_or_cas.__class__)
    return dma, dmb

def lowdin_orth(mol, ovlp=None):
    # lowdin orthonormalize
    if ovlp is None:
        s = mol.intor_symmetric('int1e_ovlp')
    else:
        s = ovlp
    caolo, cloao = lowdin(s), lowdin(s) @ s # caolo=lowdin(s)=s^-1/2, cloao=lowdin(s)@s=s^1/2
    return caolo, cloao

def build_embeded_subspace(ldm, imp_idx,lo_meth='lowdin',thres=1e-13):
    """
    Returns C(AO->AS), entropy loss, and orbital composition
    """
    # from orthonormalized obital 

    # s = mf_or_cas.mol.intor_symmetric('int1e_ovlp')
    # caolo, cloao = lowdin(s), lowdin(s) @ s # caolo=lowdin(s)=s^-1/2, cloao=lowdin(s)@s=s^1/2
    env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]

    # dma, dmb = mf_or_cas_make_rdm1s(mf_or_cas) # in atomic orbital

    # ldma = reduce(np.dot,(cloao,dma,cloao.conj().T)) # in lowdin orbital
    # ldmb = reduce(np.dot,(cloao,dmb,cloao.conj().T))

    # ldm = ldma+ldmb

    # ldm = reduce(np.dot,(cloao,dm,cloao.conj().T))

    # ldma_env = ldma[env_idx,:][:,env_idx]
    # ldmb_env = ldmb[env_idx,:][:,env_idx]

    # nat_occa, nat_coeffa = np.linalg.eigh(ldma_env)
    # nat_occb, nat_coeffb = np.linalg.eigh(ldmb_env)

    ldm_imp = ldm[imp_idx,:][:,imp_idx]
    ldm_env = ldm[env_idx,:][:,env_idx]
    ldm_imp_env = ldm[imp_idx,:][:,env_idx]
    ldm_env_imp = ldm[env_idx,:][:,imp_idx]

    occ_env, orb_env = np.linalg.eigh(ldm_env) # occupation and orbitals on environment

    nimp = len(imp_idx)
    nfv = np.sum(occ_env <  thres) # frozen occupied
    nbath = np.sum((occ_env >= thres) & (occ_env <= 2-thres)) # bath orbital
    nfo = np.sum(occ_env > 2-thres) # frozen virtual 

    # defined w.r.t enviroment orbital index
    fv_idx = np.nonzero(occ_env <  thres)[0]
    bath_idx = np.nonzero((occ_env >= thres) & (occ_env <= 2-thres))[0]
    fo_idx = np.nonzero(occ_env > 2-thres)[0]

    orb_env = np.hstack((orb_env[:, bath_idx], orb_env[:, fo_idx], orb_env[:, fv_idx]))
    
    ldm_es = np.block([[ldm_imp, ldm_imp_env @ orb_env[:,0:nbath]],
                       [orb_env[:,0:nbath].T.conj() @ ldm_env_imp, orb_env[:,0:nbath].T.conj() @ ldm_env @ orb_env[:,0:nbath]]])
    es_occ, es_nat_orb = np.linalg.eigh(ldm_es)
    es_occ = es_occ[::-1]
    es_nat_orb = es_nat_orb[:,::-1]

    print('Number of bath orbitals', nbath)

    cloes = block_diag(np.eye(nimp), orb_env) @ block_diag(es_nat_orb, np.eye(nfo+nfv))
    # caoes = caolo @ cloes

    # asorbs = (nuu,ne,nuo)

    # nat_occa = nat_occa[nat_occa > thres]
    # nat_occb = nat_occb[nat_occb > thres]
    # nat_occr = nat_occr[nat_occr > thres]
    # nat_occa = nat_occa[nat_occa < 1-thres]
    # nat_occb = nat_occb[nat_occb < 1-thres]
    # nat_occr = nat_occr[nat_occr < 2-thres]

    # ent = - np.sum(nat_occa*np.log(nat_occa)) - np.sum(nat_occb*np.log(nat_occb))
    # ent2 = - np.sum((1-nat_occa)*np.log(1-nat_occa)) - np.sum((1-nat_occb)*np.log(1-nat_occb))
    # entr = -2*np.sum(nat_occr/2*np.log(nat_occr/2))
    # entr2 = -2*np.sum((1-nat_occr/2)*np.log(1-nat_occr/2))

    # es_occ = np.int32(es_occ>1-thres) + np.int32(es_occ>2-thres)

    return cloes, nimp, nbath, nfo, nfv, es_occ #, entr - ent, asorbs

def round_off_occ(mo_occ, threshold = 1e-8): 
    # round off occpuation close to 2 or 0 to be integral 
    mo_occ = np.where(np.abs(mo_occ-2)>threshold, mo_occ, int(2))
    mo_occ = np.where(np.abs(mo_occ)>threshold, mo_occ, int(0))
    return mo_occ

def make_es_1e_ints_old(mf,caoas,asorbs):
    # hcore from mf
    hcore = mf.get_hcore()

    # HF J/K from env UO
    uos = hcore.shape[0]-asorbs[-1]
    dm_uo = caoas[:,uos:] @ caoas[:,uos:].conj().T*2
    vj, vk = mf.get_jk(dm=dm_uo)

    fock = hcore + vj - 0.5 * vk 

    nimp = hcore.shape[0]-np.sum(asorbs)
    act_inds = [*list(range(nimp)),*list(range(nimp+asorbs[0],nimp+asorbs[0]+asorbs[1]))]

    asints1e = reduce(np.dot,(caoas[:,act_inds].conj().T,fock,caoas[:,act_inds]))
    return asints1e 

def make_es_int1e(mf_or_cas, fo_orb, es_orb):
    hcore = mf_or_cas.get_hcore() # DO NOT use get_hcore(mol), since x2c 1e term is not included

    # HF J/K from env frozen occupied orbital
    fo_dm = fo_orb @ fo_orb.T.conj()*2
    vj, vk = mf_or_cas.get_jk(mol=mf_or_cas.mol, dm=fo_dm)

    fock = hcore + vj - 0.5 * vk

    es_int1e = reduce(np.dot, (es_orb.T.conj(), fock, es_orb)) # AO to embedded space
    return es_int1e

def make_es_2e_ints_old(mf,caoas,asorbs):
    nimp = caoas.shape[0]-np.sum(asorbs)
    act_inds = [*list(range(nimp)),*list(range(nimp+asorbs[0],nimp+asorbs[0]+asorbs[1]))]

    if mf._eri is not None:
        asints2e = ao2mo.full(mf._eri,caoas[:,act_inds])
    else:
        asints2e = ao2mo.full(mf.mol,caoas[:,act_inds])

    asints2e = ao2mo.full(mf.mol,caoas[:,act_inds])

    return asints2e 

def make_es_int2e(mf, es_orb):
    # if mf._eri is not None:
    #     es_int2e = ao2mo.full(mf._eri, es_orb)
    # else:
    #     es_int2e = ao2mo.full(mf.mol, es_orb)
    es_int2e = ao2mo.full(mf.mol, es_orb)
    return es_int2e

from pyscf import lib
from pyscf.lib import logger

class SSDMET(lib.StreamObject):
    """
    single-shot DMET with impurity-environment partition
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-13, max_mem=64000,verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = max_mem # TODO
        self.verbose = verbose # TODO

        # inputs
        self.dm = None
        self._imp_idx = []
        if imp_idx is not None:
            self.imp_idx = imp_idx
        else:
            self.imp_idx = self.mol.atom_symbol(0)
        self.threshold = threshold

        # NOT inputs
        self.fo_orb = None
        self.fv_orb = None
        self.es_orb = None
        self.es_occ = None

        self.nfo = None
        self.nfv = None
        self.nes = None

        self.es_int1e = None
        self.es_int2e = None

        self.es_mf = None

    @property
    def imp_idx(self):
        return self._imp_idx
    
    @imp_idx.setter
    def imp_idx(self, imp_idx):
        self._imp_idx = gto.mole._aolabels2baslst(self.mol, imp_idx, base=0)

    def make_es_int1e(self):
        return make_es_int1e(self.mf_or_cas, self.fo_orb, self.es_orb)

    def make_es_int2e(self):
        return make_es_int2e(self.mf_or_cas, self.es_orb)
    
    def load_chk(self, chk_fname):
        try:
            if not '_dmet_chk.h5' in chk_fname:
                chk_fname = chk_fname + '_dmet_chk.h5'
            if not os.path.isfile(chk_fname):
                return False
        except:
            return False

        print(f'load chk file {chk_fname}')
        with h5py.File(chk_fname, 'r') as fh5:
            dm_check = np.allclose(self.dm, fh5['dm'][:], atol=1e-5)
            imp_idx_check = compare_imp_idx(self.imp_idx, fh5['imp_idx'][:])
            threshold_check = self.threshold == fh5['threshold'][()]
            if dm_check & imp_idx_check & threshold_check:
                self.fo_orb = fh5['fo_orb'][:]
                self.fv_orb = fh5['fv_orb'][:]
                self.es_orb = fh5['es_orb'][:]
                self.es_occ = fh5['es_occ'][:]
                self.es_int1e = fh5['es_int1e'][:]
                self.es_int2e = fh5['es_int2e'][:]

                self.nfo = np.shape(self.fo_orb)[1]
                self.nfv = np.shape(self.fv_orb)[1]
                self.nes = np.shape(self.es_orb)[1]
                return True
            else:
                print(f'density matrix check {dm_check}')
                print(f'impurity index check {imp_idx_check}')
                print(f'threshold check {threshold_check}')
                print(f'build dmet subspace with imp idx {self.imp_idx} threshold {self.threshold}')
                return False
    
    def save_chk(self, chk_fname):
        with h5py.File(chk_fname, 'w') as fh5:
            fh5['dm'] = self.dm
            fh5['imp_idx'] = self.imp_idx
            fh5['threshold'] = self.threshold

            fh5['fo_orb'] = self.fo_orb
            fh5['fv_orb'] = self.fv_orb
            fh5['es_orb'] = self.es_orb
            fh5['es_occ'] = self.es_occ
            fh5['es_int1e'] = self.es_int1e
            fh5['es_int2e'] = self.es_int2e
        return 
    
    def lowdin_orth(self):
        # lowdin orthonormalize
        caolo, cloao = lowdin_orth(self.mol)
        ldm = reduce(np.dot,(cloao,self.dm,cloao.conj().T))
        return ldm, caolo, cloao
        
    def build(self, chk_fname_load='', save_chk=True):
        self.dm = self.mf_or_cas.make_rdm1()
        if np.shape(np.shape(self.dm))[0] == 3: # ROHF density matrix have dimension (2, nao, nao)
            self.dm = self.dm[0] + self.dm[1]

        loaded = self.load_chk(chk_fname_load)
        
        if not loaded:
            ldm, caolo, cloao = self.lowdin_orth()

            cloes, nimp, nbath, nfo, nfv, self.es_occ = build_embeded_subspace(ldm, self.imp_idx, thres=self.threshold)
            caoes = caolo @ cloes

            self.fo_orb = caoes[:, nimp+nbath: nimp+nbath+nfo]
            self.fv_orb = caoes[:, nimp+nbath+nfo: nimp+nbath+nfo+nfv]
            self.es_orb = caoes[:, :nimp+nbath]
        
            self.nfo = nfo
            self.nfv = nfv
            self.nes = nimp + nbath

            self.es_int1e = self.make_es_int1e()
            self.es_int2e = self.make_es_int2e()

        chk_fname_save = self.title + '_dmet_chk.h5'
        if save_chk:
            self.save_chk(chk_fname_save)
        self.es_mf = self.ROHF()
        return self.es_mf
    
    def ROHF(self, run_mf=False):
        mol = gto.M()
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelectron = self.mf_or_cas.mol.nelectron - 2*self.nfo
        mol.spin = self.mol.spin

        es_mf = scf.rohf.ROHF(mol).x2c()
        es_mf.max_memory = self.max_mem
        es_mf.mo_energy = np.zeros((self.nes))

        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: np.eye(self.nes)
        es_mf._eri = ao2mo.restore(8, self.es_int2e, self.nes)

        es_dm = np.zeros((2, self.nes, self.nes))
        es_dm[0] = np.diag(np.int32(self.es_occ>1-1e-3))
        es_dm[1] = np.diag(np.int32(self.es_occ>2-1e-3))
        # assume we only perfrom ROHF-in-ROHF embedding

        # assert np.einsum('ijj->', es_dm) == mol.nelectron
        if run_mf:
            es_mf.kernel(es_dm)
            self.es_occ = es_mf.mo_occ
        return es_mf
    
    def avas(self, aolabels, **kwargs):
        from embed_sim import myavas
        total_mf = self.total_mf()
        total_mf.mo_occ = round_off_occ(total_mf.mo_occ) # make 2/0 occupation to be int
        ncas, nelec, mo = myavas.avas(total_mf, aolabels, ncore=self.nfo, nunocc = self.nfv, canonicalize=False, **kwargs) # canonicalize should be set to False, since it require orbital energy

        es_mo = self.es_orb.T.conj() @ self.mol.intor_symmetric('int1e_ovlp') @ mo[:, self.nfo: self.nfo+self.nes]
        return ncas, nelec, es_mo 
    
    def total_mf(self):
        total_mf = scf.rohf.ROHF(self.mol).x2c()
        total_mf.mo_coeff = np.hstack((self.fo_orb, self.es_orb, self.fv_orb))
        total_mf.mo_occ = np.hstack((2*np.ones(self.nfo), self.es_occ, np.zeros(self.nfv)))
        return total_mf
    
    def total_cas(self, es_cas):
        from embed_sim import sacasscf_mixer
        total_cas = sacasscf_mixer.sacasscf_mixer(self.mf_or_cas, es_cas.ncas, es_cas.nelecas)
        total_cas.fcisolver = es_cas.fcisolver
        total_cas.ci = es_cas.ci
        total_cas.mo_coeff = np.hstack((self.fo_orb, self.es_orb @ es_cas.mo_coeff, self.fv_orb))
        return total_cas