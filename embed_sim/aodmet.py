import numpy as np
from functools import reduce
from scipy.linalg import block_diag
import h5py
import os

from pyscf.lo.orth import lowdin
from pyscf import gto, scf, ao2mo, lib
from pyscf.lib import logger

from embed_sim import ssdmet
from embed_sim.BNO_bath import get_RMP2_bath, get_UMP2_bath, get_ROMP2_bath

def lowdin_orth(mol, imp_idx, ovlp=None):
    # lowdin orthonormalize
    if ovlp is None:
        s = mol.intor_symmetric('int1e_ovlp')
    else:
        s = ovlp
    env_idx = np.array([x for x in range(mol.nao) if x not in imp_idx])
    caolo = np.eye(mol.nao)
    cloao = np.eye(mol.nao)
    s_env = s[env_idx,:][:,env_idx]
    caolo_env = lowdin(s_env)
    cloao_env = lib.dot(caolo_env, s_env)
    caolo[np.ix_(env_idx,env_idx)] = caolo_env
    cloao[np.ix_(env_idx,env_idx)] = cloao_env
    return caolo, cloao, s
    

def build_embeded_subspace(ldm, imp_idx, caolo, ovlp, lo_meth='lowdin', thres=1e-12, es_natorb=True):
    """
    Returns C(AO->AS) and orbital composition
    """
    env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]

    ldm_imp = ldm[imp_idx,:][:,imp_idx]
    ldm_env = ldm[env_idx,:][:,env_idx]
    ldm_imp_env = ldm[imp_idx,:][:,env_idx]
    ldm_env_imp = ldm[env_idx,:][:,imp_idx]

    occ_env, orb_env = np.linalg.eigh(ldm_env) # occupation and orbitals on environment

    nimp = len(imp_idx)
    nfv = np.sum(occ_env <  thres) # frozen virtual 
    nbath = np.sum(((occ_env >= thres) & (occ_env <= 2-thres)) | (occ_env >= 2+thres)) # bath orbital
    nfo = np.sum((occ_env > 2-thres) & (occ_env < 2+thres)) # frozen occupied

    # defined w.r.t enviroment orbital index
    fv_idx = np.nonzero(occ_env <  thres)[0]
    bath_idx = np.nonzero(((occ_env >= thres) & (occ_env <= 2-thres)) | (occ_env >= 2+thres))[0]
    fo_idx = np.nonzero((occ_env > 2-thres) & (occ_env < 2+thres))[0]

    orb_env = np.hstack((orb_env[:, bath_idx], orb_env[:, fo_idx], orb_env[:, fv_idx]))
    
    if es_natorb:
        ldm_es = np.block([[ldm_imp, ldm_imp_env @ orb_env[:,0:nbath]],
                           [orb_env[:,0:nbath].T.conj() @ ldm_env_imp, orb_env[:,0:nbath].T.conj() @ ldm_env @ orb_env[:,0:nbath]]])

        s_rearange = reduce(np.dot,(caolo.conj().T,ovlp,caolo))
        s_imp = s_rearange[imp_idx,:][:,imp_idx]
        s_bath = orb_env[:, 0:nbath].conj().T @ s_rearange[env_idx,:][:,env_idx] @ orb_env[:, 0:nbath]
        s_imp_bath = s_rearange[imp_idx,:][:,env_idx] @ orb_env[:, 0:nbath]
        s_bath_imp = orb_env[:, 0:nbath].conj().T @ s_rearange[env_idx,:][:,imp_idx]
        s_es = np.block([[s_imp, s_imp_bath],
                         [s_bath_imp, s_bath]])
        orth_es, orth_es_inv = lowdin(s_es), lowdin(s_es)@s_es
        ldm_es = reduce(np.dot,(orth_es_inv, ldm_es, orth_es_inv.conj().T))

        es_occ, es_nat_orb = np.linalg.eigh(ldm_es)
        es_occ = es_occ[::-1]
        es_nat_orb = es_nat_orb[:,::-1]

        cloes = block_diag(np.eye(nimp), orb_env) @ block_diag(orth_es@es_nat_orb, np.eye(nfo+nfv))
    else:
        es_occ = None
        cloes = block_diag(np.eye(nimp), orb_env)
    
    rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
    cloes = cloes[rearange_idx, :]

    return cloes, nimp, nbath, nfo, nfv, es_occ

def round_off_occ(mo_occ, threshold = 1e-8): 
    # round off occpuation close to 2 or 0 to be integral 
    mo_occ = np.where(np.abs(mo_occ-2)>threshold, mo_occ, int(2))
    mo_occ = np.where(np.abs(mo_occ-1)>threshold, mo_occ, int(1))
    mo_occ = np.where(np.abs(mo_occ)>threshold, mo_occ, int(0))
    return mo_occ

class AODMET(ssdmet.SSDMET):
    """
    single-shot AO-DMET with impurity-environment partition
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-12, es_natorb=True, bath_option=None, verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = mf_or_cas.max_memory
        self.verbose = verbose
        self.log = lib.logger.new_logger(self.mol, self.verbose)

        # inputs
        self.dm = None
        self._imp_idx = []
        if imp_idx is not None:
            self.imp_idx = imp_idx
        else:
            print('impurity index not assigned, use the first atom as impurity')
            self.imp_idx = self.mol.atom_symbol(0)
        self.threshold = threshold
        self.es_natorb = es_natorb
        self.bath_option = bath_option

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

    def dump_flags(self):
        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', self.__class__)

    @property
    def imp_idx(self):
        return self._imp_idx
    
    @imp_idx.setter
    def imp_idx(self, imp_idx):
        self._imp_idx = gto.mole._aolabels2baslst(self.mol, imp_idx, base=0)
    
    def lowdin_orth(self):
        # lowdin orthonormalize
        caolo, cloao, ovlp = lowdin_orth(self.mol, self.imp_idx)
        ldm = reduce(lib.dot, (cloao, self.dm, cloao.conj().T))
        return ldm, caolo, cloao, ovlp
        
    def build(self, chk_fname_load='', save_chk=True):
        self.dump_flags()
        dm = ssdmet.mf_or_cas_make_rdm1s(self.mf_or_cas)
        if dm.ndim == 3: # ROHF density matrix have dimension (2, nao, nao)
            self.dm = dm[0] + dm[1]
            open_shell = True
        else:
            self.dm = dm
            open_shell = False

        loaded = self.load_chk(chk_fname_load)
        
        if not loaded:
            ldm, caolo, cloao, ovlp = self.lowdin_orth()

            cloes, nimp, nbath, nfo, nfv, self.es_occ = build_embeded_subspace(ldm, self.imp_idx, caolo, ovlp, thres=self.threshold, es_natorb=self.es_natorb)
            caoes = caolo @ cloes

            self.fo_orb = caoes[:, nimp+nbath: nimp+nbath+nfo]
            self.fv_orb = caoes[:, nimp+nbath+nfo: nimp+nbath+nfo+nfv]
            self.es_orb = caoes[:, :nimp+nbath]
        
            self.nfo = nfo
            self.nfv = nfv
            self.nes = nimp + nbath
            self.log.info(f'number of impurity orbitals = {nimp}')
            self.log.info(f'number of bath orbitals = {nbath}')
            self.log.info(f'number of embedded cluster orbitals = {nimp+nbath}')
            self.log.info(f'number of frozen occupied orbitals = {nfo}')
            self.log.info(f'number of frozen virtual orbitals = {nfv}')
            self.log.info(f'number of frozen orbitals = {nfo+nfv}')
            self.log.info(f'percentage of embedded cluster orbitals = {((nimp+nbath)/self.mol.nao)*100:.2f}%%')
            self.log.info(f'percentage of frozen orbitals = {((nfo+nfv)/self.mol.nao)*100:.2f}%%')

            self.es_int1e = self.make_es_int1e()
            self.es_int2e = self.make_es_int2e()

            self.es_dm = self.make_es_dm(open_shell, cloes[:, :nimp+nbath], cloao, dm)

            if self.bath_option is not None:
                self.log.info('')
                if self.es_natorb:
                    raise RuntimeError('es_natorb must be turned off when using extra bath_option')
                lo2core = cloes[:, nimp+nbath: nimp+nbath+nfo]
                lo2vir = cloes[:, nimp+nbath+nfo: nimp+nbath+nfo+nfv]
                if isinstance(self.bath_option, dict):
                    if len(self.bath_option.keys()) == 1:
                        if 'MP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.info('ROMP2 bath expansion in used by default')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['MP2'])
                            else:
                                self.log.info('RMP2 bath expansion in used by default')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['MP2'])
                        elif 'RMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.info('ROMP2 bath expansion in used by default')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['RMP2'])
                            else:
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['RMP2'])
                        elif 'ROMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['ROMP2'])
                            else:
                                self.log.info('ROMP2 bath expansion is degraded to RMP2 for closed-shell systems')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['ROMP2'])
                        elif 'UMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.warn('UMP2 bath expansion is less preferred than ROMP2, the results must be checked carefully!')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_UMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['UMP2'])
                            else:
                                self.log.info('UMP2 bath expansion is degraded to RMP2 for closed-shell systems')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['UMP2'])
                        else:
                            raise NotImplementedError('Currently only MP2, RMP2, ROMP2 and UMP2 are supported')
                    else:
                        raise NotImplementedError('Only one key should be in bath_option')
                else:
                    raise NotImplementedError('The bath_option should be a dictionary')
                
                lo2eo = np.hstack([cloes[:, :nimp+nbath], lo2MP2_bath])
                self.es_orb = lib.dot(caolo, lo2eo)
                self.fo_orb = lib.dot(caolo, lo2MP2_core)
                self.fv_orb = lib.dot(caolo, lo2MP2_vir)

                nbath += lo2MP2_bath.shape[-1]
                nfo = self.fo_orb.shape[-1]
                nfv = self.fv_orb.shape[-1]
                self.nfo = nfo
                self.nfv = nfv
                self.nes = nimp + nbath
                self.log.info(f'number of impurity orbitals = {nimp}')
                self.log.info(f'number of bath orbitals = {nbath}')
                self.log.info(f'number of embedded cluster orbitals = {nimp+nbath}')
                self.log.info(f'number of frozen occupied orbitals = {nfo}')
                self.log.info(f'number of frozen virtual orbitals = {nfv}')
                self.log.info(f'number of frozen orbitals = {nfo+nfv}')
                self.log.info(f'percentage of embedded cluster orbitals = {((nimp+nbath)/self.mol.nao)*100:.2f}%%')
                self.log.info(f'percentage of frozen orbitals = {((nfo+nfv)/self.mol.nao)*100:.2f}%%')

                self.es_int1e = self.make_es_int1e()
                self.es_int2e = self.make_es_int2e()
                self.es_dm = self.make_es_dm(open_shell, lo2eo, cloao, dm)
            else:
                pass
        
        self.es_mf = self.ROHF()
        self.fo_ene()
        self.log.info('')
        self.log.info(f'energy from frozen occupied orbitals = {self.fo_ene}')
        self.log.info(f'deviation from DMET exact condition = {self.es_mf.e_tot+self.fo_ene-self.mf_or_cas.e_tot}')

        if save_chk:
            chk_fname_save = self.title + '_dmet_chk.h5'
            self.save_chk(chk_fname_save)
        return self.es_mf
    
    def avas(self, aolabels, *args, **kwargs):
        from embed_sim import myavas
        total_mf = self.total_mf()
        total_mf.mo_occ = round_off_occ(total_mf.mo_occ) # make 2/0 occupation to be int
        ncas, nelec, mo = myavas.avas(total_mf, aolabels, ncore=self.nfo, nunocc = self.nfv, canonicalize=False, *args, **kwargs) # canonicalize should be set to False, since it require orbital energy

        ldm, caolo, cloao, ovlp = self.lowdin_orth()
        es_mo = reduce(lib.dot, (self.es_orb.T.conj(), cloao.T, cloao, mo[:, self.nfo: self.nfo+self.nes]))
        return ncas, nelec, es_mo 
    
    def density_fit(self, with_df=None):
        from embed_sim.df import DFAODMET
        if with_df is None:
            if not getattr(self.mf_or_cas, 'with_df', False):
                raise NotImplementedError
            else:
                with_df = self.mf_or_cas.with_df
        return DFAODMET(self.mf_or_cas, self.title, imp_idx=self.imp_idx, threshold=self.threshold,
                        with_df=with_df, es_natorb=self.es_natorb, bath_option=self.bath_option, verbose=self.verbose)
