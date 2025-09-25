import numpy as np
import h5py
import tempfile
import os
import itertools 
from functools import reduce

from pyscf import __config__
from pyscf import gto, scf, lib, ao2mo, df, fci
from pyscf.mcscf import casci, mc1step
from pyscf.mrpt.nevpt2 import (Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir,
                               NEVPT, _contract4pdm)
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf.data import nist
from pyscf.lib import logger

from embed_sim import ssdmet, aodmet, siso
from embed_sim.BNO_bath import get_RMP2_bath, get_UMP2_bath, get_ROMP2_bath

def make_es_cderi(title, es_orb, with_df):
    erifile = title+'_es_cderi.h5'
    dataname = 'j3c'
    feri = df.outcore._create_h5file(erifile, dataname)
    ijmosym, nij_pair, moij, ijslice = _conc_mos(es_orb, es_orb, True)
    naux = with_df.get_naoaux()
    neo = es_orb.shape[-1]
    nao_pair = neo*(neo+1)//2
    label = '%s/%d'%(dataname, 0)
    feri[label] = np.zeros((naux,nao_pair),dtype=np.float64)
    nij = 0
    for eri1 in with_df.loop():
        Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym)
        nrow = Lij.shape[0]
        feri[label][nij:nij+nrow] = Lij
        nij += nrow
    return erifile

class DFSSDMET(ssdmet.SSDMET):
    """
    Density fitting single-shot DMET class
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-12, with_df=None, es_natorb=True, bath_option=None, verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = mf_or_cas.max_memory # TODO
        self.verbose = verbose # TODO
        self.with_df = with_df
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
        self.es_cderi = None

        self.es_mf = None
    
    def make_es_cderi(self):
        return make_es_cderi(self.title, self.es_orb, self.with_df)
    
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
            imp_idx_check = ssdmet.compare_imp_idx(self.imp_idx, fh5['imp_idx'][:])
            threshold_check = self.threshold == fh5['threshold'][()]
            if dm_check & imp_idx_check & threshold_check:
                self.fo_orb = fh5['fo_orb'][:]
                self.fv_orb = fh5['fv_orb'][:]
                self.es_orb = fh5['es_orb'][:]
                self.es_occ = fh5['es_occ'][:]
                self.es_int1e = fh5['es_int1e'][:]
                self.es_cderi = self.title+'_es_cderi.h5'
                self.es_dm = fh5['es_dm'][:]

                self.nfo = np.shape(self.fo_orb)[1]
                self.nfv = np.shape(self.fv_orb)[1]
                self.nes = np.shape(self.es_orb)[1]
                return True
            else:
                self.log.info(f'density matrix check {dm_check}')
                self.log.info(f'impurity index check {imp_idx_check}')
                self.log.info(f'threshold check {threshold_check}')
                self.log.info(f'build dmet subspace with imp idx {self.imp_idx} threshold {self.threshold}')
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
            fh5['es_dm'] = self.es_dm
        return 

    def build(self, restore_imp = False, chk_fname_load='', save_chk=True):
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
            ldm, caolo, cloao = self.lowdin_orth(restore_imp)

            cloes, nimp, nbath, nfo, nfv, self.es_occ = ssdmet.build_embeded_subspace(ldm, self.imp_idx, thres=self.threshold)
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
            self.es_cderi = self.make_es_cderi()

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
                self.es_cderi = self.make_es_cderi()
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
    
    def ROHF(self):
        mol = gto.M()
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelectron = self.mf_or_cas.mol.nelectron - 2*self.nfo
        mol.spin = self.mol.spin

        if mol.spin != 0:
            es_mf = scf.ROHF(mol).x2c().density_fit()
        else:
            es_mf = scf.RHF(mol).x2c().density_fit()
        es_mf.max_memory = self.max_mem
        es_mf.mo_energy = np.zeros((self.nes))

        es_ovlp = reduce(lib.dot, (self.es_orb.conj().T, self.mol.intor_symmetric('int1e_ovlp'), self.es_orb))
        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: es_ovlp
        es_mf.with_df._cderi = self.es_cderi

        # assume we only perfrom ROHF-in-ROHF embedding

        # assert lib.einsum('ijj->', es_dm) == mol.nelectron
        es_mf.level_shift = self.mf_or_cas.level_shift
        es_mf.conv_check = False
        
        es_fock = es_mf.get_fock(dm=self.es_dm)
        mo_energy, mo_coeff = es_mf.eig(es_fock, es_ovlp)
        mo_occ = es_mf.get_occ(mo_energy, mo_coeff)
        es_mf.mo_energy = mo_energy
        es_mf.mo_coeff = mo_coeff
        es_mf.mo_occ = mo_occ
        es_mf.e_tot = es_mf.energy_tot()
        self.es_occ = es_mf.mo_occ
        return es_mf
    
class DFAODMET(aodmet.AODMET):
    """
    Density fitting single-shot AO-DMET class
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-12, with_df=None, es_natorb=True, bath_option=None, verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = mf_or_cas.max_memory # TODO
        self.verbose = verbose # TODO
        self.with_df = with_df
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
        self.es_cderi = None

        self.es_mf = None
    
    def make_es_cderi(self):
        return make_es_cderi(self.title, self.es_orb, self.with_df)
    
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
            imp_idx_check = ssdmet.compare_imp_idx(self.imp_idx, fh5['imp_idx'][:])
            threshold_check = self.threshold == fh5['threshold'][()]
            if dm_check & imp_idx_check & threshold_check:
                self.fo_orb = fh5['fo_orb'][:]
                self.fv_orb = fh5['fv_orb'][:]
                self.es_orb = fh5['es_orb'][:]
                self.es_occ = fh5['es_occ'][:]
                self.es_int1e = fh5['es_int1e'][:]
                self.es_cderi = self.title+'_es_cderi.h5'

                self.nfo = np.shape(self.fo_orb)[1]
                self.nfv = np.shape(self.fv_orb)[1]
                self.nes = np.shape(self.es_orb)[1]
                return True
            else:
                self.log.info(f'density matrix check {dm_check}')
                self.log.info(f'impurity index check {imp_idx_check}')
                self.log.info(f'threshold check {threshold_check}')
                self.log.info(f'build dmet subspace with imp idx {self.imp_idx} threshold {self.threshold}')
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
        return 

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

            cloes, nimp, nbath, nfo, nfv, self.es_occ = aodmet.build_embeded_subspace(ldm, self.imp_idx, caolo, ovlp, thres=self.threshold, es_natorb=self.es_natorb)
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
            self.es_cderi = self.make_es_cderi()

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
                self.es_cderi = self.make_es_cderi()
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
    
    def ROHF(self):
        mol = gto.M()
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelectron = self.mf_or_cas.mol.nelectron - 2*self.nfo
        mol.spin = self.mol.spin

        if mol.spin != 0:
            es_mf = scf.ROHF(mol).x2c().density_fit()
        else:
            es_mf = scf.RHF(mol).x2c().density_fit()
        es_mf.max_memory = self.max_mem
        es_mf.mo_energy = np.zeros((self.nes))

        es_ovlp = reduce(lib.dot, (self.es_orb.conj().T, self.mol.intor_symmetric('int1e_ovlp'), self.es_orb))
        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: es_ovlp
        es_mf.with_df._cderi = self.es_cderi

        # assume we only perfrom ROHF-in-ROHF embedding

        # assert lib.einsum('ijj->', es_dm) == mol.nelectron
        es_mf.level_shift = self.mf_or_cas.level_shift
        es_mf.conv_check = False

        es_fock = es_mf.get_fock(dm=self.es_dm)
        mo_energy, mo_coeff = es_mf.eig(es_fock, es_ovlp)
        mo_occ = es_mf.get_occ(mo_energy, mo_coeff)
        es_mf.mo_energy = mo_energy
        es_mf.mo_coeff = mo_coeff
        es_mf.mo_occ = mo_occ
        es_mf.e_tot = es_mf.energy_tot()
        self.es_occ = es_mf.mo_occ
        return es_mf
    
class DFNEVPT(NEVPT):
    _keys = {
        'ncore', 'root', 'compressed_mps', 'e_corr', 'canonicalized', 'onerdm',
    }.union(casci.CASBase._keys, mc1step.CASSCF._keys)

    def __init__(self, mc, root=0, spin=0, dump_flags_verbose=4):
        super().__init__(mc, root)
        self.spin = spin
        self.dump_flags_verbose = dump_flags_verbose

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('DFNEVPT2 (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        log.info('spin = %d  root = %d', self.spin, self.root)
        
    def kernel(self):
        self.dump_flags(self.dump_flags_verbose)
        from pyscf.mcscf.addons import StateAverageFCISolver
        if isinstance(self.fcisolver, StateAverageFCISolver):
            raise RuntimeError('State-average FCI solver object cannot be used '
                               'in NEVPT2 calculation.\nA separated multi-root '
                               'CASCI calculation is required for NEVPT2 method. '
                               'See examples/mrpt/41-for_state_average.py.')

        if getattr(self._mc, 'frozen', None) is not None:
            raise NotImplementedError

        if isinstance(self.verbose, logger.Logger):
            log = self.verbose
        else:
            log = logger.Logger(self.stdout, self.verbose)
        time0 = (logger.process_clock(), logger.perf_counter())
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas

        #By defaut, _mc is canonicalized for the first root.
        #For SC-NEVPT based on compressed MPS perturber functions, _mc was already canonicalized.
        if (not self.canonicalized):
            # Need to assign roots differently if we have more than one root
            # See issue #1081 (https://github.com/pyscf/pyscf/issues/1081) for more details
            self.mo_coeff, single_ci_vec, self.mo_energy = self.canonicalize(
                self.mo_coeff, ci=self.load_ci(), cas_natorb=True, verbose=self.verbose)
            if self.fcisolver.nroots == 1:
                self.ci = single_ci_vec
            else:
                self.ci[self.root] = single_ci_vec

        if getattr(self.fcisolver, 'nevpt_intermediate', None):
            logger.info(self, 'DMRG-NEVPT')
            dm1, dm2, dm3 = self.fcisolver._make_dm123(self.load_ci(),ncas,self.nelecas,None)
        else:
            dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf',
                                               self.load_ci(), self.load_ci(), ncas, self.nelecas)
        dm4 = None

        dms = {
            '1': dm1, '2': dm2, '3': dm3, '4': dm4,
            # 'h1': hdm1, 'h2': hdm2, 'h3': hdm3
        }
        time1 = log.timer('3pdm, 4pdm', *time0)

        eris = _ERIS(self, self.mo_coeff)
        time1 = log.timer('integral transformation', *time1)

        if not getattr(self.fcisolver, 'nevpt_intermediate', None):  # regular FCI solver
            link_indexa = fci.cistring.gen_linkstr_index(range(ncas), self.nelecas[0])
            link_indexb = fci.cistring.gen_linkstr_index(range(ncas), self.nelecas[1])
            aaaa = eris['ppaa'][ncore:nocc,ncore:nocc].copy()
            f3ca = _contract4pdm('NEVPTkern_cedf_aedf', aaaa, self.load_ci(), ncas,
                                 self.nelecas, (link_indexa,link_indexb))
            f3ac = _contract4pdm('NEVPTkern_aedf_ecdf', aaaa, self.load_ci(), ncas,
                                 self.nelecas, (link_indexa,link_indexb))
            dms['f3ca'] = f3ca
            dms['f3ac'] = f3ac
        time1 = log.timer('eri-4pdm contraction', *time1)

        if self.compressed_mps:
            from pyscf.dmrgscf.nevpt_mpi import DMRG_COMPRESS_NEVPT
            if self.stored_integral: #Stored perturbation integral and read them again. For debugging purpose.
                perturb_file = DMRG_COMPRESS_NEVPT(self, maxM=self.maxM, root=self.root,
                                                   nevptsolver=self.nevptsolver,
                                                   tol=self.tol,
                                                   nevpt_integral='nevpt_perturb_integral')
            else:
                perturb_file = DMRG_COMPRESS_NEVPT(self, maxM=self.maxM, root=self.root,
                                                   nevptsolver=self.nevptsolver,
                                                   tol=self.tol)
            fh5 = h5py.File(perturb_file, 'r')
            e_Si     =   fh5['Vi/energy'][()]
            #The definition of norm changed.
            #However, there is no need to print out it.
            #Only perturbation energy is wanted.
            norm_Si  =   fh5['Vi/norm'][()]
            e_Sr     =   fh5['Vr/energy'][()]
            norm_Sr  =   fh5['Vr/norm'][()]
            fh5.close()
            logger.note(self, "Sr    (-1)',   E = %.14f",  e_Sr  )
            logger.note(self, "Si    (+1)',   E = %.14f",  e_Si  )

        else:
            norm_Sr   , e_Sr    = Sr(self, self.load_ci(), dms, eris)
            logger.note(self, "Sr    (-1)',   E = %.14f",  e_Sr  )
            time1 = log.timer("space Sr (-1)'", *time1)
            norm_Si   , e_Si    = Si(self, self.load_ci(), dms, eris)
            logger.note(self, "Si    (+1)',   E = %.14f",  e_Si  )
            time1 = log.timer("space Si (+1)'", *time1)
        norm_Sijrs, e_Sijrs = Sijrs(self, eris)
        logger.note(self, "Sijrs (0)  ,   E = %.14f", e_Sijrs)
        time1 = log.timer('space Sijrs (0)', *time1)
        norm_Sijr , e_Sijr  = Sijr(self, dms, eris)
        logger.note(self, "Sijr  (+1) ,   E = %.14f",  e_Sijr)
        time1 = log.timer('space Sijr (+1)', *time1)
        norm_Srsi , e_Srsi  = Srsi(self, dms, eris)
        logger.note(self, "Srsi  (-1) ,   E = %.14f",  e_Srsi)
        time1 = log.timer('space Srsi (-1)', *time1)
        norm_Srs  , e_Srs   = Srs(self, dms, eris)
        logger.note(self, "Srs   (-2) ,   E = %.14f",  e_Srs )
        time1 = log.timer('space Srs (-2)', *time1)
        norm_Sij  , e_Sij   = Sij(self, dms, eris)
        logger.note(self, "Sij   (+2) ,   E = %.14f",  e_Sij )
        time1 = log.timer('space Sij (+2)', *time1)
        norm_Sir  , e_Sir   = Sir(self, dms, eris)
        logger.note(self, "Sir   (0)' ,   E = %.14f",  e_Sir )
        time1 = log.timer("space Sir (0)'", *time1)

        nevpt_e  = e_Sr + e_Si + e_Sijrs + e_Sijr + e_Srsi + e_Srs + e_Sij + e_Sir
        logger.note(self, "Nevpt2 Energy = %.15f", nevpt_e)
        log.timer('SC-NEVPT2', *time0)

        self.e_corr = nevpt_e
        return nevpt_e

# register NEVPT2 in MCSCF
casci.CASBase.NEVPT2 = DFNEVPT

def _ERIS(mc, mo):
    ncore = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nocc = ncore + ncas
    nav = nmo - ncore
    nvir = nmo - nocc
    
    temp_file = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR, delete=True)
    filename = temp_file.name
    erifile = h5py.File(filename, mode='r+')
    ppaa = erifile.create_dataset('ppaa', (nmo,nmo,ncas,ncas), 'f8')
    papa = erifile.create_dataset('papa', (nmo,ncas,nmo,ncas), 'f8')
    pacv = erifile.create_dataset('pacv', (nmo,ncas,ncore,nvir), 'f8')
    cvcv = erifile.create_dataset('cvcv', (ncore*nvir,ncore*nvir), 'f8')
    
    ijmosym, nij_pair, moij, ijslice = _conc_mos(mo, mo, True)
    for eri1 in mc._scf.with_df.loop():
        Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym)
        Lij = lib.unpack_tril(Lij)
        ppaa[:] += lib.einsum('Pij,Pkl->ijkl', Lij, Lij[:,ncore:nocc,ncore:nocc])
        papa[:] += lib.einsum('Pij,Pkl->ijkl', Lij[:,:,ncore:nocc], Lij[:,:,ncore:nocc])
        pacv[:] += lib.einsum('Pij,Pkl->ijkl', Lij[:,:,ncore:nocc], Lij[:,:ncore,nocc:])
        max_memory = int(mc.max_memory - lib.current_memory()[0])
        nvcv = nvir*ncore*nvir
        blksize = max(1, int(max_memory*.35e6/8/nvcv//10))
        for cstart, cend in lib.prange(0, ncore, blksize):
            nc = cend - cstart
            cvcv[cstart*nvir:cend*nvir] += lib.dot(Lij[:,cstart:cend,nocc:].reshape(-1,nc*nvir).T,Lij[:,:ncore,nocc:].reshape(-1,ncore*nvir))

    dmcore = lib.dot(mo[:,:ncore], mo[:,:ncore].T)
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(lib.dot, (mo.T, vj*2-vk, mo))

    eris = {}
    eris['vhf_c'] = vhfcore
    eris['ppaa'] = ppaa
    eris['papa'] = papa
    eris['pacv'] = pacv
    eris['cvcv'] = cvcv
    eris['h1eff'] = reduce(lib.dot, (mo.T, mc.get_hcore(), mo)) + vhfcore
    return eris

def auxe2(mol, auxmol, title, int3c='int3c2e_pvxp1', aosym='s1', comp=3, verbose=5):
    feri_name = title+'_'+int3c+'.h5'
    if not os.path.exists(feri_name):
        log = logger.Logger(mol.stdout, verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        df.outcore.cholesky_eri_b(mol, feri_name, auxbasis=auxmol.basis, int3c=int3c, aosym=aosym, comp=comp, verbose=verbose)
        t0 = log.timer('int3c2e_pvxp1', *t0)
    else:
        print('Load from {}'.format(feri_name))
    return

class DFSISO(siso.SISO):
    def __init__(self, title, mc, statelis=None, save_mag=True, save_Hmat=False, save_old_Hal=False, verbose=5, with_df=None):
        self.title = title
        self.mol = mc.mol
        self.mc = mc
        self.with_df = with_df

        # if statelis is None:
        #     statelis = gen_statelis(self.mc.ncas, self.mc.nelecas)
        # self.statelis = np.asarray(statelis, dtype=int)
        self.statelis = siso.read_statelis(mc)
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

        self.save_mag = save_mag
        self.save_Hmat = save_Hmat
        self.save_old_Hal = save_old_Hal
        self.verbose = verbose

    def calc_z(self):
        # 1e SOC integrals
        hso1e = self.mol.intor('int1e_pnucxp',3)

        # All electron SISO
        mo_cas = self.mc.mo_coeff[:,self.mc.ncore:self.mc.ncore+self.mc.ncas]
        sodm1 = self.mc.make_rdm1()

        # 2e SOC J/K1/K2 integrals
        log = logger.Logger(self.mol.stdout, self.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        log.info('SISO with density fitting')
        mol = self.with_df.mol
        auxmol = self.with_df.auxmol
        nao = mol.nao
        with df.addons.load(self.with_df._cderi, self.with_df._dataname) as feri:
            if isinstance(feri, np.ndarray):
                naoaux = feri.shape[0]
            else:
                if isinstance(feri, h5py.Group):
                    naoaux = feri['0'].shape[0]
                else:
                    naoaux = feri.shape[0]
        
        auxe2(mol, auxmol, self.title, int3c='int3c2e_pvxp1', aosym='s2ij', comp=3, verbose=self.verbose)
        def load(aux_slice):
            if self.with_df._cderi is None:
                self.with_df.build()
            
            feri_name = self.title+'_int3c2e_pvxp1.h5'
            b0, b1 = aux_slice
            with df.addons.load(feri_name, 'j3c') as feri:
                j3c_pvxp1 = _load_from_h5g(feri, b0, b1)
            with df.addons.load(self.with_df._cderi, self.with_df._dataname) as feri:
                if isinstance(feri, np.ndarray):
                    j3c =  np.asarray(feri[b0:b1], order='C')
                else:
                    if isinstance(feri, h5py.Group):
                        j3c = _load_from_h5g(feri, b0, b1)
                    else:
                        j3c =  np.asarray(feri[b0:b1])
            return j3c_pvxp1, j3c

        nao_pair = nao*(nao+1)//2
        max_memory = int(mol.max_memory - lib.current_memory()[0])
        blksize = max(16, int(max_memory*.06e6/8/nao_pair**2/3))
        nstep = -(-naoaux//blksize)
        vj = vk = vk2 = 0
        p1 = 0
        for istep, aux_slice in enumerate(lib.prange(0, naoaux, blksize)):
            t1 = (logger.process_clock(), logger.perf_counter())
            t2 = (logger.process_clock(), logger.perf_counter())
            log.debug1('2e SOC J/K1/K2 integrals [%d/%d]', istep+1, nstep)
            j3c_pvxp1, j3c = load(aux_slice)
            p0, p1 = aux_slice
            nrow = p1 - p0
            j3c_pvxp1 = lib.unpack_tril(j3c_pvxp1.reshape(3*nrow,-1),filltriu=2).reshape(3,nrow,nao,nao)
            j3c = lib.unpack_tril(j3c)
            vj += lib.einsum('xPij,Pkl,kl->xij', j3c_pvxp1, j3c, sodm1)
            t2 = log.timer_debug1('contracting vj AO [{}/{}], nrow = {}'.format(p0, p1, nrow), *t2)
            vk += lib.einsum('xPij,Pkl,jk->xil', j3c_pvxp1, j3c, sodm1)
            t2 = log.timer_debug1('contracting vk AO [{}/{}], nrow = {}'.format(p0, p1, nrow), *t2)
            vk2 += lib.einsum('xPij,Pkl,li->xkj', j3c_pvxp1, j3c, sodm1)
            t2 = log.timer_debug1('contracting vk2 AO [{}/{}], nrow = {}'.format(p0, p1, nrow), *t2)
            t1 = log.timer('2e SOC J/K1/K2 integrals [{}/{}]'.format(istep+1, nstep), *t1)
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
