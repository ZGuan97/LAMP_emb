# Regularized DIIS technique
# Reg terms could be bare polarization, dS, ...

import numpy as np 
from pyscf.lib import logger
from pyscf import scf, lib, gto
import scipy.linalg as linalg

from functools import reduce

from embed_sim import ssdmet

class RDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None, rdiis_prop='dS', imp_idx=None,power=.2,kernel=None,mute=False):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.space = 8
        self.rdiis_prop = rdiis_prop 
        self.imp_idx = imp_idx
        self.kernel = kernel
        self.power = power
        self.mute = mute
        self.ent = 1.0
        self.ent_conv_tol = 0.1

    def get_err_vec1(self, s, d, f):
        '''error vector = SDF - FDS + R'''
        sinv = linalg.inv(s)
        occ, orb = linalg.eigh(d, sinv)
        split_occ = ssdmet.split_occ(occ)
        dm = np.zeros((2, *np.shape(d)))
        dm[0] = reduce(np.dot, (sinv,orb,np.diag(split_occ[0]),orb.conj().T,sinv))
        dm[1] = reduce(np.dot, (sinv,orb,np.diag(split_occ[1]),orb.conj().T,sinv))
        if isinstance(f, np.ndarray) and f.ndim == 2:
            sdf = reduce(np.dot, (s,d,f))
            errvec = sdf.T.conj() - sdf

            kernel = self.kernel
            power = self.power
            # if self.reg == 'dS':

            caolo, cloao = ssdmet.lowdin_orth(None, ovlp=s)
            ldm = np.zeros(np.shape(dm))
            ldm[0] = reduce(np.dot,(cloao,dm[0],cloao.conj().T))
            ldm[1] = reduce(np.dot,(cloao,dm[1],cloao.conj().T))
            ent = ssdmet.get_rdiis_property(ldm, self.imp_idx, self.rdiis_prop)
            self.ent = ent

            if not self.mute:
                logger.info(self, '----------RDIIS-Entropy %.3f', ent)
            if np.abs(ent) > self.ent_conv_tol:
                if kernel is None:
                    errvec = errvec+np.eye(errvec.shape[0])*power*np.abs(ent)
                else:
                    errvec = errvec+kernel*power*np.abs(ent)

            # elif self.reg == 'P':
            #     # TODO
            #     pol = ssdmet.get_dmet_env_pol(vmf,self.imp_idx)
            #     # print('kernel', kernel)
            #     if kernel is None:
            #         # kernel = np.eye(errvec.shape[0])
            #         pass
            #     tagged = (np.linalg.norm(errvec) < np.linalg.norm(kernel*power*pol)) and (pol >= 0.1*mf.mol.spin) 
            #     logger.info(self, '----------RDDIS-pol %.3f %s',np.linalg.norm(pol),tagged)
            #     if tagged:
            #         errvec = errvec + kernel*power*pol

        # elif isinstance(f, np.ndarray) and f.ndim == 3 and s.ndim == 3:
        #     errvec = []
        #     for i in range(f.shape[0]):
        #         sdf = reduce(np.dot, (s[i], d[i], f[i]))
        #         errvec.append((sdf.T.conj() - sdf))
        #     errvec = np.vstack(errvec)

        # elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        #     nao = s.shape[-1]
        #     s = lib.asarray((s,s)).reshape(-1,nao,nao)
        #     return get_err_vec1(s, d.reshape(s.shape), f.reshape(s.shape))
        else:
            raise RuntimeError('Unknown SCF DIIS type')
        return errvec

    def update(self, s, d, f, *args, **kwargs):
        errvec = self.get_err_vec1(s, d, f)
        logger.debug1(self, 'diis-norm(errvec)=%g', np.linalg.norm(errvec))
        xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

def rdiis_check_convergence(envs):
    mf = envs['mf']
    e_tot = envs['e_tot']
    last_hf_e = envs['last_hf_e']
    norm_gorb = envs['norm_gorb']
    conv_tol = envs['conv_tol']
    conv_tol_grad = envs['conv_tol_grad']
    assert isinstance(mf.diis, RDIIS)
    if mf.diis.ent < mf.diis.ent_conv_tol and abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
        print('Converged by rdiis_check_convergence with entropy', mf.diis.ent)
        return True
    else:
        return False
