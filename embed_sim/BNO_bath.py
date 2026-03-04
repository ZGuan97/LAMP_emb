from pyscf import df, ao2mo, lib
from pyscf.lib import logger
from pyscf.mp.dfmp2 import _DFINCOREERIS as _RDFINCOREERIS
from pyscf.mp.dfmp2 import _DFOUTCOREERIS as _RDFOUTCOREERIS
from pyscf.mp.dfump2 import _DFINCOREERIS as _UDFINCOREERIS
from pyscf.mp.dfump2 import _DFOUTCOREERIS as _UDFOUTCOREERIS
from functools import reduce
import numpy as np
import ctypes
import os


def _load_mp2_matrix_allow_empty(fname):
    """
    Return:
      - None: file does not exist / contains only whitespace / read failed / invalid shape
      - (n, n) ndarray: valid square matrix
    """
    if (not os.path.isfile(fname)) or (os.path.getsize(fname) == 0):
        return None
    try:
        M = np.loadtxt(fname)
    except Exception:
        return None
    if M.ndim != 2 or (M.shape[0] != M.shape[1]):
        return None
    return M

def choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True):
    """
    Choose eta such that:
      nbath_new = sum(eigvals_core < 2 - eta) + sum(eigvals_vir > eta) == G
    
    Parameters
    ----------
    eigvals_core, eigvals_vir : 1D arrays
    G : int, target number of bath orbitals
    tol : numerical tolerance
    strict : if True, use strict inequalities (consistent with the original code);
             if False, use comparisons with tolerance
    
    Returns
    -------
    eta : float
    """
    eigvals_core = np.asarray(eigvals_core).ravel()
    eigvals_vir  = np.asarray(eigvals_vir ).ravel()

    # Express both conditions in the form “eta < threshold”
    # core: eta < (2 - eigvals_core)
    # vir : eta < (eigvals_vir)
    T = np.concatenate((2.0 - eigvals_core, eigvals_vir))

    # Reachable range of the number of bath orbitals:
    # eta → +inf  => 0
    # eta → -inf  => T.size
    K = T.size
    if not (0 <= G <= K):
        raise ValueError(f"G must be between 0 and {K}, got {G}")

    # Use unique thresholds to partition the real line; 
    # within each interval between adjacent thresholds, the count remains unchanged
    u = np.unique(T)
    u.sort()          # Up Order
    u = u[::-1]       # Down Order

    # Candidate eta: midpoints between each pair of adjacent thresholds
    # (ensures the value does not fall exactly on an equality boundary)
    # shape: (len(u) - 1,)
    if u.size == 1:
        # All thresholds are identical; only K or 0 can be obtained (depending on whether eta < u[0])
        if G == K:
            return u[0] - 10*tol
        if G == 0:
            return u[0] + 10*tol
        raise ValueError("All thresholds equal -> cannot realize intermediate G.")

    mids = 0.5 * (u[:-1] + u[1:])  # nbath remains constant at these points

    # Compute nbath for each midpoint (vectorized, no loops)
    if strict:
        counts = np.sum(T[None, :] > mids[:, None], axis=1)
    else:
        counts = np.sum(T[None, :] > (mids[:, None] + tol), axis=1)

    # Also consider the two ends:
    # eta > max(T) -> 0
    # eta < min(T) -> K
    # We include the two ends in the candidate set as well
    eta_hi = u[0] + 10*tol   # Make eta > max(T)
    eta_lo = u[-1] - 10*tol  # Make eta < min(T)

    # Check whether there exists a midpoint where counts == G
    hit = np.where(counts == G)[0]
    if hit.size > 0:
        return float(mids[hit[0]])

    # Check whether either endpoint satisfies the condition
    if G == 0:
        return float(eta_hi)
    if G == K:
        return float(eta_lo)

    # Otherwise, it means that due to repeated thresholds / discontinuous jumps,
    # G cannot be obtained exactly.
    # Here we return the eta that gives the closest value (this could also be changed to directly raise an error)
    idx = np.argmin(np.abs(counts - G))
    raise ValueError(f"Cannot achieve nbath_new == {G} exactly. Closest is {counts[idx]} at eta={mids[idx]}.")


def get_RMP2_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir, eta=1e-4, verbose=None):
    
    log = logger.new_logger(mf, verbose=verbose)
    log.info('')
    log.info('constructing RMP2 bath')
    
    nocc = (mf.mo_occ>0).sum()
    nvir = (mf.mo_occ==0).sum()
    
    occ_coeff = mf.mo_coeff[:,mf.mo_occ>0]
    vir_coeff = mf.mo_coeff[:,mf.mo_occ==0]
    es_occ_coeff = lib.dot(ao2eo, es_mf.mo_coeff[:,es_mf.mo_occ>0])
    es_vir_coeff = lib.dot(ao2eo, es_mf.mo_coeff[:,es_mf.mo_occ==0])
    
    occ_energy = mf.mo_energy[mf.mo_occ>0]
    vir_energy = mf.mo_energy[mf.mo_occ==0]
    es_occ_energy = es_mf.mo_energy[es_mf.mo_occ>0]
    es_vir_energy = es_mf.mo_energy[es_mf.mo_occ==0]
    
    def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
        log = logger.new_logger(mf, verbose)
    
        with_df = getattr(mf, 'with_df', None)
        assert( with_df is not None )
    
        if with_df._cderi is None:
            log.debug('Caching ovL-type integrals directly')
            if with_df.auxmol is None:
                with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
        else:
            log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
    
        assert (occ_coeff is not None and vir_coeff is not None)
    
        # determine incore or outcore
        nocc = occ_coeff.shape[1]
        nvir = vir_coeff.shape[1]
        naux = with_df.get_naoaux()
    
        if ovL is not None:
            if isinstance(ovL, np.ndarray):
                outcore = False
            elif isinstance(ovL, str):
                outcore = True
            else:
                log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                          type(ovL))
                raise TypeError
        else:
            mem_now = mf.max_memory - lib.current_memory()[0]
            mem_df = nocc*nvir*naux*8/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            
            outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
        log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')
    
        if outcore:
            eris = _RDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                  ovL=ovL, ovL_to_save=ovL_to_save,
                                  verbose=log.verbose, stdout=log.stdout)
        else:
            eris = _RDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                 ovL=ovL,
                                 verbose=log.verbose, stdout=log.stdout)
        eris.build()
    
        return eris
    
    def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
    
        log = logger.new_logger(mf, verbose)
    
        assert (ao2mo is not None)
    
        nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
        assert (occ_energy is not None and vir_energy is not None)
        moevv = np.asarray(vir_energy[:,None] + vir_energy, order='C')
    
        mem_avail = mf.max_memory - lib.current_memory()[0]
    
        if with_t2:
            t2 = np.zeros((nocc,nocc,nvir,nvir), dtype=eris.dtype)
            t2_ptr = t2.ctypes.data_as(ctypes.c_void_p)
            mem_avail -= t2.size * eris.dsize / 1e6
        else:
            t2 = None
            t2_ptr = lib.c_null_ptr()
    
        if mem_avail < 0:
            log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
            raise MemoryError
    
        libmp = lib.load_library('libmp')
        drv = libmp.MP2_contract_d
    
        # determine occ blksize
        if isinstance(eris.ovL, np.ndarray):    # incore ovL
            occ_blksize = nocc
        else:   # outcore ovL
            # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
            occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvir**2)/(2*naux*nvir)))
            occ_blksize = min(nocc, max(1, occ_blksize))
    
        log.debug('occ blksize for %s loop: %d/%d', mf.__class__.__name__, occ_blksize, nocc)
    
        cput1 = (logger.process_clock(), logger.perf_counter())
    
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
            nocci = i1-i0
            iaL = eris.get_occ_blk(i0,i1)
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                noccj = j1-j0
                if ibatch == jbatch:
                    jbL = iaL
                else:
                    jbL = eris.get_occ_blk(j0,j1)
    
                ed = np.zeros(1, dtype=np.float64)
                ex = np.zeros(1, dtype=np.float64)
                moeoo_block = np.asarray(
                    occ_energy[i0:i1,None] + occ_energy[j0:j1], order='C')
                s2symm = 1
                t2_ex = 0
                drv(
                    ed.ctypes.data_as(ctypes.c_void_p),
                    ex.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(s2symm),
                    iaL.ctypes.data_as(ctypes.c_void_p),
                    jbL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(i0), ctypes.c_int(j0),
                    ctypes.c_int(nocci), ctypes.c_int(noccj),
                    ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
                    moeoo_block.ctypes.data_as(ctypes.c_void_p),
                    moevv.ctypes.data_as(ctypes.c_void_p),
                    t2_ptr, ctypes.c_int(t2_ex)
                )
    
                jbL = None
            iaL = None
    
            cput1 = log.timer_debug1('i-block [%d:%d]/%d' % (i0,i1,nocc), *cput1)
    
        return t2
    
    def _gamma1_intermediates(mf, t2=None, eris=None):
        assert (t2 is not None)
        nocc, nocc, nvir, nvir = t2.shape
        dtype = t2.dtype
    
        dm1occ = np.zeros((nocc,nocc), dtype=dtype)
        dm1vir = np.zeros((nvir,nvir), dtype=dtype)
        for i in range(nocc):
            t2i = t2[i]
            l2i = t2i
            dm1vir += lib.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                    - lib.einsum('jca,jbc->ba', l2i, t2i)
            dm1occ += lib.einsum('iab,jab->ij', l2i, t2i) * 2 \
                    - lib.einsum('iab,jba->ij', l2i, t2i)
        
        dm1occ *= -1
        dm1vir += dm1vir.T
        dm1occ += dm1occ.T
        dm1occ[np.diag_indices(nocc)] += 2
        return dm1occ, dm1vir
    
    eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff)
    eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff)
    
    t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov)
    t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV)
    
    D_IJ = _gamma1_intermediates(mf, t_IJab, eris_Ov)[0]
    D_AB = _gamma1_intermediates(mf, t_ijAB, eris_oV)[1]
    
    S = mf.get_ovlp()
    D_IJ_ao = lib.einsum('pi,ij,qj->pq', occ_coeff, D_IJ, occ_coeff)
    D_AB_ao = lib.einsum('pi,ij,qj->pq', vir_coeff, D_AB, vir_coeff)
    D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
    D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))
    
    bins = np.array([10**-x for x in range(0,11)][::-1])
    eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
    histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
    log.info('Occupied BNO histogram')
    log.info('%s',histogram_core)
    log.info('')
    
    eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
    histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
    log.info('Virtual BNO histogram')
    log.info('%s',histogram_vir)
    log.info('')
    
    MP2_bath_core = (eigvals_core < 2 - eta)
    MP2_bath_vir = (eigvals_vir > eta)

    lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
    lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
    lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
    lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
    lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])
    
    nbath_new_core = MP2_bath_core.sum()
    nbath_new_vir = MP2_bath_vir.sum()
    nbath_new = nbath_new_core + nbath_new_vir
    ncore_new = (~MP2_bath_core).sum()
    log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
    # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
    log.info('')
    
    return lo2MP2_bath, lo2MP2_core, lo2MP2_vir

def get_UMP2_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir, eta=1e-2, verbose=None):
    
    log = logger.new_logger(mf, verbose=verbose)
    log.info('')
    log.info('constructing UMP2 bath')
    
    mf = mf.to_uhf()
    es_mf = es_mf.to_uhf()
    
    nocc = [(mf.mo_occ[i]>0).sum() for i in range(2)]
    nvir = [(mf.mo_occ[i]==0).sum() for i in range(2)]
    
    occ_coeff = [mf.mo_coeff[i][:,mf.mo_occ[i]>0] for i in range(2)]
    vir_coeff = [mf.mo_coeff[i][:,mf.mo_occ[i]==0] for i in range(2)]
    es_occ_coeff = [lib.dot(ao2eo, es_mf.mo_coeff[i][:,es_mf.mo_occ[i]>0]) for i in range(2)]
    es_vir_coeff = [lib.dot(ao2eo, es_mf.mo_coeff[i][:,es_mf.mo_occ[i]==0]) for i in range(2)]
    
    occ_energy = [mf.mo_energy[i][mf.mo_occ[i]>0] for i in range(2)]
    vir_energy = [mf.mo_energy[i][mf.mo_occ[i]==0] for i in range(2)]
    es_occ_energy = [es_mf.mo_energy[i][es_mf.mo_occ[i]>0] for i in range(2)]
    es_vir_energy = [es_mf.mo_energy[i][es_mf.mo_occ[i]==0] for i in range(2)]
    
    def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
        log = logger.new_logger(mf, verbose)
    
        with_df = getattr(mf, 'with_df', None)
        assert( with_df is not None )
    
        if with_df._cderi is None:
            log.debug('Caching ovL-type integrals directly')
            if with_df.auxmol is None:
                with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
        else:
            log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
    
        assert (occ_coeff is not None and vir_coeff is not None)
    
        # determine incore or outcore
        nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
        nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
        naux = with_df.get_naoaux()
    
        if ovL is not None:
            if isinstance(ovL, np.ndarray):
                outcore = False
            elif isinstance(ovL, str):
                outcore = True
            else:
                log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                          type(ovL))
                raise TypeError
        else:
            mem_now = mf.max_memory - lib.current_memory()[0]
            mem_df = sum(nocc*nvir)*8/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            
            outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
        log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')
    
        if outcore:
            eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                  ovL=ovL, ovL_to_save=ovL_to_save,
                                  verbose=log.verbose, stdout=log.stdout)
        else:
            eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                 ovL=ovL,
                                 verbose=log.verbose, stdout=log.stdout)
        eris.build()
    
        return eris
    
    def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
    
        log = logger.new_logger(mf, verbose)
    
        assert (ao2mo is not None)
    
        nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
        nvirmax = max(nvir)
        assert (occ_energy is not None and vir_energy is not None)
    
        mem_avail = mf.max_memory - lib.current_memory()[0]
    
        if with_t2:
            t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                  np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                  np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
            t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
            mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
        else:
            t2 = None
            t2_ptr = [lib.c_null_ptr()] * 3
    
        if mem_avail < 0:
            log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
            raise MemoryError
    
        libmp = lib.load_library('libmp')
        drv = libmp.MP2_contract_d
    
        # determine occ blksize
        if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
            occ_blksize = nocc
        else:   # outcore ovL
            # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
            occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
            occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]
    
        log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                  occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])
    
        cput1 = (logger.process_clock(), logger.perf_counter())
    
        for s in [0,1]:
            s_t2 = 0 if s == 0 else 2
            moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
            for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                nocci = i1-i0
                iaL = eris.get_occ_blk(s,i0,i1)
                for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                    noccj = j1-j0
                    if ibatch == jbatch:
                        jbL = iaL
                    else:
                        jbL = eris.get_occ_blk(s,j0,j1)
    
                    ed = np.zeros(1, dtype=np.float64)
                    ex = np.zeros(1, dtype=np.float64)
                    moeoo_block = np.asarray(
                        occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                    s2symm = 1
                    t2_ex = True
                    drv(
                        ed.ctypes.data_as(ctypes.c_void_p),
                        ex.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(s2symm),
                        iaL.ctypes.data_as(ctypes.c_void_p),
                        jbL.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(i0), ctypes.c_int(j0),
                        ctypes.c_int(nocci), ctypes.c_int(noccj),
                        ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                        moeoo_block.ctypes.data_as(ctypes.c_void_p),
                        moevv.ctypes.data_as(ctypes.c_void_p),
                        t2_ptr[s_t2], ctypes.c_int(t2_ex)
                    )
    
                    jbL = None
                iaL = None
    
                cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                         *cput1)
                
        # opposite spin
        sa, sb = 0, 1
        drv = libmp.MP2_OS_contract_d
        moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
            nocci = i1-i0
            iaL = eris.get_occ_blk(sa,i0,i1)
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                noccj = j1-j0
                jbL = eris.get_occ_blk(sb,j0,j1)
    
                ed = np.zeros(1, dtype=np.float64)
                moeoo_block = np.asarray(
                    occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                drv(
                    ed.ctypes.data_as(ctypes.c_void_p),
                    iaL.ctypes.data_as(ctypes.c_void_p),
                    jbL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(i0), ctypes.c_int(j0),
                    ctypes.c_int(nocci), ctypes.c_int(noccj),
                    ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                    ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                    ctypes.c_int(naux),
                    moeoo_block.ctypes.data_as(ctypes.c_void_p),
                    moevv.ctypes.data_as(ctypes.c_void_p),
                    t2_ptr[1]
                )
    
                jbL = None
            iaL = None
    
            cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                     *cput1)
    
        return t2
    
    def _gamma1_intermediates(mf, t2=None, eris=None):
        assert (t2 is not None)
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2[1].shape
        
        dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
        dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
        doob  = lib.einsum('imef,jmef->ij', t2bb, t2bb) *-.5
        doob -= lib.einsum('mief,mjef->ij', t2ab, t2ab)
    
        dvva  = lib.einsum('mnae,mnbe->ba', t2aa, t2aa) * .5
        dvva += lib.einsum('mnae,mnbe->ba', t2ab, t2ab)
        dvvb  = lib.einsum('mnae,mnbe->ba', t2bb, t2bb) * .5
        dvvb += lib.einsum('mnea,mneb->ba', t2ab, t2ab)
        
        dooa += dooa.T
        doob += doob.T
        dvva += dvva.T
        dvvb += dvvb.T
        dooa *= 0.5
        doob *= 0.5
        dvva *= 0.5
        dvvb *= 0.5
        dooa[np.diag_indices(nocca)] += 1
        doob[np.diag_indices(noccb)] += 1
        
        dm1occ = [dooa,doob]
        dm1vir = [dvva,dvvb]
        return dm1occ, dm1vir
    
    eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
    eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)
    
    t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose)
    t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose)
    
    D_IJ = _gamma1_intermediates(mf, t_IJab, eris_Ov)[0]
    D_AB = _gamma1_intermediates(mf, t_ijAB, eris_oV)[1]
    
    S = mf.get_ovlp()
    D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
    D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])
    D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
    D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))
    
    bins = np.array([10**-x for x in range(0,11)][::-1])
    eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
    histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
    log.info('Occupied BNO histogram')
    log.info('%s',histogram_core)
    log.info('')
    
    eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
    histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
    log.info('Virtual BNO histogram')
    log.info('%s',histogram_vir)
    log.info('')
    
    MP2_bath_core = (eigvals_core < 2 - eta)
    MP2_bath_vir = (eigvals_vir > eta)
    lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
    lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
    lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
    lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
    lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])
    
    nbath_new_core = MP2_bath_core.sum()
    nbath_new_vir = MP2_bath_vir.sum()
    nbath_new = nbath_new_core + nbath_new_vir
    ncore_new = (~MP2_bath_core).sum()
    log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
    # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
    log.info('')
    
    return lo2MP2_bath, lo2MP2_core, lo2MP2_vir

def get_ROMP2_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir, ao = False, readmp2 = False, eta=1e-2, verbose=None):
    log = logger.new_logger(mf, verbose=verbose)
    if readmp2:
        log.info('========== Read from MP2.TXT files ===========')
        if eta < 1:
            log.info('')
            log.info('========== eta < 1 ===========')
            log.info('constructing ROMP2 bath')

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            S = mf.get_ovlp()
            D_MP2_core = _load_mp2_matrix_allow_empty('D_MP2_core.txt')
            D_MP2_vir  = _load_mp2_matrix_allow_empty('D_MP2_vir.txt')

            if D_MP2_core is None:
                log.info('D_MP2_core.txt is empty/unavailable -> no bath added from core.')
                eigvals_core = np.array([], dtype=float)
                eigvecs_core = None
                MP2_bath_core = np.zeros(lo2core.shape[1], dtype=bool)  # 全 False
            else:
                eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
                if ao:
                    log.info('Bath expansion with AO-DMET')
                    MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
                else:
                    MP2_bath_core = (eigvals_core < 2 - eta)

            bins = np.array([10**-x for x in range(0,11)][::-1])
            histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
            log.info('Occupied BNO histogram')
            log.info('%s',histogram_core)
            log.info('')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            MP2_bath_vir = (eigvals_vir > eta)

            if MP2_bath_core.size == 0:
                lo2MP2_bath_core = lo2core[:, :0]
                lo2MP2_core = lo2core
            else:
                lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
                lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])

            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum() if MP2_bath_core.size else 0
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum() if MP2_bath_core.size else lo2core.shape[1]
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')

        else:
            G = eta
            log.info('')
            log.info('========== eta >= 1 ===========')
            log.info('constructing ROMP2 bath')

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            S = mf.get_ovlp()
            D_MP2_core = _load_mp2_matrix_allow_empty('D_MP2_core.txt')
            D_MP2_vir  = _load_mp2_matrix_allow_empty('D_MP2_vir.txt')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)

            bins = np.array([10**-x for x in range(0,11)][::-1])
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            if D_MP2_core is None:
                log.info('D_MP2_core.txt is empty/unavailable -> no bath added from core.')
                eigvals_core = np.array([], dtype=float)
                eigvecs_core = None
                MP2_bath_core = np.zeros(lo2core.shape[1], dtype=bool)  # 全 False
                eta = choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True)
                log.info("================This is new eta================")
                log.info(str(eta))
            else:
                eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
                eta = choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True)
                log.info("================This is new eta================")
                log.info(str(eta))
                if ao:
                    log.info('Bath expansion with AO-DMET')
                    MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
                else:
                    MP2_bath_core = (eigvals_core < 2 - eta)
            
            if D_MP2_core is not None:
                bins = np.array([10**-x for x in range(0,11)][::-1])
                histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
                log.info('Occupied BNO histogram')
                log.info('%s', histogram_core)
                log.info('')
            else:
                log.info('Occupied BNO histogram skipped (empty core MP2 matrix).')

            MP2_bath_vir = (eigvals_vir > eta)

            if MP2_bath_core.size == 0:
                lo2MP2_bath_core = lo2core[:, :0]
                lo2MP2_core = lo2core
            else:
                lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
                lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])

            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum() if MP2_bath_core.size else 0
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum() if MP2_bath_core.size else lo2core.shape[1]
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')

    else:
        log.info('========== NOT Read from MP2.TXT files ===========')
        if eta < 1:
            log.info('')
            log.info('========== eta < 1 ===========')
            log.info('constructing ROMP2 bath')

            def semi_canonicalize(mf):
                fock = mf.get_fock()
                focka,fockb = fock.focka,fock.fockb
                mo = mf.mo_coeff
                coreidx = mf.mo_occ == 2
                viridx = mf.mo_occ == 0
                openidx = ~(coreidx|viridx)
                mo_focka = reduce(lib.dot, (mo.T, focka, mo))
                mo_fockb = reduce(lib.dot, (mo.T, fockb, mo))
                ea_occ,coeff_occa = np.linalg.eigh(mo_focka[coreidx|openidx,:][:,coreidx|openidx])
                ea_vir,coeff_vira = np.linalg.eigh(mo_focka[viridx,:][:,viridx])
                eb_occ,coeff_occb = np.linalg.eigh(mo_fockb[coreidx,:][:,coreidx])
                eb_vir,coeff_virb = np.linalg.eigh(mo_fockb[openidx|viridx,:][:,openidx|viridx])
                mo_coeff_occa = mo[:,coreidx|openidx]@coeff_occa
                mo_coeff_vira = mo[:,viridx]@coeff_vira
                mo_coeff_occb = mo[:,coreidx]@coeff_occb
                mo_coeff_virb = mo[:,openidx|viridx]@coeff_virb
                mo_a = np.hstack((mo_coeff_occa,mo_coeff_vira))
                mo_b = np.hstack((mo_coeff_occb,mo_coeff_virb))
                ea = np.concatenate((ea_occ,ea_vir))
                eb = np.concatenate((eb_occ,eb_vir))
                return (mo_a,mo_b), (ea,eb), (focka,fockb)

            semi_mo_coeff, semi_mo_energy, fockab = semi_canonicalize(mf)
            es_semi_mo_coeff, es_semi_mo_energy, es_fockab = semi_canonicalize(es_mf)

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            occ_coeff = [semi_mo_coeff[i][:,mo_occ[i]>0] for i in range(2)]
            vir_coeff = [semi_mo_coeff[i][:,mo_occ[i]==0] for i in range(2)]
            es_occ_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]>0]) for i in range(2)]
            es_vir_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]==0]) for i in range(2)]

            occ_energy = [semi_mo_energy[i][mo_occ[i]>0] for i in range(2)]
            vir_energy = [semi_mo_energy[i][mo_occ[i]==0] for i in range(2)]
            es_occ_energy = [es_semi_mo_energy[i][es_mo_occ[i]>0] for i in range(2)]
            es_vir_energy = [es_semi_mo_energy[i][es_mo_occ[i]==0] for i in range(2)]

            def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
                log = logger.new_logger(mf, verbose)

                with_df = getattr(mf, 'with_df', None)
                assert( with_df is not None )

                if with_df._cderi is None:
                    log.debug('Caching ovL-type integrals directly')
                    if with_df.auxmol is None:
                        with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
                else:
                    log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')

                assert (occ_coeff is not None and vir_coeff is not None)

                # determine incore or outcore
                nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
                nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
                naux = with_df.get_naoaux()

                if ovL is not None:
                    if isinstance(ovL, np.ndarray):
                        outcore = False
                    elif isinstance(ovL, str):
                        outcore = True
                    else:
                        log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                                  type(ovL))
                        raise TypeError
                else:
                    mem_now = mf.max_memory - lib.current_memory()[0]
                    mem_df = sum(nocc*nvir)*8/1024**2.
                    log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)

                    outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
                log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')

                if outcore:
                    eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                          ovL=ovL, ovL_to_save=ovL_to_save,
                                          verbose=log.verbose, stdout=log.stdout)
                else:
                    eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                         ovL=ovL,
                                         verbose=log.verbose, stdout=log.stdout)
                eris.build()

                return eris

            #t1 is f f = (f^\alpha, f^\beta) 
            def get_t1(mf, fockab, occ_coeff=None, vir_coeff=None, occ_energy=None, vir_energy=None):
                focka, fockb = fockab
                #g is numerator of f
                gia = reduce(lib.dot, [occ_coeff[0].T, focka, vir_coeff[0]])
                gib = reduce(lib.dot, [occ_coeff[1].T, fockb, vir_coeff[1]])
                t1a = gia/lib.direct_sum('i-a->ia',occ_energy[0],vir_energy[0])
                t1b = gib/lib.direct_sum('i-a->ia',occ_energy[1],vir_energy[1])
                t1 = (t1a, t1b)
                return t1
            #t2 is t
            def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
            
                log = logger.new_logger(mf, verbose)

                assert (ao2mo is not None)

                nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
                nvirmax = max(nvir)
                assert (occ_energy is not None and vir_energy is not None)

                mem_avail = mf.max_memory - lib.current_memory()[0]

                if with_t2:
                    t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                          np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                          np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
                    t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
                    mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
                else:
                    t2 = None
                    t2_ptr = [lib.c_null_ptr()] * 3

                if mem_avail < 0:
                    log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
                    raise MemoryError

                libmp = lib.load_library('libmp')
                drv = libmp.MP2_contract_d

                # determine occ blksize
                if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
                    occ_blksize = nocc
                else:   # outcore ovL
                    # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
                    occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
                    occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]

                log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                          occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])

                cput1 = (logger.process_clock(), logger.perf_counter())

                for s in [0,1]:
                    s_t2 = 0 if s == 0 else 2
                    moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
                    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                        nocci = i1-i0
                        iaL = eris.get_occ_blk(s,i0,i1)
                        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                            noccj = j1-j0
                            if ibatch == jbatch:
                                jbL = iaL
                            else:
                                jbL = eris.get_occ_blk(s,j0,j1)

                            ed = np.zeros(1, dtype=np.float64)
                            ex = np.zeros(1, dtype=np.float64)
                            moeoo_block = np.asarray(
                                occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                            s2symm = 1
                            t2_ex = True
                            drv(
                                ed.ctypes.data_as(ctypes.c_void_p),
                                ex.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(s2symm),
                                iaL.ctypes.data_as(ctypes.c_void_p),
                                jbL.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(i0), ctypes.c_int(j0),
                                ctypes.c_int(nocci), ctypes.c_int(noccj),
                                ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                                moeoo_block.ctypes.data_as(ctypes.c_void_p),
                                moevv.ctypes.data_as(ctypes.c_void_p),
                                t2_ptr[s_t2], ctypes.c_int(t2_ex)
                            )

                            jbL = None
                        iaL = None

                        cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                                 *cput1)

                # opposite spin
                sa, sb = 0, 1
                drv = libmp.MP2_OS_contract_d
                moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
                    nocci = i1-i0
                    iaL = eris.get_occ_blk(sa,i0,i1)
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                        noccj = j1-j0
                        jbL = eris.get_occ_blk(sb,j0,j1)

                        ed = np.zeros(1, dtype=np.float64)
                        moeoo_block = np.asarray(
                            occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                        drv(
                            ed.ctypes.data_as(ctypes.c_void_p),
                            iaL.ctypes.data_as(ctypes.c_void_p),
                            jbL.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(i0), ctypes.c_int(j0),
                            ctypes.c_int(nocci), ctypes.c_int(noccj),
                            ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                            ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                            ctypes.c_int(naux),
                            moeoo_block.ctypes.data_as(ctypes.c_void_p),
                            moevv.ctypes.data_as(ctypes.c_void_p),
                            t2_ptr[1]
                        )

                        jbL = None
                    iaL = None

                    cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                             *cput1)

                return t2

            def _gamma1_intermediates(mf, t1=None, t2=None, eris=None):
                assert (t1 is not None and t2 is not None)
                t1a, t1b = t1
                fa = np.conj(t1a).T
                fb = np.conj(t1b).T
                t2aa, t2ab, t2bb = t2
                log.info(f"The size of t2aa is  = {t2aa.shape}")
                log.info(f"The size of t2ab is  = {t2ab.shape}")
                log.info(f"The size of t2bb is  = {t2bb.shape}")

                t2ba = t2ab.transpose(1,0,3,2)
                nocca, noccb, nvira, nvirb = t2[1].shape

                #dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
                #dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
                #dooa -= lib.einsum('ie,je->ij',t1a,t1a)
                term1 = lib.einsum('kjab,kiab->ij', np.conj(t2aa), t2aa)
                term2 = lib.einsum('kjab,ikab->ij', np.conj(t2aa), t2aa)
                dooa = - (term1 - term2)
                dooa -= lib.einsum('kjab,kiab->ij',np.conj(t2ba),t2ba)
                dooa -= lib.einsum('aj,ai->ij',np.conj(fa),fa)

                term1 = lib.einsum('kjab,kiab->ij', np.conj(t2bb), t2bb)
                term2 = lib.einsum('kjab,ikab->ij', np.conj(t2bb), t2bb)
                doob = - (term1 - term2)
                doob -= lib.einsum('jkab,ikab->ij', np.conj(t2ba), t2ba)
                doob -= lib.einsum('aj,ai->ij', np.conj(fb), fb)

                term1 = lib.einsum('klcb,kldb->cd', np.conj(t2aa), t2aa)
                term2 = lib.einsum('klcb,lkdb->cd', np.conj(t2aa), t2aa)
                dvva = (term1 - term2)
                dvva += lib.einsum('klbc,klbd->cd',np.conj(t2ba),t2ba)
                dvva += lib.einsum('ck,dk->cd', np.conj(fa), fa)

                term1 = lib.einsum('klcb,kldb->cd', np.conj(t2bb), t2bb)
                term2 = lib.einsum('klcb,lkdb->cd', np.conj(t2bb), t2bb)
                dvvb = (term1 - term2)
                dvvb += lib.einsum('klcb,kldb->cd', np.conj(t2ba), t2ba)
                dvvb += lib.einsum('ck,dk->cd',np.conj(fb), fb)

                dooa[np.diag_indices(nocca)] += 1
                doob[np.diag_indices(noccb)] += 1

                dm1occ = [dooa,doob]
                dm1vir = [dvva,dvvb]
                return dm1occ, dm1vir

            eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
            eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)

            t_Ia = get_t1(mf, fockab, occ_coeff, es_vir_coeff, occ_energy, es_vir_energy)
            t_iA = get_t1(mf, fockab, es_occ_coeff, vir_coeff, es_occ_energy, vir_energy)
            t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose)
            t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose)
            
            #D is to be projected
            D_IJ = _gamma1_intermediates(mf, t_Ia, t_IJab, eris_Ov)[0]
            D_AB = _gamma1_intermediates(mf, t_iA, t_ijAB, eris_oV)[1]

            S = mf.get_ovlp()
            #I should check this
            D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
            D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])
            
            D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
            D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))

            np.savetxt(
                'D_MP2_core.txt',
                D_MP2_core,
                fmt='%.16e'
                        )

            np.savetxt(
                'D_MP2_vir.txt',
                D_MP2_vir,
                fmt='%.16e'
            )

            bins = np.array([10**-x for x in range(0,11)][::-1])
            eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
            histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
            log.info('Occupied BNO histogram')
            log.info('%s',histogram_core)
            log.info('')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            if ao:
                log.info('Bath expansion with AO-DMET')
                MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
            else:
                MP2_bath_core = (eigvals_core < 2 - eta)

            MP2_bath_vir = (eigvals_vir > eta)
            lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum()
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum()
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')

        else:
            G = eta
            log.info('')
            log.info('========== eta >= 1 ===========')
            log.info('constructing ROMP2 bath')

            def semi_canonicalize(mf):
                fock = mf.get_fock()
                focka,fockb = fock.focka,fock.fockb
                mo = mf.mo_coeff
                coreidx = mf.mo_occ == 2
                viridx = mf.mo_occ == 0
                openidx = ~(coreidx|viridx)
                mo_focka = reduce(lib.dot, (mo.T, focka, mo))
                mo_fockb = reduce(lib.dot, (mo.T, fockb, mo))
                ea_occ,coeff_occa = np.linalg.eigh(mo_focka[coreidx|openidx,:][:,coreidx|openidx])
                ea_vir,coeff_vira = np.linalg.eigh(mo_focka[viridx,:][:,viridx])
                eb_occ,coeff_occb = np.linalg.eigh(mo_fockb[coreidx,:][:,coreidx])
                eb_vir,coeff_virb = np.linalg.eigh(mo_fockb[openidx|viridx,:][:,openidx|viridx])
                mo_coeff_occa = mo[:,coreidx|openidx]@coeff_occa
                mo_coeff_vira = mo[:,viridx]@coeff_vira
                mo_coeff_occb = mo[:,coreidx]@coeff_occb
                mo_coeff_virb = mo[:,openidx|viridx]@coeff_virb
                mo_a = np.hstack((mo_coeff_occa,mo_coeff_vira))
                mo_b = np.hstack((mo_coeff_occb,mo_coeff_virb))
                ea = np.concatenate((ea_occ,ea_vir))
                eb = np.concatenate((eb_occ,eb_vir))
                return (mo_a,mo_b), (ea,eb), (focka,fockb)

            semi_mo_coeff, semi_mo_energy, fockab = semi_canonicalize(mf)
            es_semi_mo_coeff, es_semi_mo_energy, es_fockab = semi_canonicalize(es_mf)

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            occ_coeff = [semi_mo_coeff[i][:,mo_occ[i]>0] for i in range(2)]
            vir_coeff = [semi_mo_coeff[i][:,mo_occ[i]==0] for i in range(2)]
            es_occ_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]>0]) for i in range(2)]
            es_vir_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]==0]) for i in range(2)]

            occ_energy = [semi_mo_energy[i][mo_occ[i]>0] for i in range(2)]
            vir_energy = [semi_mo_energy[i][mo_occ[i]==0] for i in range(2)]
            es_occ_energy = [es_semi_mo_energy[i][es_mo_occ[i]>0] for i in range(2)]
            es_vir_energy = [es_semi_mo_energy[i][es_mo_occ[i]==0] for i in range(2)]

            def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
                log = logger.new_logger(mf, verbose)

                with_df = getattr(mf, 'with_df', None)
                assert( with_df is not None )

                if with_df._cderi is None:
                    log.debug('Caching ovL-type integrals directly')
                    if with_df.auxmol is None:
                        with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
                else:
                    log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')

                assert (occ_coeff is not None and vir_coeff is not None)

                # determine incore or outcore
                nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
                nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
                naux = with_df.get_naoaux()

                if ovL is not None:
                    if isinstance(ovL, np.ndarray):
                        outcore = False
                    elif isinstance(ovL, str):
                        outcore = True
                    else:
                        log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                                  type(ovL))
                        raise TypeError
                else:
                    mem_now = mf.max_memory - lib.current_memory()[0]
                    mem_df = sum(nocc*nvir)*8/1024**2.
                    log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)

                    outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
                log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')

                if outcore:
                    eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                          ovL=ovL, ovL_to_save=ovL_to_save,
                                          verbose=log.verbose, stdout=log.stdout)
                else:
                    eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                         ovL=ovL,
                                         verbose=log.verbose, stdout=log.stdout)
                eris.build()

                return eris

            def get_t1(mf, fockab, occ_coeff=None, vir_coeff=None, occ_energy=None, vir_energy=None):
                focka, fockb = fockab
                gia = reduce(lib.dot, [occ_coeff[0].T, focka, vir_coeff[0]])
                gib = reduce(lib.dot, [occ_coeff[1].T, fockb, vir_coeff[1]])
                t1a = gia/lib.direct_sum('i-a->ia',occ_energy[0],vir_energy[0])
                t1b = gib/lib.direct_sum('i-a->ia',occ_energy[1],vir_energy[1])
                t1 = (t1a, t1b)
                return t1

            def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
            
                log = logger.new_logger(mf, verbose)

                assert (ao2mo is not None)

                nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
                nvirmax = max(nvir)
                assert (occ_energy is not None and vir_energy is not None)

                mem_avail = mf.max_memory - lib.current_memory()[0]

                if with_t2:
                    t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                          np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                          np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
                    t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
                    mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
                else:
                    t2 = None
                    t2_ptr = [lib.c_null_ptr()] * 3

                if mem_avail < 0:
                    log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
                    raise MemoryError

                libmp = lib.load_library('libmp')
                drv = libmp.MP2_contract_d

                # determine occ blksize
                if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
                    occ_blksize = nocc
                else:   # outcore ovL
                    # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
                    occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
                    occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]

                log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                          occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])

                cput1 = (logger.process_clock(), logger.perf_counter())

                for s in [0,1]:
                    s_t2 = 0 if s == 0 else 2
                    moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
                    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                        nocci = i1-i0
                        iaL = eris.get_occ_blk(s,i0,i1)
                        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                            noccj = j1-j0
                            if ibatch == jbatch:
                                jbL = iaL
                            else:
                                jbL = eris.get_occ_blk(s,j0,j1)

                            ed = np.zeros(1, dtype=np.float64)
                            ex = np.zeros(1, dtype=np.float64)
                            moeoo_block = np.asarray(
                                occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                            s2symm = 1
                            t2_ex = True
                            drv(
                                ed.ctypes.data_as(ctypes.c_void_p),
                                ex.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(s2symm),
                                iaL.ctypes.data_as(ctypes.c_void_p),
                                jbL.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(i0), ctypes.c_int(j0),
                                ctypes.c_int(nocci), ctypes.c_int(noccj),
                                ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                                moeoo_block.ctypes.data_as(ctypes.c_void_p),
                                moevv.ctypes.data_as(ctypes.c_void_p),
                                t2_ptr[s_t2], ctypes.c_int(t2_ex)
                            )

                            jbL = None
                        iaL = None

                        cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                                 *cput1)

                # opposite spin
                sa, sb = 0, 1
                drv = libmp.MP2_OS_contract_d
                moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
                    nocci = i1-i0
                    iaL = eris.get_occ_blk(sa,i0,i1)
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                        noccj = j1-j0
                        jbL = eris.get_occ_blk(sb,j0,j1)

                        ed = np.zeros(1, dtype=np.float64)
                        moeoo_block = np.asarray(
                            occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                        drv(
                            ed.ctypes.data_as(ctypes.c_void_p),
                            iaL.ctypes.data_as(ctypes.c_void_p),
                            jbL.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(i0), ctypes.c_int(j0),
                            ctypes.c_int(nocci), ctypes.c_int(noccj),
                            ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                            ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                            ctypes.c_int(naux),
                            moeoo_block.ctypes.data_as(ctypes.c_void_p),
                            moevv.ctypes.data_as(ctypes.c_void_p),
                            t2_ptr[1]
                        )

                        jbL = None
                    iaL = None

                    cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                             *cput1)

                return t2

            def _gamma1_intermediates(mf, t1=None, t2=None, eris=None):
                assert (t1 is not None and t2 is not None)
                t1a, t1b = t1
                fa = np.conj(t1a).T
                fb = np.conj(t1b).T
                t2aa, t2ab, t2bb = t2
                log.info(f"The size of t2aa is  = {t2aa.shape}")
                log.info(f"The size of t2ab is  = {t2ab.shape}")
                log.info(f"The size of t2bb is  = {t2bb.shape}")

                t2ba = t2ab.transpose(1,0,3,2)
                nocca, noccb, nvira, nvirb = t2[1].shape

                #dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
                #dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
                #dooa -= lib.einsum('ie,je->ij',t1a,t1a)
                term1 = lib.einsum('kjab,kiab->ij', np.conj(t2aa), t2aa)
                term2 = lib.einsum('kjab,ikab->ij', np.conj(t2aa), t2aa)
                dooa = - (term1 - term2)
                dooa -= lib.einsum('kjab,kiab->ij',np.conj(t2ba),t2ba)
                dooa -= lib.einsum('aj,ai->ij',np.conj(fa),fa)

                term1 = lib.einsum('kjab,kiab->ij', np.conj(t2bb), t2bb)
                term2 = lib.einsum('kjab,ikab->ij', np.conj(t2bb), t2bb)
                doob = - (term1 - term2)
                doob -= lib.einsum('jkab,ikab->ij', np.conj(t2ba), t2ba)
                doob -= lib.einsum('aj,ai->ij', np.conj(fb), fb)

                term1 = lib.einsum('klcb,kldb->cd', np.conj(t2aa), t2aa)
                term2 = lib.einsum('klcb,lkdb->cd', np.conj(t2aa), t2aa)
                dvva = (term1 - term2)
                dvva += lib.einsum('klbc,klbd->cd',np.conj(t2ba),t2ba)
                dvva += lib.einsum('ck,dk->cd', np.conj(fa), fa)

                term1 = lib.einsum('klcb,kldb->cd', np.conj(t2bb), t2bb)
                term2 = lib.einsum('klcb,lkdb->cd', np.conj(t2bb), t2bb)
                dvvb = (term1 - term2)
                dvvb += lib.einsum('klcb,kldb->cd', np.conj(t2ba), t2ba)
                dvvb += lib.einsum('ck,dk->cd',np.conj(fb), fb)

                dooa[np.diag_indices(nocca)] += 1
                doob[np.diag_indices(noccb)] += 1

                dm1occ = [dooa,doob]
                dm1vir = [dvva,dvvb]
                return dm1occ, dm1vir

            eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
            eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)

            t_Ia = get_t1(mf, fockab, occ_coeff, es_vir_coeff, occ_energy, es_vir_energy)
            t_iA = get_t1(mf, fockab, es_occ_coeff, vir_coeff, es_occ_energy, vir_energy)
            t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose)
            t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose)

            D_IJ = _gamma1_intermediates(mf, t_Ia, t_IJab, eris_Ov)[0]
            D_AB = _gamma1_intermediates(mf, t_iA, t_ijAB, eris_oV)[1]

            S = mf.get_ovlp()
            D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
            D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])
            D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
            D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))

            np.savetxt(
                'D_MP2_core.txt',
                D_MP2_core,
                fmt='%.16e'
                        )

            np.savetxt(
                'D_MP2_vir.txt',
                D_MP2_vir,
                fmt='%.16e'
            )

            bins = np.array([10**-x for x in range(0,11)][::-1])
            eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
            histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
            log.info('Occupied BNO histogram')
            log.info('%s',histogram_core)
            log.info('')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            eta = choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True)
            log.info("================This is new eta================")
            log.info(str(eta))

            if ao:
                log.info('Bath expansion with AO-DMET')
                MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
            else:
                MP2_bath_core = (eigvals_core < 2 - eta)

            MP2_bath_vir = (eigvals_vir > eta)
            lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum()
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum()
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')
            
    return lo2MP2_bath, lo2MP2_core, lo2MP2_vir

def make_histogram(values, bins, labels=True, binwidth=6, height=10, fill=":", show_number=False, invertx=True, rstrip=True):
    '''
    Modified from https://github.com/BoothGroup/Vayesta/blob/master/vayesta/core/bath/helper.py
    Original author: Max Nusspickel & Charles J. C. Scott
    '''
    hist = np.histogram(values, bins)[0]
    if invertx:
        bins, hist = bins[::-1], hist[::-1]
    hmax = hist.max()
    
    binwidths = [len(str(hval))-2 for hval in hist]

    width = binwidth * len(hist) + sum(binwidths)
    plot = np.zeros((height + show_number, width), dtype=str)
    plot[:] = " "
    if hmax > 0:
        for i, hval in enumerate(hist):
            colstart = i * binwidth + sum(binwidths[:i])
            colend = (i + 1) * binwidth + sum(binwidths[:(i+1)])
            barheight = int(np.rint(height * hval / hmax))
            if barheight == 0:
                continue
            # Top
            plot[-barheight, colstart + 1 : colend - 1] = "_"
            if show_number:
                number = " {:^{w}s}".format("%d" % hval, w=binwidth - 1 + binwidths[i])
                for idx, i in enumerate(range(colstart, colend)):
                    plot[-barheight - 1, i] = number[idx]

            if barheight == 1:
                continue
            # Fill
            if fill:
                plot[-barheight + 1 :, colstart + 1 : colend] = fill
            # Left/right border
            plot[-barheight + 1 :, colstart] = "|"
            plot[-barheight + 1 :, colend - 1] = "|"

    lines = ["".join(plot[r, :].tolist()) for r in range(height)]
    # Baseline
    lines.append("+" + ((width - 2) * "-") + "+")
    
    labelwides = np.hstack([6+np.array(binwidths)[1:],np.array([6])])
    if labels:
        lines += ["{:<{w}}".format("E-0", w=4) + "".join(["{:<{w}}".format("E-%d" % d, w=labelwides[i]) for i,d in enumerate(range(1, 11))])]

    if rstrip:
        lines = [line.rstrip() for line in lines]
    txt = "\n".join(lines)
    return txt
