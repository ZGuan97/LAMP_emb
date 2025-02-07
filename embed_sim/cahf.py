import numpy as np
from embed_sim import spin_utils
from pyscf import scf
from pyscf.scf import hf

def get_coeffs(ncas, nelecas, spin):
    na, nb = spin_utils.unpack_nelec(nelecas, spin)
    lambda_J = (ncas*nelecas*(nelecas-1)-2*na*nb) / (nelecas**2*(ncas-1))
    lambda_K = (ncas*nelecas*(nelecas-1)-2*ncas*na*nb) / (nelecas**2*(ncas-1))
    coulomb_a = lambda_J
    exchange_b = lambda_K*2
    f = nelecas / ncas / 2
    alpha = (1-coulomb_a)/(1-f)
    beta = (1-exchange_b)/(1-f)
    return f, coulomb_a, exchange_b, alpha, beta

def CAHF_get_veff(f, a, b):
    def _get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array((dm*.5, dm*.5))

        if self._eri is not None or not self.direct_scf:
            if hasattr(dm, 'mo_occ') and np.ndim(dm.mo_occ) == 1:
                mo_occa = (dm.mo_occ > 0).astype(np.double)
                mo_occb = (dm.mo_occ ==2).astype(np.double)
                dm = lib.tag_array(dm, mo_coeff=(dm.mo_coeff,)*2,
                                   mo_occ=(mo_occa,mo_occb))
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf[0] = ((2*f+2*a*f*(f-1))*vj[0] + (2-2*f-2*a*f*(f-1))*vj[1]
                +(-f-b*(f-1)*f)*vk[0]+(-1+f+b*(f-1)*f)*vk[1])
            vhf[1] = ((2*f+2*a*f**2)*vj[0] + (2-2*f-2*a*f**2)*vj[1]
                +(-f-b*f**2)*vk[0]+(-1+f+b*f**2)*vk[1])
        else:
            ddm = dm - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf[0] = ((2*f+2*a*f*(f-1))*vj[0] + (2-2*f-2*a*f*(f-1))*vj[1]
                +(-f-b*(f-1)*f)*vk[0]+(-1+f+b*(f-1)*f)*vk[1])
            vhf[1] = ((2*f+2*a*f**2)*vj[0] + (2-2*f-2*a*f**2)*vj[1]
                +(-f-b*f**2)*vk[0]+(-1+f+b*f**2)*vk[1])
            vhf += np.asarray(vhf_last)
        return vhf
    return _get_veff

from functools import reduce
from pyscf import lib
def CAHF_get_roothaan_fock(f):
    def _get_roothaan_fock(focka_fockb, dma_dmb, s):
        nao = s.shape[0]
        focka, fockb = focka_fockb
        dma, dmb = dma_dmb
        fc = f*focka + (1-f)*fockb
        # fc = (focka + fockb) * .5
    # Projector for core, open-shell, and virtual
        pc = np.dot(dmb, s)
        po = np.dot(dma-dmb, s)
        pv = np.eye(nao) - np.dot(dma, s)
        fock  = reduce(np.dot, (pc.conj().T, fc, pc)) * .5
        fock += reduce(np.dot, (po.conj().T, fc, po)) * .5
        fock += reduce(np.dot, (pv.conj().T, fc, pv)) * .5
        fock += reduce(np.dot, (po.conj().T, fockb, pc))
        fock += reduce(np.dot, (po.conj().T, focka, pv))
        fock += reduce(np.dot, (pv.conj().T, fc, pc))
        fock += reduce(np.dot, (pv.conj().T, fc, pc))
        fock = fock + fock.conj().T
        fock = lib.tag_array(fock, focka=focka, fockb=fockb)
        return fock
    return _get_roothaan_fock

def CAHF_get_fock(frac):
    def _get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                fock_last=None):
        '''Build fock matrix based on Roothaan's effective fock.
        See also :func:`get_roothaan_fock`
        '''
        if h1e is None: h1e = mf.get_hcore()
        if s1e is None: s1e = mf.get_ovlp()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)
        if dm is None: dm = mf.make_rdm1()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array((dm*.5, dm*.5))
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        get_roothaan_fock = CAHF_get_roothaan_fock(frac)
        f = get_roothaan_fock((focka,fockb), dm, s1e)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if diis_start_cycle is None:
            diis_start_cycle = mf.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = mf.level_shift
        if damp_factor is None:
            damp_factor = mf.damp

        dm_tot = dm[0] + dm[1]
        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
            raise NotImplementedError('ROHF Fock-damping')
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm_tot, f, mf, h1e, vhf, f_prev=fock_last)
        if abs(level_shift_factor) > 1e-4:
            f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
        f = lib.tag_array(f, focka=focka, fockb=fockb)
        return f
    return _get_fock

from pyscf.lib import logger
def CAHF_get_occ(ncas, nelecas):
    def _get_occ(mf, mo_energy=None, mo_coeff=None):
        '''Label the occupancies for each orbital.
        NOTE the occupancies are not assigned based on the orbital energy ordering.
        The first N orbitals are assigned to be occupied orbitals.

        Examples:

        >>> mol = gto.M(atom='H 0 0 0; O 0 0 1.1', spin=1)
        >>> mf = scf.hf.SCF(mol)
        >>> energy = numpy.array([-10., -1., 1, -2., 0, -3])
        >>> mf.get_occ(energy)
        array([2, 2, 2, 2, 1, 0])
        '''

        if mo_energy is None: mo_energy = mf.mo_energy
        if getattr(mo_energy, 'mo_ea', None) is not None:
            mo_ea = mo_energy.mo_ea
            mo_eb = mo_energy.mo_eb
        else:
            mo_ea = mo_eb = mo_energy
        nmo = mo_ea.size
        if getattr(mf, 'nelec', None) is None:
            nelec = mf.mol.nelec
        else:
            nelec = mf.nelec
        if nelec[0] > nelec[1]:
            nocc, ncore = nelec
        else:
            ncore, nocc = nelec
        nopen = nocc - ncore
        # mo_occ = _fill_rohf_occ(mo_energy, mo_ea, mo_eb, ncore, nopen)

        ncore = int((np.sum(np.array(mf.mol.nelec))-nelecas)/2)
        mo_occ = np.zeros(mf.mol.nao)
        mo_occ[:ncore] = 2
        mo_occ[ncore:ncore+ncas] = nelecas/ncas

        if mf.verbose >= logger.INFO and nocc < nmo and ncore > 0:
            ehomo = max(mo_energy[mo_occ> 0])
            elumo = min(mo_energy[mo_occ==0])
            if ehomo+1e-3 > elumo:
                logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', ehomo, elumo)
            else:
                logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', ehomo, elumo)
            if nopen > 0 and mf.verbose >= logger.DEBUG:
                core_idx = mo_occ == 2
                open_idx = mo_occ == 1
                vir_idx = mo_occ == 0
                logger.debug(mf, '                  Roothaan           | alpha              | beta')
                logger.debug(mf, '  Highest 2-occ = %18.15g | %18.15g | %18.15g',
                            max(mo_energy[core_idx]),
                            max(mo_ea[core_idx]), max(mo_eb[core_idx]))
                logger.debug(mf, '  Lowest 0-occ =  %18.15g | %18.15g | %18.15g',
                            min(mo_energy[vir_idx]),
                            min(mo_ea[vir_idx]), min(mo_eb[vir_idx]))
                for i in np.where(open_idx)[0]:
                    logger.debug(mf, '  1-occ =         %18.15g | %18.15g | %18.15g',
                                mo_energy[i], mo_ea[i], mo_eb[i])

            if mf.verbose >= logger.DEBUG:
                np.set_printoptions(threshold=nmo)
                logger.debug(mf, '  Roothaan mo_energy =\n%s', mo_energy)
                logger.debug1(mf, '  alpha mo_energy =\n%s', mo_ea)
                logger.debug1(mf, '  beta  mo_energy =\n%s', mo_eb)
                np.set_printoptions(threshold=1000)
        return mo_occ
    return _get_occ

def get_grad(mo_coeff, mo_occ, fock, f):
    '''ROHF gradients is the off-diagonal block [co + cv + ov], where
    [ cc co cv ]
    [ oc oo ov ]
    [ vc vo vv ]
    '''
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa.reshape(-1,1) & occidxa # vc vo
    uniq_var_b = viridxb.reshape(-1,1) & occidxb # oc vc

    if getattr(fock, 'focka', None) is not None:
        focka = fock.focka
        fockb = fock.fockb
    elif isinstance(fock, (tuple, list)) or getattr(fock, 'ndim', None) == 3:
        focka, fockb = fock
    else:
        focka = fockb = fock
    focka = mo_coeff.conj().T.dot(focka).dot(mo_coeff)
    fockb = mo_coeff.conj().T.dot(fockb).dot(mo_coeff)

    openidx = viridxb & occidxa
    oc_block = openidx.reshape(-1,1) & occidxb
    vo_block = viridxa.reshape(-1,1) & openidx
    vc_block = viridxa.reshape(-1,1) & occidxb
    g = np.zeros_like(focka)
    # g[uniq_var_a]  = focka[uniq_var_a]
    # g[uniq_var_b] += fockb[uniq_var_b]

    g[oc_block] = fockb[oc_block]
    g[vo_block] = focka[vo_block]
    g[vc_block] = 2*f*focka[vc_block] + 2*(1-f)*fockb[vc_block]
    return g[uniq_var_a | uniq_var_b]

def CAHF_get_grad(f):
    def _get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock, f)
    return _get_grad

def CAHF_energy_elec(frac):
    def _energy_elec(mf, dm=None, h1e=None, vhf=None):
        '''Electronic energy of Unrestricted Hartree-Fock

        Note this function has side effects which cause mf.scf_summary updated.

        Returns:
            Hartree-Fock electronic energy and the 2-electron part contribution
        '''
        if dm is None: dm = mf.make_rdm1()
        if h1e is None:
            h1e = mf.get_hcore()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array((dm*.5, dm*.5))
        if vhf is None:
            vhf = mf.get_veff(mf.mol, dm)
        if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
            h1e = (h1e, h1e)
        e1 = np.einsum('ij,ji->', h1e[0], dm[0]) * 2 * frac
        e1+= np.einsum('ij,ji->', h1e[1], dm[1]) * 2 * (1-frac)
        e_coul =np.einsum('ij,ji->', vhf[0], dm[0]) * frac + \
                np.einsum('ij,ji->', vhf[1], dm[1]) * (1-frac)
        e_elec = (e1 + e_coul).real

        mf.scf_summary['e1'] = e1.real
        mf.scf_summary['e2'] = e_coul.real
        logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
        return e_elec, e_coul
    return _energy_elec

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    '''
    moditfy UHF.init_guess_by_chkfile
    '''
    from pyscf import gto
    from pyscf.scf import addons, chkfile
    import numpy, scipy
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)
    if project is None:
        project = not gto.same_basis_set(chk_mol, mol)

    # Check whether the two molecules are similar
    im1 = scipy.linalg.eigvalsh(mol.inertia_moment())
    im2 = scipy.linalg.eigvalsh(chk_mol.inertia_moment())
    # im1+1e-7 to avoid 'divide by zero' error
    if abs((im1-im2)/(im1+1e-7)).max() > 0.01:
        logger.warn(mol, "Large deviations found between the input "
                    "molecule and the molecule from chkfile\n"
                    "Initial guess density matrix may have large error.")

    if project:
        s = hf.get_ovlp(mol)

    def fproj(mo):
        if project:
            mo = addons.project_mo_nr2nr(chk_mol, mo, mol)
            norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
            mo /= numpy.sqrt(norm)
        return mo

    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if getattr(mo[0], 'ndim', None) == 1:  # RHF
        if numpy.iscomplexobj(mo):
            raise NotImplementedError('TODO: project DHF orbital to UHF orbital')
        mo_coeff = fproj(mo)
        mo_occa = (mo_occ>1e-8).astype(numpy.double)
        mo_occb = mo_occ - mo_occa
        # dm = scf.rohf.make_rdm1([mo_coeff,mo_coeff], [mo_occa,mo_occb])
        dm = scf.rohf.make_rdm1(mo_coeff, mo_occ)
    else:  #UHF
        if getattr(mo[0][0], 'ndim', None) == 2:  # KUHF
            logger.warn(mol, 'k-point UHF results are found.  Density matrix '
                        'at Gamma point is used for the molecular SCF initial guess')
            mo = mo[0]
        dm = scf.rohf.make_rdm1([fproj(mo[0]),fproj(mo[1])], mo_occ)
    return dm

from pyscf.soscf.newton_ah import _CIAH_SOSCF, gen_g_hop_uhf
def CAHF_gen_g_hop(frac):
    def _gen_g_hop(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                    with_symmetry=True):
        if getattr(fock_ao, 'focka', None) is None:
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            fock_ao = mf.get_fock(h1e, dm=dm0)
        fock_ao = fock_ao.focka, fock_ao.fockb
        mo_occa = occidxa = mo_occ > 0
        mo_occb = occidxb = mo_occ ==2
        ug, uh_op, uh_diag = gen_g_hop_uhf(mf, (mo_coeff,)*2, (mo_occa,mo_occb),
                                        fock_ao, None, with_symmetry)

        viridxa = ~occidxa
        viridxb = ~occidxb
        uniq_var_a = viridxa[:,None] & occidxa
        uniq_var_b = viridxb[:,None] & occidxb
        uniq_ab = uniq_var_a | uniq_var_b
        nmo = mo_coeff.shape[-1]
        nocca = np.count_nonzero(mo_occa)
        nvira = nmo - nocca

        def sum_ab(x):
            x1 = np.zeros((nmo,nmo), dtype=x.dtype)
            x1[uniq_var_a]  = x[:nvira*nocca]
            x1[uniq_var_b] += x[nvira*nocca:]

            x_aug = np.zeros((nmo,nmo), dtype=x.dtype)
            x_aug[uniq_var_a]  = x[:nvira*nocca]
            x_aug[uniq_var_b] -= x[nvira*nocca:]

            x1[uniq_var_a & uniq_var_b] += (2*frac-1) * x_aug[uniq_var_a & uniq_var_b]
            return x1[uniq_ab]

        g = sum_ab(ug)
        h_diag = sum_ab(uh_diag)
        def h_op(x):
            x1 = np.zeros((nmo,nmo), dtype=x.dtype)
            # unpack ROHF rotation parameters
            x1[uniq_ab] = x
            x1 = np.hstack((x1[uniq_var_a],x1[uniq_var_b]))
            return sum_ab(uh_op(x1))
        return g, h_op, h_diag
    return _gen_g_hop

class _SecondOrderCAHF(_CIAH_SOSCF):
    def gen_g_hop(self, *args, **kwargs):
        _gen_g_hop = CAHF_gen_g_hop(self._scf.frac)
        return _gen_g_hop(self, *args, **kwargs)

from pyscf.soscf.newton_ah import newton as pyscf_newton

def cahf_newton(mf):
    print('cahf_newton called')
    from pyscf import scf
    if isinstance(mf, _CIAH_SOSCF):
        return mf

    assert isinstance(mf, hf.SCF)

    if mf.istype('CAHF'):
        cls = _SecondOrderCAHF
        return lib.set_class(cls(mf), (cls, mf.__class__))
    else:
        return pyscf_newton(mf)

from pyscf.soscf import newton_ah
newton_ah.newton = cahf_newton

class CAHF(scf.rohf.ROHF):
    def __init__(self, mol, ncas, nelecas, spin):
        hf.SCF.__init__(self, mol)
        self.conv_check = False
        self.nelec = None
        self.ncas = ncas
        self.nelecas = nelecas
        self.spin = spin
        self.frac, self.coulomb_a, self.exchange_b, self.alpha, self.beta = \
            get_coeffs(ncas, nelecas, spin)

    def get_veff(self, *args, **kwargs):
        _get_veff = CAHF_get_veff(self.frac, self.alpha, self.beta)
        return _get_veff(self, *args, **kwargs)
    
    def get_grad(self, *args, **kwargs):
        _get_grad = CAHF_get_grad(self.frac)
        return _get_grad(self, *args, **kwargs)
    
    def get_fock(self, *args, **kwargs):
        _get_fock = CAHF_get_fock(self.frac)
        return _get_fock(self, *args, **kwargs)
    
    def get_occ(self, *args, **kwargs):
        _get_occ = CAHF_get_occ(self.ncas, self.nelecas)
        return _get_occ(self, *args, **kwargs)
    
    def energy_elec(self, *args, **kwargs):
        _energy_elec = CAHF_energy_elec(self.frac)
        return _energy_elec(self, *args, **kwargs)
    
    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)


if __name__ == '__main__':
    get_coeffs(5, 2, 2)