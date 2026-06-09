from dataclasses import dataclass
from functools import reduce
from numbers import Integral

import numpy as np
from scipy.linalg import null_space
from pyscf.lo.orth import lowdin

from pyscf import gto, scf, ao2mo
from pyscf.lib import logger

from embed_sim import ssdmet


@dataclass
class FragmentReference:
    """Container for one impurity-ligand fragment reference calculation."""
    mol: gto.mole.Mole
    mf: object
    local_to_global_idx: list
    local_imp_idx: list
    local_fragment_idx: list


def _parse_aolabels(mol, aolabels, name):
    indices = [int(x) for x in gto.mole._aolabels2baslst(mol, aolabels, base=0)]
    if len(indices) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate AO indices")
    return indices


def _parse_fragments(mol, fragments):
    if fragments is None:
        return None
    return [_parse_aolabels(mol, frag, "fragment") for frag in fragments]


def _atom_ids_from_ao_indices(mol, ao_idx):
    ao_labels = mol.ao_labels(fmt=False)
    return sorted({ao_labels[idx][0] for idx in ao_idx})


def _resolve_local_ao_indices(mol, ao_idx, name):
    if isinstance(ao_idx, str):
        return _parse_aolabels(mol, ao_idx, name)
    try:
        indices = list(ao_idx)
    except TypeError:
        raise TypeError(f"{name} must be AO labels or AO indices")
    if all(isinstance(idx, Integral) for idx in indices):
        return [int(idx) for idx in indices]
    return _parse_aolabels(mol, ao_idx, name)


def _resolve_impurity_embedded_indices(obj, ao_idx, name):
    if isinstance(ao_idx, str):
        indices = obj.search_impurity_ao_label(ao_idx).tolist()
        if len(indices) == 0:
            raise ValueError(f"{name} does not match any impurity AO labels")
        return indices
    try:
        indices = list(ao_idx)
    except TypeError:
        raise TypeError(f"{name} must be AO labels or AO indices")
    if all(isinstance(idx, int) for idx in indices):
        return indices
    indices = obj.search_impurity_ao_label(ao_idx).tolist()
    if len(indices) == 0:
        raise ValueError(f"{name} does not match any impurity AO labels")
    return indices


def _make_fragment_mol(parent_mol, impurity_idx, fragment_idx, charge=None,
                       spin=None, verbose=None):
    global_idx = list(impurity_idx) + list(fragment_idx)
    if len(set(global_idx)) != len(global_idx):
        raise ValueError("impurity and ligand fragment AO indices overlap")

    atom_idx = _atom_ids_from_ao_indices(parent_mol, global_idx)
    if spin is None:
        spin = parent_mol.spin
    if verbose is None:
        verbose = parent_mol.verbose

    atom = [
        [parent_mol.atom_symbol(i), parent_mol.atom_coord(i)]
        for i in atom_idx
    ]
    frag_mol = gto.M(
        atom=atom,
        basis=parent_mol._basis,
        ecp=parent_mol._ecp,
        charge=charge,
        spin=spin,
        unit='Bohr',
        cart=parent_mol.cart,
        symmetry=False,
        verbose=verbose,
        max_memory=parent_mol.max_memory,
    )

    imp_set = set(impurity_idx)
    frag_set = set(fragment_idx)
    old_atom_by_new = dict(enumerate(atom_idx))
    parent_ao_by_atom = {atom_id: [] for atom_id in atom_idx}
    for parent_ao, label in enumerate(parent_mol.ao_labels(fmt=False)):
        atom_id = label[0]
        if atom_id in parent_ao_by_atom:
            parent_ao_by_atom[atom_id].append(parent_ao)
    local_count_by_atom = {atom_id: 0 for atom_id in atom_idx}
    local_to_global_idx = []
    local_imp_idx = []
    local_fragment_idx = []
    for local_ao, label in enumerate(frag_mol.ao_labels(fmt=False)):
        old_atom = old_atom_by_new[label[0]]
        offset = local_count_by_atom[old_atom]
        parent_ao = parent_ao_by_atom[old_atom][offset]
        local_count_by_atom[old_atom] += 1
        local_to_global_idx.append(parent_ao)
        if parent_ao in imp_set:
            local_imp_idx.append(local_ao)
        elif parent_ao in frag_set:
            local_fragment_idx.append(local_ao)

    return frag_mol, local_to_global_idx, local_imp_idx, local_fragment_idx


def run_fragment_scf(mol, impurity_idx, fragment_idx, charge,
                     fragment_scf="rohf", fragment_scf_verbose=3, **kwargs):
    """
    Run a low-level reference calculation for one impurity-fragment subsystem.
    """
    spin = kwargs.pop('spin', None)
    frag_mol, local_to_global_idx, local_imp_idx, local_fragment_idx = \
        _make_fragment_mol(mol, impurity_idx, fragment_idx, charge=charge,
                           spin=spin, verbose=fragment_scf_verbose)

    fragment_scf = fragment_scf.lower()
    if fragment_scf == "cahf":
        from embed_sim import cahf
        ncas = kwargs.pop('ncas', 5)
        nelecas = kwargs.pop('nelecas', 7)
        cahf_spin = kwargs.pop('cahf_spin', frag_mol.spin)
        mf = cahf.CAHF(frag_mol, ncas=ncas, nelecas=nelecas,
                       spin=cahf_spin).x2c()
    elif fragment_scf == "rohf":
        mf = scf.rohf.ROHF(frag_mol).x2c()
    else:
        raise ValueError(f"unsupported fragment_scf {fragment_scf}")

    mf.max_memory = kwargs.pop('max_memory', mol.max_memory)
    mf.max_cycle = kwargs.pop('max_cycle', getattr(mf, 'max_cycle', 50))
    mf.level_shift = kwargs.pop('level_shift', getattr(mf, 'level_shift', 0))

    diis = kwargs.pop('diis', None)
    if diis is not None:
        if isinstance(diis, str):
            if diis.lower() != 'rdiis':
                raise ValueError(f"unsupported fragment DIIS {diis}")
            from embed_sim import rdiis
            rdiis_imp_idx = kwargs.pop('rdiis_imp_idx', local_imp_idx)
            rdiis_imp_idx = _resolve_local_ao_indices(
                frag_mol, rdiis_imp_idx, 'rdiis_imp_idx'
            )
            mf.diis = rdiis.RDIIS(
                rdiis_prop=kwargs.pop('rdiis_prop', 'dS'),
                imp_idx=rdiis_imp_idx,
                power=kwargs.pop('rdiis_power', 0.2),
                kernel=kwargs.pop('rdiis_kernel', None),
                mute=kwargs.pop(
                    'rdiis_mute', fragment_scf_verbose < logger.INFO
                ),
            )
            if 'rdiis_ent_conv_tol' in kwargs:
                mf.diis.ent_conv_tol = kwargs.pop('rdiis_ent_conv_tol')
            if 'rdiis_space' in kwargs:
                mf.diis.space = kwargs.pop('rdiis_space')
        else:
            mf.diis = diis

    if kwargs:
        raise TypeError(f"unknown fragment SCF options: {sorted(kwargs)}")
    mf.kernel()

    return FragmentReference(
        mol=frag_mol,
        mf=mf,
        local_to_global_idx=local_to_global_idx,
        local_imp_idx=local_imp_idx,
        local_fragment_idx=local_fragment_idx,
    )


def fragment_bath_from_fragment_density(dm, impurity_idx, fragment_idx,
                                        mol=None, overlap=None,
                                        threshold=1e-13):
    """
    Construct bath orbitals from one fragment density matrix.

    The input density is in the local fragment AO basis.  It is transformed by
    the same impurity-preserving Lowdin path used by SSDMET, then restricted to
    the impurity + selected ligand-fragment axes.  Only the fragment/environment
    density block is diagonalized, so the returned bath orbitals do not mix
    with the fixed impurity block.
    """
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("fragment density must be a square AO matrix")
    if mol is None and overlap is None:
        raise ValueError("mol or overlap must be provided")
    if overlap is not None and overlap.shape != dm.shape:
        raise ValueError("overlap and density shapes are inconsistent")

    active_idx = list(impurity_idx) + list(fragment_idx)
    if len(set(active_idx)) != len(active_idx):
        raise ValueError("impurity and fragment AO indices overlap")
    if max(active_idx) >= dm.shape[0] or min(active_idx) < 0:
        raise ValueError("fragment AO index is out of range")

    caolo, cloao = ssdmet.lowdin_orth(
        mol, ovlp=overlap, imp_idx=list(impurity_idx),
        preserve_imp=True
    )
    ldm = reduce(np.dot, (cloao, dm, cloao.conj().T))

    ldm_active = ldm[active_idx, :][:, active_idx]
    nimp = len(impurity_idx)
    ldm_env = ldm_active[nimp:, nimp:]
    occ_env, orb_env = np.linalg.eigh(ldm_env)

    fv_idx = np.nonzero(occ_env < threshold)[0]
    bath_idx = np.nonzero(
        (occ_env >= threshold) & (occ_env <= 2-threshold)
    )[0]
    fo_idx = np.nonzero(occ_env > 2-threshold)[0]

    env_orb = caolo[:, fragment_idx]
    bath_orb = env_orb @ orb_env[:, bath_idx]
    fo_orb = env_orb @ orb_env[:, fo_idx]
    fv_orb = env_orb @ orb_env[:, fv_idx]
    return bath_orb, fo_orb, fv_orb


def fragment_orbitals_to_global_orbitals(local_orb, local_to_global_idx, nao):
    global_orb = np.zeros((nao, local_orb.shape[1]), dtype=local_orb.dtype)
    for local_ao, global_ao in enumerate(local_to_global_idx):
        global_orb[global_ao, :] = local_orb[local_ao, :]
    return global_orb


def _global_preserve_imp_basis(mol, impurity_idx):
    caolo, cloao = ssdmet.lowdin_orth(
        mol, imp_idx=list(impurity_idx), preserve_imp=True
    )
    return caolo, cloao


def _orthogonalize_global_bath(mol, bath_blocks):
    if len(bath_blocks) == 0:
        return np.zeros((mol.nao, 0))
    nonempty_blocks = [block for block in bath_blocks if block.size]
    if len(nonempty_blocks) == 0:
        return np.zeros((mol.nao, 0))
    raw_bath = np.hstack(nonempty_blocks)
    if raw_bath.shape[1] == 0:
        return np.zeros((mol.nao, 0))

    s = mol.intor_symmetric('int1e_ovlp')
    # Fragment baths are already orthogonal to the fixed impurity by
    # construction.  This step only orthonormalizes bath blocks from different
    # fragments in the parent AO metric.
    bath_overlap = raw_bath.T.conj() @ s @ raw_bath
    bath_overlap = (bath_overlap + bath_overlap.T.conj()) * 0.5
    return raw_bath @ lowdin(bath_overlap)


def _complete_global_orbitals(mol, impurity_idx, bath_orb, fo_orb=None,
                              svd_tol=1e-10):
    caolo, cloao = _global_preserve_imp_basis(mol, impurity_idx)
    env_idx = [idx for idx in range(mol.nao) if idx not in impurity_idx]
    c_imp = caolo[:, impurity_idx]
    if fo_orb is None:
        fo_orb = np.zeros((mol.nao, 0))

    env_basis = caolo[:, env_idx]
    if bath_orb.shape[1] == 0:
        qbath = np.zeros((len(env_idx), 0))
    else:
        bath_coord = cloao @ bath_orb
        env_bath_coord = bath_coord[env_idx, :]
        qbath, rbath = np.linalg.qr(env_bath_coord)
        rank = np.sum(np.abs(np.diag(rbath)) > svd_tol)
        qbath = qbath[:, :rank]
        bath_orb = env_basis @ qbath

    if fo_orb.shape[1] == 0:
        qfo = np.zeros((len(env_idx), 0))
    else:
        fo_coord = cloao @ fo_orb
        env_fo_coord = fo_coord[env_idx, :]
        if qbath.shape[1] > 0:
            env_fo_coord = env_fo_coord - qbath @ (
                qbath.T.conj() @ env_fo_coord
            )
        qfo, rfo = np.linalg.qr(env_fo_coord)
        rank = np.sum(np.abs(np.diag(rfo)) > svd_tol)
        qfo = qfo[:, :rank]
        fo_orb = env_basis @ qfo

    occupied_env = np.hstack((qbath, qfo))
    if occupied_env.shape[1] == 0:
        qcomp = np.eye(len(env_idx))
    else:
        qcomp = null_space(occupied_env.T.conj(), rcond=svd_tol)
    c_fv = caolo[:, env_idx] @ qcomp
    return c_imp, bath_orb, fo_orb, c_fv


def _aufbau_occ(nelectron, spin, norb):
    nalpha = (nelectron + spin) // 2
    nbeta = (nelectron - spin) // 2
    if nalpha + nbeta != nelectron or nalpha < nbeta:
        raise ValueError("inconsistent electron count and spin")
    if nalpha > norb:
        raise ValueError(
            "fragment-DMET without frozen occupied orbitals assigns all "
            "electrons to the impurity+bath space, but this space has fewer "
            "orbitals than alpha electrons"
        )
    occ = np.zeros(norb)
    occ[:nbeta] = 2
    occ[nbeta:nalpha] = 1
    return occ


def _fragment_density_to_parent_density(mol, fragment_refs):
    dm_global = np.zeros((mol.nao, mol.nao))
    dm_imp_acc = np.zeros((mol.nao, mol.nao))
    imp_count = np.zeros((mol.nao, mol.nao))

    for frag_ref in fragment_refs:
        dm = ssdmet.mf_or_cas_make_rdm1s(frag_ref.mf)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        l2g = np.asarray(frag_ref.local_to_global_idx, dtype=int)
        loc_imp = np.asarray(frag_ref.local_imp_idx, dtype=int)
        loc_frag = np.asarray(frag_ref.local_fragment_idx, dtype=int)
        glob_imp = l2g[loc_imp]
        glob_frag = l2g[loc_frag]

        dm_global[np.ix_(glob_frag, glob_frag)] += dm[
            np.ix_(loc_frag, loc_frag)
        ]
        dm_global[np.ix_(glob_imp, glob_frag)] += dm[
            np.ix_(loc_imp, loc_frag)
        ]
        dm_global[np.ix_(glob_frag, glob_imp)] += dm[
            np.ix_(loc_frag, loc_imp)
        ]
        dm_imp_acc[np.ix_(glob_imp, glob_imp)] += dm[
            np.ix_(loc_imp, loc_imp)
        ]
        imp_count[np.ix_(glob_imp, glob_imp)] += 1

    mask = imp_count > 0
    dm_global[mask] = dm_imp_acc[mask] / imp_count[mask]
    return (dm_global + dm_global.T.conj()) * 0.5


def _rescale_density_trace(dm, nelectron):
    trace = np.trace(dm).real
    if abs(trace) < 1e-12:
        raise ValueError("fragment-projected density has near-zero trace")
    return dm * (nelectron / trace), trace, nelectron / trace


class FDMET(ssdmet.SSDMET):
    """
    Fragment-DMET scaffold.

    Parameters
    ----------
    mol
        PySCF Mole object.  FDMET does not create or consume a whole-system SCF
        reference for bath construction.
    imp_idx
        Impurity AO labels.
    imp_charge
        Charge assigned to the impurity fragment.
    fragments
        Ligand fragment AO labels.  Each item is one fragment.
    fragment_charges
        Charge assigned to each ligand fragment.
    """
    def __init__(self, mol, title='untitled', 
                 imp_idx=None,imp_charge=None, 
                 fragments=None, fragment_charges=None,
                 fragment_scf='cahf',
                 fragment_scf_options=None, threshold=1e-13,
                 keep_fv_orbitals=False,
                 embedded_init_guess='aufbau',
                 embedded_active_aolabels=None,
                 verbose=logger.INFO):
        
        self.imp_charge = imp_charge
        self.fragment_charges = None
        self.fragment_scf = fragment_scf
        self.fragment_scf_options = (
            {} if fragment_scf_options is None else dict(fragment_scf_options)
        )
        self.keep_fv_orbitals = keep_fv_orbitals # for debug
        self.embedded_init_guess = embedded_init_guess
        self.embedded_active_aolabels = embedded_active_aolabels

        self.mol = mol
        self.max_mem = getattr(self.mol, 'max_memory', 4000)
        self.title = title
        self.verbose = verbose
        self.mf_or_cas = scf.rohf.ROHF(self.mol).x2c()
        self.mf_or_cas.max_memory = self.max_mem
        self.dm = None
        self._imp_idx = []
        self.imp_idx = imp_idx
        self._build_impurity_label_map()
        self.threshold = threshold

        self.fo_orb = None
        self.fv_orb = None
        self.es_orb = None
        self.es_occ = None

        self.nfo = None
        self.nfv = None
        self.nes = None
        self.nimp = None
        self.nbath = None
        self.nembedded_fo = None
        self.nkept_fv = None

        self.es_int1e = None
        self.es_int2e = None

        self.es_mf = None
        self.es_dm = None
        self.es_init_guess_info = None
        self.fragment_refs = None
        self.fragments = _parse_fragments(self.mol, fragments)
        if self.fragments is not None:
            if fragment_charges is not None:
                self.fragment_charges = list(fragment_charges)
                if len(self.fragment_charges) != len(self.fragments):
                    raise ValueError(
                        "fragment_charges must have the same length as fragments"
                    )

    def _build_impurity_label_map(self):
        self.impurity_ao_to_embedded_idx = {
            int(parent_idx): embedded_idx
            for embedded_idx, parent_idx in enumerate(self.imp_idx)
        }
        ao_labels = self.mol.ao_labels()
        self.impurity_aolabel_to_embedded_idx = {
            ao_labels[int(parent_idx)]: embedded_idx
            for embedded_idx, parent_idx in enumerate(self.imp_idx)
        }

    def search_impurity_ao_label(self, aolabels, base=0):
        parent_idx = [
            int(idx)
            for idx in gto.mole._aolabels2baslst(self.mol, aolabels, base=0)
        ]
        embedded_idx = [
            self.impurity_ao_to_embedded_idx[idx]
            for idx in parent_idx
            if idx in self.impurity_ao_to_embedded_idx
        ]
        return np.asarray(embedded_idx, dtype=int) + base

    def dump_flags(self):
        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('fragment scf = %s', self.fragment_scf)
        log.info('fragment scf options = %s', self.fragment_scf_options)
        log.info('embedded init guess = %s', self.embedded_init_guess)
        log.info('embedded active AO labels = %s',
                 self.embedded_active_aolabels)
        log.info('keep frozen virtual orbitals in embedded space = %s',
                 self.keep_fv_orbitals)
        log.info('impurity charge = %s', self.imp_charge)
        if self.fragments is None:
            log.info('fragments = None')
        else:
            log.info('number of fragments = %d', len(self.fragments))
            log.info('fragment charges = %s', self.fragment_charges)

    def build(self, fragment_scf_verbose=3, chk_fname_load='', save_chk=False):
        self.dump_flags()
        self.fragment_refs = []
        for ifrag, fragment_idx in enumerate(self.fragments):
            logger.info(self, 'build impurity-ligand fragment %d', ifrag)
            scf_options = dict(self.fragment_scf_options)
            charge = self.imp_charge + self.fragment_charges[ifrag]
            frag_ref = run_fragment_scf(
                self.mol, self.imp_idx, fragment_idx,
                charge=charge,
                fragment_scf=self.fragment_scf,
                fragment_scf_verbose=fragment_scf_verbose,
                **scf_options,
            )
            self.fragment_refs.append(frag_ref)

        bath_blocks = []
        fo_blocks = []
        for frag_ref in self.fragment_refs:
            bath_orb, fo_orb, _ = fragment_bath_from_fragment_density(
                ssdmet.mf_or_cas_make_rdm1s(frag_ref.mf),
                frag_ref.local_imp_idx,
                frag_ref.local_fragment_idx,
                mol=frag_ref.mol,
                threshold=self.threshold,
            )
            bath_blocks.append(fragment_orbitals_to_global_orbitals(
                bath_orb, frag_ref.local_to_global_idx, self.mol.nao
            ))
            fo_blocks.append(fragment_orbitals_to_global_orbitals(
                fo_orb, frag_ref.local_to_global_idx, self.mol.nao
            ))

        bath_orb = _orthogonalize_global_bath(self.mol, bath_blocks)
        fo_orb = _orthogonalize_global_bath(self.mol, fo_blocks)
        c_imp, bath_orb, self.fo_orb, self.fv_orb = _complete_global_orbitals(
            self.mol, self.imp_idx, bath_orb, fo_orb=fo_orb,
            svd_tol=self.threshold
        )
        nbath = bath_orb.shape[1]
        nembedded_fo = self.fo_orb.shape[1]
        if self.keep_fv_orbitals:
            nkept_fv = self.fv_orb.shape[1]
        else:
            nkept_fv = 0

        es_env_blocks = [bath_orb]
        if self.fo_orb.shape[1] > 0:
            es_env_blocks.append(self.fo_orb)
            self.fo_orb = np.zeros((self.mol.nao, 0))
        if self.keep_fv_orbitals and self.fv_orb.shape[1] > 0:
            es_env_blocks.append(self.fv_orb)
            self.fv_orb = np.zeros((self.mol.nao, 0))

        es_env_orb = np.hstack(es_env_blocks)
        self.es_orb = np.hstack((c_imp, es_env_orb))
        self.nfo = self.fo_orb.shape[1]
        self.nfv = self.fv_orb.shape[1]
        self.nes = self.es_orb.shape[1]
        self.nimp = c_imp.shape[1]
        self.nbath = nbath
        self.nembedded_fo = nembedded_fo
        self.nkept_fv = nkept_fv
        self.es_occ = _aufbau_occ(
            self.mol.nelectron - 2*self.nfo, self.mol.spin, self.nes
        )

        logger.info(self, 'number of impurity orbitals %d', len(self.imp_idx))
        logger.info(self, 'number of bath orbitals %d', nbath)
        logger.info(self, 'number of occupied environment orbitals embedded %d',
                    nembedded_fo)
        logger.info(self, 'number of frozen virtual orbitals kept in embedded space %d',
                    nkept_fv)
        logger.info(self, 'number of frozen occupied orbitals %d', self.nfo)
        logger.info(self, 'number of frozen virtual orbitals %d', self.nfv)

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        if save_chk:
            raise NotImplementedError(
                "checkpoint saving is not implemented for FDMET fragment "
                "bath builds because no global reference density is stored"
            )
        if chk_fname_load:
            logger.warn(
                self,
                'chk_fname_load is ignored for FDMET fragment bath builds'
            )
        self.es_mf = self.CAHF(**self.fragment_scf_options)
        logger.info(self, 'energy from frozen occupied orbitals %s',
                    self.fo_ene(e_nuc=False))
        logger.info(self, 'nuclear repulsion energy %s',
                    self.mol.energy_nuc())
        return self.es_mf

    def CAHF(self, run_mf=False, **scf_options):
        from embed_sim import cahf

        mol = gto.M()
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelectron = self.mol.nelectron - 2*self.nfo
        mol.spin = scf_options.pop('spin', self.mol.spin)

        ncas = scf_options.pop('ncas', min(self.nes, max(2, mol.nelectron)))
        nelecas = scf_options.pop('nelecas', min(mol.nelectron, 2*ncas))
        cahf_spin = scf_options.pop('cahf_spin', mol.spin)

        es_mf = cahf.CAHF(mol, ncas=ncas, nelecas=nelecas,
                          spin=cahf_spin).x2c()
        es_mf.max_memory = scf_options.pop('max_memory', self.max_mem)
        es_mf.max_cycle = scf_options.pop(
            'max_cycle', getattr(es_mf, 'max_cycle', 50)
        )
        es_mf.level_shift = scf_options.pop(
            'level_shift', getattr(es_mf, 'level_shift', 0)
        )
        es_mf.mo_energy = np.zeros((self.nes))

        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: np.eye(self.nes)
        es_mf._eri = ao2mo.restore(8, self.es_int2e, self.nes)

        def _get_embedded_occ(mf, mo_energy=None, mo_coeff=None):
            if mo_energy is None:
                mo_energy = mf.mo_energy
            nmo = mo_energy.size
            ncore = int((mf.mol.nelectron - nelecas) / 2)
            if ncore < 0 or ncore + ncas > nmo:
                raise ValueError(
                    "embedded CAHF active space is inconsistent with the "
                    "number of embedded orbitals"
                )
            mo_occ = np.zeros(nmo)
            mo_occ[:ncore] = 2
            mo_occ[ncore:ncore+ncas] = nelecas/ncas
            return mo_occ

        es_mf.get_occ = _get_embedded_occ.__get__(es_mf, es_mf.__class__)

        diis = scf_options.pop('diis', None)
        if diis is not None:
            if isinstance(diis, str):
                if diis.lower() != 'rdiis':
                    raise ValueError(f"unsupported embedded DIIS {diis}")
                from embed_sim import rdiis
                default_imp_idx = list(range(len(self.imp_idx)))
                rdiis_imp_idx = scf_options.pop(
                    'rdiis_imp_idx', default_imp_idx
                )
                rdiis_imp_idx = _resolve_impurity_embedded_indices(
                    self, rdiis_imp_idx, 'rdiis_imp_idx'
                )
                es_mf.diis = rdiis.RDIIS(
                    rdiis_prop=scf_options.pop('rdiis_prop', 'dS'),
                    imp_idx=rdiis_imp_idx,
                    power=scf_options.pop('rdiis_power', 0.2),
                    kernel=scf_options.pop('rdiis_kernel', None),
                    mute=scf_options.pop('rdiis_mute',
                                         self.verbose < logger.INFO),
                )
                if 'rdiis_ent_conv_tol' in scf_options:
                    es_mf.diis.ent_conv_tol = scf_options.pop(
                        'rdiis_ent_conv_tol'
                    )
                if 'rdiis_space' in scf_options:
                    es_mf.diis.space = scf_options.pop('rdiis_space')
            else:
                es_mf.diis = diis

        if scf_options:
            raise TypeError(
                f"unknown embedded CAHF options: {sorted(scf_options)}"
            )

        mo_coeff, mo_occ, es_dm = self._embedded_cahf_initial_guess(
            ncas, nelecas, mol.nelectron
        )
        self.es_dm = es_dm
        es_mf.mo_coeff = mo_coeff
        es_mf.mo_occ = mo_occ
        es_mf.get_init_guess = lambda *args, **kwargs: self.es_dm
        es_mf.init_guess = self.embedded_init_guess

        if run_mf:
            es_mf.kernel(self.es_dm)
            self.es_occ = es_mf.mo_occ
        return es_mf

    def _embedded_cahf_initial_guess(self, ncas, nelecas, nelectron):
        if self.embedded_init_guess != 'fragment_density':
            es_dm = np.diag(self.es_occ)
            self.es_init_guess_info = {'kind': 'aufbau'}
            return np.eye(self.nes), self.es_occ, es_dm

        if self.embedded_active_aolabels is None:
            raise ValueError(
                "embedded_active_aolabels is required for "
                "embedded_init_guess='fragment_density'"
            )
        active_idx = self.search_impurity_ao_label(
            self.embedded_active_aolabels
        )
        if active_idx.size != ncas:
            raise ValueError(
                "embedded active AO labels must select exactly ncas orbitals: "
                f"selected {active_idx.size}, ncas {ncas}"
            )

        dm_global = _fragment_density_to_parent_density(
            self.mol, self.fragment_refs
        )
        s = self.mol.intor_symmetric('int1e_ovlp')
        dm_es = self.es_orb.T.conj() @ s @ dm_global @ s @ self.es_orb
        dm_es = (dm_es + dm_es.T.conj()) * 0.5
        dm_es, trace_raw, trace_scale = _rescale_density_trace(
            dm_es, nelectron
        )

        all_idx = np.arange(self.nes)
        env_idx = np.setdiff1d(all_idx, active_idx)
        ncore = (nelectron - nelecas) // 2
        if 2*ncore + nelecas != nelectron:
            raise ValueError("embedded CAHF electron count is inconsistent")
        if ncore > env_idx.size:
            raise ValueError(
                "not enough environment orbitals to hold embedded CAHF core"
            )

        dm_env = dm_es[np.ix_(env_idx, env_idx)]
        occ_env, orb_env = np.linalg.eigh(dm_env)
        order = np.argsort(occ_env)[::-1]
        occ_env = occ_env[order]
        orb_env = orb_env[:, order]

        env_orb = np.eye(self.nes)[:, env_idx] @ orb_env
        active_orb = np.eye(self.nes)[:, active_idx]
        core_orb = env_orb[:, :ncore]
        vir_orb = env_orb[:, ncore:]
        mo_coeff = np.hstack((core_orb, active_orb, vir_orb))

        mo_occ = np.zeros(self.nes)
        mo_occ[:ncore] = 2
        mo_occ[ncore:ncore+ncas] = nelecas / ncas
        es_dm = (mo_coeff * mo_occ) @ mo_coeff.T.conj()
        es_dm = (es_dm + es_dm.T.conj()) * 0.5
        self.es_occ = mo_occ
        self.es_init_guess_info = {
            'kind': 'fragment_density',
            'active_idx': active_idx.copy(),
            'trace_raw': trace_raw,
            'trace_scale': trace_scale,
            'env_occ': occ_env.copy(),
            'core_occ_min': occ_env[:ncore].min() if ncore else None,
            'core_occ_max': occ_env[:ncore].max() if ncore else None,
            'virtual_occ_max': occ_env[ncore] if ncore < occ_env.size else None,
        }
        logger.info(
            self,
            'fragment-density embedded guess: raw trace %.12g, scale %.12g',
            trace_raw, trace_scale
        )
        logger.info(
            self,
            'fragment-density embedded guess: active %s -> %s',
            self.embedded_active_aolabels, active_idx.tolist()
        )
        if ncore < occ_env.size:
            logger.info(
                self,
                'fragment-density embedded guess: env core min %.12g, '
                'first virtual %.12g',
                occ_env[:ncore].min(), occ_env[ncore]
            )
        return mo_coeff, mo_occ, es_dm

    def density_fit(self, with_df=None):
        raise NotImplementedError(
            "density fitting for FDMET is not implemented yet"
        )
