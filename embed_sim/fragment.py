from dataclasses import dataclass
from functools import reduce
from numbers import Integral

import numpy as np
from scipy.linalg import null_space, block_diag

from pyscf import gto, scf, ao2mo
from pyscf.lib import logger

from embed_sim import ssdmet, cahf, rdiis


@dataclass
class LigandReference:
    """Container for one impurity-ligand fragment reference calculation."""
    mol: gto.mole.Mole
    mf: object
    fragment_to_parent_idx: list
    fragment_imp_idx: list
    fragment_lig_idx: list


def _resolve_atom_ids(mol, atoms, name):
    """Resolve user atom specification to a sorted list of atom IDs.

    Accepts a single int, a single element symbol string, or a list of
    ints/symbols.  Returns a sorted, validated list of unique atom IDs.
    """
    natm = mol.natm
    if isinstance(atoms, (int, Integral)):
        ids = [int(atoms)]
    elif isinstance(atoms, str):
        ids = [i for i in range(natm) if mol.atom_symbol(i) == atoms]
        if len(ids) == 0:
            raise ValueError(f"{name}: no atom with symbol '{atoms}' in molecule")
    else:
        try:
            atoms_iter = list(atoms)
        except TypeError:
            raise TypeError(f"{name} must be an atom ID, element symbol, or list thereof")
        ids = []
        for a in atoms_iter:
            if isinstance(a, (int, Integral)):
                ids.append(int(a))
            elif isinstance(a, str):
                matched = [i for i in range(natm) if mol.atom_symbol(i) == a]
                if len(matched) == 0:
                    raise ValueError(f"{name}: no atom with symbol '{a}' in molecule")
                ids.extend(matched)
            else:
                raise TypeError(f"{name}: each item must be an int or element symbol, " f"got {type(a)}")

    for atom_id in ids:
        if atom_id < 0 or atom_id >= natm:
            raise ValueError(
                f"{name}: atom ID {atom_id} out of range [0, {natm})"
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate atom IDs")
    if len(ids) == 0:
        raise ValueError(f"{name} must not be empty")
    return sorted(ids)


def _atom_ids_to_ao_indices(mol, atom_ids):
    """Return all AO indices on the given atoms."""
    aoslice = mol.aoslice_by_atom()
    ao_indices = []
    for atom_id in atom_ids:
        ao_start = aoslice[atom_id, 2]
        ao_end = aoslice[atom_id, 3]
        ao_indices.extend(range(ao_start, ao_end))
    return sorted(ao_indices)


def _resolve_local_ao_indices(mol, ao_idx, name):
    if isinstance(ao_idx, str):
        indices = [int(x) for x in gto.mole._aolabels2baslst(
            mol, ao_idx, base=0)]
        if len(indices) == 0:
            raise ValueError(f"{name} must not be empty")
        return indices
    try:
        indices = list(ao_idx)
    except TypeError:
        raise TypeError(f"{name} must be AO labels or AO indices")
    if all(isinstance(idx, Integral) for idx in indices):
        return [int(idx) for idx in indices]
    indices = [int(x) for x in gto.mole._aolabels2baslst(
        mol, ao_idx, base=0)]
    if len(indices) == 0:
        raise ValueError(f"{name} must not be empty")
    return indices


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


def _make_fragment_mol(parent_mol, impurity_atoms, ligand_atoms, charge=None,
                       spin=None, verbose=None):
    """Build a fragment sub-molecule from atom IDs.

    Parameters
    ----------
    parent_mol : gto.Mole
        Parent molecule.
    impurity_atoms : list[int]
        Parent atom IDs for the impurity.
    ligand_atoms : list[int]
        Parent atom IDs for the ligand.
    """
    imp_set = set(impurity_atoms)
    lig_set = set(ligand_atoms)
    overlap = imp_set & lig_set
    if overlap:
        raise ValueError(
            f"impurity and ligand atoms overlap: {sorted(overlap)}"
        )

    atom_ids = sorted(imp_set | lig_set)
    if spin is None:
        spin = parent_mol.spin
    if verbose is None:
        verbose = parent_mol.verbose

    atom = [
        [parent_mol.atom_symbol(i), parent_mol.atom_coord(i)]
        for i in atom_ids
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

    # Build fragment-to-parent AO index mapping.
    parent_aoslice = parent_mol.aoslice_by_atom()
    frag_aoslice = frag_mol.aoslice_by_atom()
    # Map: fragment atom index -> parent atom ID
    frag_to_parent_atom = dict(enumerate(atom_ids))

    fragment_to_parent_idx = []
    fragment_imp_idx = []
    fragment_lig_idx = []
    for frag_atom in range(frag_mol.natm):
        parent_atom = frag_to_parent_atom[frag_atom]
        frag_ao_start = frag_aoslice[frag_atom, 2]
        frag_ao_end = frag_aoslice[frag_atom, 3]
        parent_ao_start = parent_aoslice[parent_atom, 2]
        nao_on_atom = frag_ao_end - frag_ao_start
        for k in range(nao_on_atom):
            frag_ao = frag_ao_start + k
            parent_ao = parent_ao_start + k
            fragment_to_parent_idx.append(parent_ao)
            if parent_atom in imp_set:
                fragment_imp_idx.append(frag_ao)
            else:
                fragment_lig_idx.append(frag_ao)

    return frag_mol, fragment_to_parent_idx, fragment_imp_idx, fragment_lig_idx


def run_fragment_scf(mol, impurity_atoms, ligand_atoms, charge,
                     fragment_scf="rohf", fragment_scf_verbose=3, **kwargs):
    """
    Run a low-level reference calculation for one impurity-ligand subsystem.

    Parameters
    ----------
    mol : gto.Mole
        Parent molecule.
    impurity_atoms : list[int]
        Parent atom IDs for the impurity.
    ligand_atoms : list[int]
        Parent atom IDs for the ligand.
    """
    spin = kwargs.pop('spin', None)
    frag_mol, fragment_to_parent_idx, fragment_imp_idx, fragment_lig_idx = \
        _make_fragment_mol(mol, impurity_atoms, ligand_atoms, charge=charge,
                           spin=spin, verbose=fragment_scf_verbose)

    fragment_scf = fragment_scf.lower()
    if fragment_scf == "cahf":
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

    diis = kwargs.pop('diis', 'diis')

    if isinstance(diis, str):
        if diis.lower() != 'rdiis':
            raise ValueError(f"unsupported fragment DIIS {diis}")
        rdiis_imp_idx = kwargs.pop('rdiis_imp_idx', fragment_imp_idx)
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

    if kwargs:
        raise TypeError(f"unknown fragment SCF options: {sorted(kwargs)}")
    mf.kernel()

    return LigandReference(
        mol=frag_mol,
        mf=mf,
        fragment_to_parent_idx=fragment_to_parent_idx,
        fragment_imp_idx=fragment_imp_idx,
        fragment_lig_idx=fragment_lig_idx,
    )


def bath_from_ligand_density(dm, fragment_imp_idx, fragment_lig_idx,
                             mol=None, overlap=None,
                             threshold=1e-13):
    """
    Construct bath orbitals from one fragment density matrix.

    Follows the same impurity-preserving Lowdin path and embedded subspace
    natural-orbital rotation as ``ssdmet.build_embeded_subspace``.  The
    fragment density is transformed to the Lowdin basis, the environment
    (ligand) block is diagonalized, and an embedded subspace density is
    built in the impurity + bath block.  After diagonalization of this
    embedded subspace density, the full transformation is assembled with
    ``block_diag`` and rearranged back to the fragment AO ordering.
    Bath, frozen-occupied and frozen-virtual orbitals are returned in the
    parent AO basis.
    """
    if dm.ndim == 3:
        dm = dm[0] + dm[1]

    fragment_idx = list(fragment_imp_idx) + list(fragment_lig_idx)
    if len(set(fragment_idx)) != len(fragment_idx):
        raise ValueError("impurity and ligand AO indices overlap")
    if max(fragment_idx) >= dm.shape[0] or min(fragment_idx) < 0:
        raise ValueError("fragment AO index is out of range")

    caolo, cloao = ssdmet.lowdin_orth(
        mol, ovlp=overlap, imp_idx=list(fragment_imp_idx),
        preserve_imp=True
    )
    ldm = reduce(np.dot, (cloao, dm, cloao.conj().T))

    # Environment = ligand in fragment space
    imp_idx = list(fragment_imp_idx)
    env_idx = list(fragment_lig_idx)
    nimp = len(imp_idx)

    ldm_imp = ldm[imp_idx, :][:, imp_idx]
    ldm_env = ldm[env_idx, :][:, env_idx]
    ldm_imp_env = ldm[imp_idx, :][:, env_idx]
    ldm_env_imp = ldm[env_idx, :][:, imp_idx]

    occ_env, orb_env = np.linalg.eigh(ldm_env)

    nbath = np.sum((occ_env >= threshold) & (occ_env <= 2 - threshold))
    nfo = np.sum(occ_env > 2 - threshold)
    nfv = np.sum(occ_env < threshold)

    fv_idx = np.nonzero(occ_env < threshold)[0]
    bath_idx = np.nonzero(
        (occ_env >= threshold) & (occ_env <= 2 - threshold)
    )[0]
    fo_idx = np.nonzero(occ_env > 2 - threshold)[0]

    # Reorder env orbitals: [bath, fo, fv]
    orb_env = np.hstack((orb_env[:, bath_idx], orb_env[:, fo_idx], orb_env[:, fv_idx]))

    # Build embedded subspace density (imp + bath block)
    ldm_es = np.block([
        [ldm_imp, ldm_imp_env @ orb_env[:, :nbath]],
        [orb_env[:, :nbath].T.conj() @ ldm_env_imp,
         orb_env[:, :nbath].T.conj() @ ldm_env @ orb_env[:, :nbath]]
    ])
    es_occ, es_nat_orb = np.linalg.eigh(ldm_es)
    es_occ = es_occ[::-1]
    es_nat_orb = es_nat_orb[:, ::-1]

    # Full transformation in fragment Lowdin space
    nfrag = ldm.shape[0]
    n_es = nimp + nbath
    cloes = block_diag(np.eye(nimp), orb_env) @ block_diag(
        es_nat_orb, np.eye(nfo + nfv)
    )

    # Rearrange from [imp, env] ordering back to fragment AO ordering
    rearrange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
    cloes = cloes[rearrange_idx, :]

    # Extract bath / fo / fv from environment part, convert to AO basis
    bath_orb = caolo[:, env_idx] @ cloes[nimp:, :nbath]
    fo_orb = caolo[:, env_idx] @ cloes[nimp:, nbath:nbath + nfo]
    fv_orb = caolo[:, env_idx] @ cloes[nimp:, nbath + nfo:]
    return bath_orb, fo_orb, fv_orb


def fragment_orbitals_to_parent_orbitals(frag_orb, fragment_to_parent_idx, nao):
    parent_orb = np.zeros((nao, frag_orb.shape[1]), dtype=frag_orb.dtype)
    for frag_ao, parent_ao in enumerate(fragment_to_parent_idx):
        parent_orb[parent_ao, :] = frag_orb[frag_ao, :]
    return parent_orb


def _complete_global_orbitals(mol, impurity_idx, bath_orb, fo_orb=None,
                              svd_tol=1e-10):
    caolo, cloao = ssdmet.lowdin_orth(
        mol, imp_idx=list(impurity_idx), preserve_imp=True
    )
    env_idx = [idx for idx in range(mol.nao) if idx not in impurity_idx]
    c_imp = caolo[:, impurity_idx]
    if fo_orb is None:
        fo_orb = np.zeros((mol.nao, 0))

    env_basis = caolo[:, env_idx]
    if fo_orb.shape[1] == 0:
        qfo = np.zeros((len(env_idx), 0))
    else:
        fo_coord = cloao @ fo_orb
        env_fo_coord = fo_coord[env_idx, :]
        qfo, rfo = np.linalg.qr(env_fo_coord)
        rank = np.sum(np.abs(np.diag(rfo)) > svd_tol)
        qfo = qfo[:, :rank]
        fo_orb = env_basis @ qfo

    if bath_orb.shape[1] == 0:
        qbath = np.zeros((len(env_idx), 0))
    else:
        bath_coord = cloao @ bath_orb
        env_bath_coord = bath_coord[env_idx, :]
        if qfo.shape[1] > 0:
            env_bath_coord = env_bath_coord - qfo @ (
                qfo.T.conj() @ env_bath_coord
            )
        qbath, rbath = np.linalg.qr(env_bath_coord)
        rank = np.sum(np.abs(np.diag(rbath)) > svd_tol)
        qbath = qbath[:, :rank]
        bath_orb = env_basis @ qbath

    occupied_env = np.hstack((qbath, qfo))
    if occupied_env.shape[1] == 0:
        qcomp = np.eye(len(env_idx))
    else:
        qcomp = null_space(occupied_env.T.conj(), rcond=svd_tol)
    c_fv = caolo[:, env_idx] @ qcomp
    return c_imp, bath_orb, fo_orb, c_fv


# def _aufbau_occ(nelectron, spin, norb):
#     nalpha = (nelectron + spin) // 2
#     nbeta = (nelectron - spin) // 2
#     if nalpha + nbeta != nelectron or nalpha < nbeta:
#         raise ValueError("inconsistent electron count and spin")
#     if nalpha > norb:
#         raise ValueError(
#             "fragment-DMET without frozen occupied orbitals assigns all "
#             "electrons to the impurity+bath space, but this space has fewer "
#             "orbitals than alpha electrons"
#         )
#     occ = np.zeros(norb)
#     occ[:nbeta] = 2
#     occ[nbeta:nalpha] = 1
#     return occ


def _fragment_density_to_parent_density(mol, ligand_refs):
    dm_parent = np.zeros((mol.nao, mol.nao))
    dm_imp_acc = np.zeros((mol.nao, mol.nao))
    imp_count = np.zeros((mol.nao, mol.nao))

    for lig_ref in ligand_refs:
        dm = ssdmet.mf_or_cas_make_rdm1s(lig_ref.mf)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        f2p = np.asarray(lig_ref.fragment_to_parent_idx, dtype=int)
        frag_imp = np.asarray(lig_ref.fragment_imp_idx, dtype=int)
        frag_lig = np.asarray(lig_ref.fragment_lig_idx, dtype=int)
        parent_imp = f2p[frag_imp]
        parent_lig = f2p[frag_lig]

        dm_parent[np.ix_(parent_lig, parent_lig)] += dm[
            np.ix_(frag_lig, frag_lig)
        ]
        dm_parent[np.ix_(parent_imp, parent_lig)] += dm[
            np.ix_(frag_imp, frag_lig)
        ]
        dm_parent[np.ix_(parent_lig, parent_imp)] += dm[
            np.ix_(frag_lig, frag_imp)
        ]
        dm_imp_acc[np.ix_(parent_imp, parent_imp)] += dm[
            np.ix_(frag_imp, frag_imp)
        ]
        imp_count[np.ix_(parent_imp, parent_imp)] += 1

    mask = imp_count > 0
    dm_parent[mask] = dm_imp_acc[mask] / imp_count[mask]
    return (dm_parent + dm_parent.T.conj()) * 0.5


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
    imp_atoms
        Impurity atom specification: atom ID (int), element symbol (str),
        or a list thereof.  All basis functions on the specified atoms are
        included in the impurity.
    imp_charge
        Charge assigned to the impurity fragment.
    ligand_atoms
        Ligand atom specification.  Each item is one ligand, given
        as a list of atom IDs.
    ligand_charges
        Charge assigned to each ligand.
    """
    def __init__(self, mol, title='untitled',
                 imp_atoms=None, imp_charge=None,
                 ligand_atoms=None, ligand_charges=None,
                 fragment_scf='cahf',
                 fragment_scf_options=None, threshold=1e-13,
                 keep_fv_orbitals=False,
                 embedded_init_guess='aufbau',
                 embedded_active_aolabels=None,
                 verbose=logger.INFO):

        self.imp_charge = imp_charge
        self.ligand_charges = None
        self.fragment_scf = fragment_scf
        self.fragment_scf_options = (
            {} if fragment_scf_options is None else dict(fragment_scf_options)
        )
        self.keep_fv_orbitals = keep_fv_orbitals
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
        self._imp_atoms = []
        self.imp_atoms = imp_atoms
        self._build_impurity_label_map()
        self.threshold = threshold

        self.fo_orb = None
        self.fv_orb = None
        self.es_orb = None
        self.es_occ = None

        self.nfo = None
        self.nfv = None
        self.nes = None
        self.nbath = None
        self.nappended_fo = None
        self.nkept_fv = None

        self.es_int1e = None
        self.es_int2e = None

        self.es_mf = None
        self.es_dm = None
        self.es_init_guess_info = None
        self.ligand_refs = None
        self.ligand_atoms = None
        if ligand_atoms is not None:
            self.ligand_atoms = [
                _resolve_atom_ids(self.mol, lig, f'ligand {i}')
                for i, lig in enumerate(ligand_atoms)
            ]
            if ligand_charges is not None:
                self.ligand_charges = list(ligand_charges)
                if len(self.ligand_charges) != len(self.ligand_atoms):
                    raise ValueError(
                        "ligand_charges must have the same length as ligand_atoms"
                    )

    @property
    def imp_atoms(self):
        """Atom IDs defining the impurity."""
        return self._imp_atoms

    @imp_atoms.setter
    def imp_atoms(self, value):
        atom_ids = _resolve_atom_ids(self.mol, value, 'imp_atoms')
        self._imp_atoms = atom_ids
        self._imp_idx = _atom_ids_to_ao_indices(self.mol, atom_ids)

    @property
    def imp_idx(self):
        """AO indices defining the impurity (derived from imp_atoms)."""
        return self._imp_idx

    @imp_idx.setter
    def imp_idx(self, value):
        # Redirect to imp_atoms setter when called with atom IDs
        self.imp_atoms = value

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
        log.info('impurity atoms = %s (%d AO indices)',
                 self._imp_atoms, len(self.imp_idx))
        log.info('impurity charge = %s', self.imp_charge)
        if self.ligand_atoms is None:
            log.info('ligand_atoms = None')
        else:
            log.info('number of ligands = %d', len(self.ligand_atoms))
            log.info('ligand atoms = %s', self.ligand_atoms)
            log.info('ligand charges = %s', self.ligand_charges)

    def build(self, fragment_scf_verbose=3):
        self.dump_flags()
        if self.ligand_atoms is None:
            raise ValueError(
                "fragment bath construction requires ligand_atoms to be "
                "assigned"
            )
        self.ligand_refs = []
        for ifrag, lig_atoms in enumerate(self.ligand_atoms):
            logger.info(self, 'build impurity-ligand fragment %d', ifrag)
            scf_options = dict(self.fragment_scf_options)
            charge = self.imp_charge + self.ligand_charges[ifrag]
            lig_ref = run_fragment_scf(
                self.mol, self._imp_atoms, lig_atoms,
                charge=charge,
                fragment_scf=self.fragment_scf,
                fragment_scf_verbose=fragment_scf_verbose,
                **scf_options,
            )
            self.ligand_refs.append(lig_ref)

        bath_blocks = []
        fo_blocks = []
        for lig_ref in self.ligand_refs:
            bath_orb, fo_orb, _ = bath_from_ligand_density(
                ssdmet.mf_or_cas_make_rdm1s(lig_ref.mf),
                lig_ref.fragment_imp_idx,
                lig_ref.fragment_lig_idx,
                mol=lig_ref.mol,
                threshold=self.threshold,
            )
            bath_blocks.append(fragment_orbitals_to_parent_orbitals(
                bath_orb, lig_ref.fragment_to_parent_idx, self.mol.nao
            ))
            fo_blocks.append(fragment_orbitals_to_parent_orbitals(
                fo_orb, lig_ref.fragment_to_parent_idx, self.mol.nao
            ))

        raw_bath = np.hstack(bath_blocks) if bath_blocks else np.zeros((self.mol.nao, 0))
        raw_fo = np.hstack(fo_blocks) if fo_blocks else np.zeros((self.mol.nao, 0))
        c_imp, bath_orb, self.fo_orb, self.fv_orb = _complete_global_orbitals(
            self.mol, self.imp_idx, raw_bath, fo_orb=raw_fo,
            svd_tol=self.threshold
        )
        nbath = bath_orb.shape[1]
        nappended_fo = self.fo_orb.shape[1]
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
        self.nbath = nbath
        self.nappended_fo = nappended_fo
        self.nkept_fv = nkept_fv
        # self.es_occ = _aufbau_occ(
        #     self.mol.nelectron - 2*self.nfo, self.mol.spin, self.nes
        # )

        logger.info(self, 'number of impurity orbitals %d', len(self.imp_idx))
        logger.info(self, 'number of bath orbitals %d', nbath)
        logger.info(self, 'number of occupied environment orbitals appended to embedded space %d',
                    nappended_fo)
        logger.info(self, 'number of frozen virtual orbitals kept in embedded space %d',
                    nkept_fv)
        logger.info(self, 'number of frozen occupied orbitals %d', self.nfo)
        logger.info(self, 'number of frozen virtual orbitals %d', self.nfv)

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        self.es_mf = self.CAHF(**self.fragment_scf_options)
        logger.info(self, 'energy from frozen occupied orbitals %s',
                    self.fo_ene(e_nuc=False))
        logger.info(self, 'nuclear repulsion energy %s',
                    self.mol.energy_nuc())
        return self.es_mf

    def CAHF(self, run_mf=False, **scf_options):

        mol = gto.M()
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelectron = self.mol.nelectron - 2*self.nfo
        mol.spin = scf_options.pop('spin', self.mol.spin)

        ncas = scf_options.pop('ncas')
        nelecas = scf_options.pop('nelecas')
        cahf_spin = scf_options.pop('cahf_spin', mol.spin)

        es_mf = cahf.CAHF(mol, ncas=ncas, nelecas=nelecas,
                          spin=cahf_spin).x2c()
        es_mf.max_memory = scf_options.pop('max_memory', self.max_mem)
        es_mf.max_cycle = scf_options.pop(
            'max_cycle', getattr(es_mf, 'max_cycle', 200)
        )
        es_mf.level_shift = scf_options.pop(
            'level_shift', getattr(es_mf, 'level_shift', 0)
        )
        es_mf.mo_energy = np.zeros((self.nes))

        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: np.eye(self.nes)
        es_mf._eri = ao2mo.restore(8, self.es_int2e, self.nes)

        diis = scf_options.pop('diis', 'diis')
        if isinstance(diis, str):
            if diis.lower() != 'rdiis':
                raise ValueError(f"unsupported embedded DIIS {diis}")
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

        dm_parent = _fragment_density_to_parent_density(
            self.mol, self.ligand_refs
        )
        s = self.mol.intor_symmetric('int1e_ovlp')
        dm_es = self.es_orb.T.conj() @ s @ dm_parent @ s @ self.es_orb
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
