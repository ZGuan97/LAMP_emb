from dataclasses import dataclass
from functools import reduce
from numbers import Integral

import numpy as np
from scipy.linalg import null_space, block_diag

from pyscf import gto, scf, ao2mo
from pyscf.lib import logger

from embed_sim import ssdmet, cahf, rdiis


@dataclass
class FragmentMolecule:
    """A fragment sub-molecule (impurity + one ligand group).

    Encapsulates the truncated molecule, AO-index mappings, and the
    SCF/CAHF result.  Provides classmethod ``from_parent_mol`` and
    instance methods ``run_scf``, ``build_bath``, ``orbitals_to_parent``.

    Parameters
    ----------
    mol : gto.Mole
        Fragment molecule.
    parent_atom_ids : list
        Mapping from fragment-internal atom index to parent atom ID.
    fragment_to_parent_idx : list
        Mapping from fragment AO index to parent AO index.
    fragment_imp_idx : list
        AO indices within the fragment belonging to impurity atoms.
    fragment_lig_idx : list
        AO indices within the fragment belonging to ligand atoms.
    charge : int
        Total charge of the fragment.
    mf : object, optional
        SCF/CAHF result.  ``None`` until ``run_scf`` is called.
    """
    mol: gto.Mole
    parent_atom_ids: list
    fragment_to_parent_idx: list
    fragment_imp_idx: list
    fragment_lig_idx: list
    charge: int
    mf: object = None

    @classmethod
    def from_parent_mol(cls, parent_mol, impurity_atoms, ligand_atoms,
                        charge=None, spin=None, verbose=None):
        """Build a fragment sub-molecule from parent-molecule atom IDs.

        Parameters
        ----------
        parent_mol : gto.Mole
            Parent molecule.
        impurity_atoms : list[int]
            Parent atom IDs for the impurity.
        ligand_atoms : list[int]
            Parent atom IDs for the ligand.
        charge : int, optional
            Total charge for the fragment molecule.
        spin : int, optional
            Spin.  Defaults to ``parent_mol.spin``.
        verbose : int, optional
            Verbosity.  Defaults to ``parent_mol.verbose``.

        Returns
        -------
        FragmentMolecule
        """
        imp_set = set(impurity_atoms)
        lig_set = set(ligand_atoms)
        overlap = imp_set & lig_set
        if overlap:
            raise ValueError(f"impurity and ligand atoms overlap: {sorted(overlap)}")

        atom_ids = sorted(imp_set | lig_set)
        if spin is None:
            spin = parent_mol.spin
        if verbose is None:
            verbose = parent_mol.verbose

        atom = [[parent_mol.atom_symbol(i), parent_mol.atom_coord(i)]
                for i in atom_ids]
        frag_mol = gto.M(atom=atom, basis=parent_mol._basis,
                         ecp=parent_mol._ecp, charge=charge,
                         spin=spin, unit='Bohr', cart=parent_mol.cart,
                         symmetry=False, verbose=verbose,
                         max_memory=parent_mol.max_memory)

        # Build fragment-to-parent AO index mapping.
        parent_aoslice = parent_mol.aoslice_by_atom()
        frag_aoslice = frag_mol.aoslice_by_atom()
        frag_to_parent_atom = dict(enumerate(atom_ids))

        fragment_to_parent_idx = []
        fragment_imp_idx = []
        fragment_lig_idx = []
        for frag_atom in range(frag_mol.natm):
            parent_atom = frag_to_parent_atom[frag_atom]
            frag_ao_start, frag_ao_end = frag_aoslice[frag_atom, 2], frag_aoslice[frag_atom, 3]
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

        return cls(mol=frag_mol,
                   parent_atom_ids=list(atom_ids),
                   fragment_to_parent_idx=fragment_to_parent_idx,
                   fragment_imp_idx=fragment_imp_idx,
                   fragment_lig_idx=fragment_lig_idx,
                   charge=charge)

    def run_scf(self, fragment_scf="rohf", verbose=3, **kwargs):
        """Run fragment SCF/CAHF, stores result in ``self.mf``.

        Parameters
        ----------
        fragment_scf : str
            ``"cahf"`` or ``"rohf"``.
        verbose : int
            Verbosity for the fragment SCF output.
        **kwargs
            Additional SCF options.
        """
        fragment_scf = fragment_scf.lower()
        if fragment_scf == "cahf":
            ncas = kwargs.pop('ncas', 5)
            nelecas = kwargs.pop('nelecas', 7)
            mf = cahf.CAHF(self.mol, ncas=ncas, nelecas=nelecas,
                           spin=kwargs.pop('cahf_spin', self.mol.spin)).x2c()
        elif fragment_scf == "cahf-soscf":
            ncas = kwargs.pop('ncas', 5)
            nelecas = kwargs.pop('nelecas', 7)
            spin = kwargs.pop('cahf_spin', self.mol.spin)
            avas_aolabels = kwargs.pop('avas_aolabels')
            avas_threshold = kwargs.pop('avas_threshold', 0.5)
            init_guess = kwargs.pop('init_guess', 'atom')
            pre_scf_max_cycle = kwargs.pop('pre_scf_max_cycle', 0)
            max_cycle = kwargs.pop('max_cycle', 200)
            conv_tol = kwargs.pop('conv_tol', 1e-9)
            level_shift = kwargs.pop('level_shift', 0)
            kwargs.pop('max_memory', None)
            # Consume RDIIS keys (SOSCF does not use RDIIS)
            if kwargs:
                rdiis.RDIIS.setup(kwargs, self.fragment_imp_idx, True)
                if kwargs:
                    raise TypeError(f"unknown fragment SCF options: {sorted(kwargs)}")
            self.mf = cahf.CAHF_SOSCF(
                self.mol, ncas=ncas, nelecas=nelecas, spin=spin,
                avas_aolabels=avas_aolabels, avas_threshold=avas_threshold,
                init_guess=init_guess, pre_scf_max_cycle=pre_scf_max_cycle,
                max_cycle=max_cycle, conv_tol=conv_tol, level_shift=level_shift,
                verbose=verbose,
            )
            return
        elif fragment_scf == "rohf":
            mf = scf.rohf.ROHF(self.mol).x2c()
        else:
            raise ValueError(f"unsupported fragment_scf {fragment_scf}")

        mf.max_memory = kwargs.pop('max_memory', self.mol.max_memory)
        mf.max_cycle = kwargs.pop('max_cycle', getattr(mf, 'max_cycle', 50))
        mf.level_shift = kwargs.pop('level_shift', getattr(mf, 'level_shift', 0))

        if 'rdiis_imp_idx' in kwargs:
            kwargs['rdiis_imp_idx'] = _resolve_indices(
                kwargs['rdiis_imp_idx'], 'rdiis_imp_idx',
                lambda labels: gto.mole._aolabels2baslst(self.mol, labels, base=0))
        mf.diis = rdiis.RDIIS.setup(kwargs, self.fragment_imp_idx,
                                    verbose < logger.INFO)

        if kwargs:
            raise TypeError(f"unknown fragment SCF options: {sorted(kwargs)}")
        mf.kernel()
        self.mf = mf

    def build_bath(self, threshold=1e-13):
        """Construct bath orbitals from the fragment density matrix.

        Follows the impurity-preserving Lowdin + embedded subspace NO
        path from ``ssdmet.build_embeded_subspace``.

        Parameters
        ----------
        threshold : float
            Occupation-number threshold for classifying environment
            natural orbitals as bath, frozen-occupied, or frozen-virtual.

        Returns
        -------
        bath_orb : ndarray
            Bath orbital coefficients in fragment AO basis.
        fo_orb : ndarray
            Frozen-occupied orbital coefficients in fragment AO basis.
        fv_orb : ndarray
            Frozen-virtual orbital coefficients in fragment AO basis.
        """
        if self.mf is None:
            raise RuntimeError("run_scf() must be called before build_bath()")

        dm = ssdmet.mf_or_cas_make_rdm1s(self.mf)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        imp_idx = list(self.fragment_imp_idx)
        env_idx = list(self.fragment_lig_idx)
        fragment_idx = imp_idx + env_idx
        if len(set(fragment_idx)) != len(fragment_idx):
            raise ValueError("impurity and ligand AO indices overlap")
        if max(fragment_idx) >= dm.shape[0] or min(fragment_idx) < 0:
            raise ValueError("fragment AO index is out of range")

        caolo, cloao = ssdmet.lowdin_orth(
            self.mol, imp_idx=imp_idx, preserve_imp=True)
        ldm = reduce(np.dot, (cloao, dm, cloao.conj().T))

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
        bath_idx = np.nonzero((occ_env >= threshold) & (occ_env <= 2 - threshold))[0]
        fo_idx = np.nonzero(occ_env > 2 - threshold)[0]

        # Reorder env orbitals: [bath, fo, fv]
        orb_env = np.hstack((orb_env[:, bath_idx], orb_env[:, fo_idx],
                             orb_env[:, fv_idx]))

        # Build embedded subspace density (imp + bath block)
        ldm_es = np.block(
            [[ldm_imp, ldm_imp_env @ orb_env[:, :nbath]],
             [orb_env[:, :nbath].T.conj() @ ldm_env_imp,
              orb_env[:, :nbath].T.conj() @ ldm_env @ orb_env[:, :nbath]]])
        es_occ, es_nat_orb = np.linalg.eigh(ldm_es)
        es_occ = es_occ[::-1]
        es_nat_orb = es_nat_orb[:, ::-1]

        # Full transformation in fragment Lowdin space, rearrange to AO order
        nfrag = ldm.shape[0]
        cloes = block_diag(np.eye(nimp), orb_env) @ block_diag(
            es_nat_orb, np.eye(nfo + nfv))
        rearrange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
        cloes = cloes[rearrange_idx, :]

        # Extract bath/fo/fv from environment part, convert to AO basis
        bath_orb = caolo[:, env_idx] @ cloes[nimp:, :nbath]
        fo_orb = caolo[:, env_idx] @ cloes[nimp:, nbath:nbath + nfo]
        fv_orb = caolo[:, env_idx] @ cloes[nimp:, nbath + nfo:]
        return bath_orb, fo_orb, fv_orb

    def orbitals_to_parent(self, frag_orb, nao):
        """Embed fragment orbital coefficients into the parent AO basis.

        Parameters
        ----------
        frag_orb : ndarray
            Orbital coefficients in fragment AO basis, shape
            ``(n_frag_ao, n_orb)``.
        nao : int
            Number of AOs in the parent molecule.

        Returns
        -------
        ndarray
            Orbital coefficients in parent AO basis, shape
            ``(nao, n_orb)``.
        """
        parent_orb = np.zeros((nao, frag_orb.shape[1]), dtype=frag_orb.dtype)
        for frag_ao, parent_ao in enumerate(self.fragment_to_parent_idx):
            parent_orb[parent_ao, :] = frag_orb[frag_ao, :]
        return parent_orb


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


def _resolve_indices(ao_idx, name, lookup):
    """Resolve AO labels or AO indices to integer indices via *lookup* callable."""
    if isinstance(ao_idx, str):
        indices = list(lookup(ao_idx))
        if len(indices) == 0:
            raise ValueError(f"{name} does not match any AO labels")
        return indices
    try:
        indices = list(ao_idx)
    except TypeError:
        raise TypeError(f"{name} must be AO labels or AO indices")
    if all(isinstance(idx, Integral) for idx in indices):
        return [int(idx) for idx in indices]
    indices = list(lookup(ao_idx))
    if len(indices) == 0:
        raise ValueError(f"{name} does not match any AO labels")
    return indices


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


def _fragment_density_to_parent_density(mol, fragment_mols):
    dm_parent = np.zeros((mol.nao, mol.nao))
    dm_imp_acc = np.zeros((mol.nao, mol.nao))
    imp_count = np.zeros((mol.nao, mol.nao))

    for frag in fragment_mols:
        dm = ssdmet.mf_or_cas_make_rdm1s(frag.mf)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        f2p = np.asarray(frag.fragment_to_parent_idx, dtype=int)
        frag_imp = np.asarray(frag.fragment_imp_idx, dtype=int)
        frag_lig = np.asarray(frag.fragment_lig_idx, dtype=int)
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

        self.ligand_atoms = None
        if ligand_atoms is not None:
            self.ligand_atoms = [
                _resolve_atom_ids(self.mol, lig, f'ligand {i}')
                for i, lig in enumerate(ligand_atoms)
            ]
            if ligand_charges is not None:
                self.ligand_charges = list(ligand_charges)
                if len(self.ligand_charges) != len(self.ligand_atoms):
                    raise ValueError("ligand_charges must have the same length as ligand_atoms")

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
        parent_idx = list(gto.mole._aolabels2baslst(self.mol, aolabels, base=0))
        embedded_idx = [
            self.impurity_ao_to_embedded_idx[idx]
            for idx in parent_idx
            if idx in self.impurity_ao_to_embedded_idx
        ]
        return np.asarray(embedded_idx, dtype=int) + base

    def make_fragment_mol(self, ligand_atoms, charge, spin=None,
                           verbose=None):
        return FragmentMolecule.from_parent_mol(
            self.mol, self._imp_atoms, ligand_atoms,
            charge=charge, spin=spin, verbose=verbose)

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

    def build(self, verbose=3):
        self.dump_flags()
        if self.ligand_atoms is None:
            raise ValueError("fragment bath construction requires ligand_atoms to be assigned")
        self.fragment_mols = []
        bath_blocks = []
        fo_blocks = []
        for ifrag, lig_atoms in enumerate(self.ligand_atoms):
            logger.info(self, 'build impurity-ligand fragment %d', ifrag)

            charge = self.imp_charge + self.ligand_charges[ifrag]
            frag = self.make_fragment_mol(lig_atoms, charge=charge, verbose=verbose)

            scf_options = dict(self.fragment_scf_options)
            frag.run_scf(fragment_scf=self.fragment_scf, verbose=verbose, **scf_options)
            self.fragment_mols.append(frag)

            bath_orb, fo_orb, _ = frag.build_bath(threshold=self.threshold)

            bath_blocks.append(frag.orbitals_to_parent(bath_orb, self.mol.nao))
            fo_blocks.append(frag.orbitals_to_parent(fo_orb, self.mol.nao))

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
        logger.info(self, 'energy from frozen occupied orbitals %s', self.fo_ene(e_nuc=False))
        logger.info(self, 'nuclear repulsion energy %s', self.mol.energy_nuc())
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

        es_mf = cahf.CAHF(mol, ncas=ncas, nelecas=nelecas, spin=cahf_spin).x2c()
        es_mf.max_memory = scf_options.pop('max_memory', self.max_mem)
        es_mf.max_cycle = scf_options.pop('max_cycle', 200)
        es_mf.conv_tol = scf_options.pop('conv_tol', 1e-9)
        es_mf.level_shift = scf_options.pop('level_shift', 5)
        es_mf.mo_energy = np.zeros((self.nes))

        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: np.eye(self.nes)
        es_mf._eri = ao2mo.restore(8, self.es_int2e, self.nes)

        scf_options['rdiis_imp_idx'] = _resolve_indices(
            scf_options['rdiis_imp_idx'], 'rdiis_imp_idx', self.search_impurity_ao_label)
        es_mf.diis = rdiis.RDIIS.setup(scf_options, scf_options['rdiis_imp_idx'],
                                       self.verbose < logger.INFO)

        newton = scf_options.pop('newton', False)
        # Discard SOSCF-specific keys (consumed by run_scf in fragment stage)
        scf_options.pop('avas_aolabels', None)
        scf_options.pop('avas_threshold', None)
        scf_options.pop('init_guess', None)
        scf_options.pop('pre_scf_max_cycle', None)

        if scf_options:
            raise TypeError(f"unknown embedded CAHF options: {sorted(scf_options)}")

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
            if newton:
                es_mf = es_mf.newton()
                es_mf.kernel()
            self.es_occ = es_mf.mo_occ
        return es_mf

    def _embedded_cahf_initial_guess(self, ncas, nelecas, nelectron):
        if self.embedded_init_guess != 'fragment_density':
            es_dm = np.diag(self.es_occ)

        dm_parent = _fragment_density_to_parent_density(
            self.mol, self.fragment_mols
        )
        s = self.mol.intor_symmetric('int1e_ovlp')
        dm_es = self.es_orb.T.conj() @ s @ dm_parent @ s @ self.es_orb
        dm_es = (dm_es + dm_es.T.conj()) * 0.5
        dm_es, trace_raw, trace_scale = _rescale_density_trace(
            dm_es, nelectron
        )
        logger.info(
            self,
            'fragment-density embedded guess: raw trace %.12g, scale %.12g',
            trace_raw, trace_scale
        )

        mo_coeff, mo_occ, es_dm = self._cahf_mo_from_density(
            dm_es, ncas, nelecas, nelectron
        )
        self.es_init_guess_info['kind'] = 'fragment_density'
        self.es_init_guess_info['trace_raw'] = trace_raw
        self.es_init_guess_info['trace_scale'] = trace_scale
        return mo_coeff, mo_occ, es_dm

    def _cahf_mo_from_density(self, dm_es, ncas, nelecas, nelectron):
        """Construct CAHF mo_coeff, mo_occ, es_dm from an embedded-space density.

        Diagonalizes the environment block of *dm_es* to get natural orbitals,
        then builds ``[core | active | virtual]`` ordering with CAHF fractional
        occupations.

        Parameters
        ----------
        dm_es : ndarray (nes, nes)
            Density matrix in the current embedded-space basis.
        ncas : int
            Number of active orbitals.
        nelecas : int
            Number of active electrons.
        nelectron : int
            Total embedded-space electrons.

        Returns
        -------
        mo_coeff, mo_occ, es_dm
        """
        if self.embedded_active_aolabels is None:
            raise ValueError("embedded_active_aolabels is required")
        active_idx = self.search_impurity_ao_label(
            self.embedded_active_aolabels
        )
        if active_idx.size != ncas:
            raise ValueError(
                "embedded active AO labels must select exactly ncas orbitals: "
                f"selected {active_idx.size}, ncas {ncas}"
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
            'active_idx': active_idx.copy(),
            'env_occ': occ_env.copy(),
            'core_occ_min': occ_env[:ncore].min() if ncore else None,
            'core_occ_max': occ_env[:ncore].max() if ncore else None,
            'virtual_occ_max': occ_env[ncore] if ncore < occ_env.size else None,
        }
        logger.info(
            self,
            'CAHF guess from density: active %s -> %s',
            self.embedded_active_aolabels, active_idx.tolist()
        )
        if ncore < occ_env.size:
            logger.info(
                self,
                'CAHF guess from density: env core min %.12g, '
                'first virtual %.12g',
                occ_env[:ncore].min(), occ_env[ncore]
            )
        return mo_coeff, mo_occ, es_dm

    def rebuild_from_embedded_density(self, threshold=None):
        """Re-select bath orbitals using the converged embedded CAHF density.

        Maps the embedded CAHF density back to parent AO Lowdin basis,
        decomposes the environment block into natural orbitals, and
        re-classifies them as bath, frozen-occupied, or frozen-virtual.
        Rebuilds ``es_orb``, ``es_int1e``, ``es_int2e``, and ``es_mf``.

        Parameters
        ----------
        threshold : float, optional
            Occupation-number threshold for classifying environment
            natural orbitals.  Defaults to ``self.threshold``.

        Returns
        -------
        es_mf : CAHF
            New embedded CAHF object (not converged).
        """
        if self.es_mf is None:
            raise RuntimeError("embedded CAHF must be run before rebuild")
        if threshold is None:
            threshold = self.threshold

        mo = self.es_mf.mo_coeff
        mo_occ = self.es_mf.mo_occ
        dm_es = (mo * mo_occ) @ mo.T.conj()
        dm_es = (dm_es + dm_es.T.conj()) * 0.5

        caolo, cloao = ssdmet.lowdin_orth(
            self.mol, imp_idx=self.imp_idx, preserve_imp=True
        )
        U = cloao @ self.es_orb
        ldm = U @ dm_es @ U.T.conj()
        # supplement with frozen-occupied density (FV has occ=0, no contribution)
        if self.fo_orb.shape[1] > 0:
            fo_ldm = cloao @ self.fo_orb
            ldm = ldm + fo_ldm @ fo_ldm.T.conj() * 2
        ldm = (ldm + ldm.T.conj()) * 0.5

        env_idx = [x for x in range(ldm.shape[0]) if x not in self.imp_idx]
        ldm_env = ldm[env_idx, :][:, env_idx]
        occ_env, orb_env = np.linalg.eigh(ldm_env)

        bath_idx = np.nonzero(
            (occ_env >= threshold) & (occ_env <= 2 - threshold))[0]
        fo_idx = np.nonzero(occ_env > 2 - threshold)[0]
        nbath = len(bath_idx)
        nfo_env = len(fo_idx)

        bath_orb = (caolo[:, env_idx] @ orb_env[:, bath_idx]
                    if nbath > 0 else np.zeros((self.mol.nao, 0)))
        fo_orb = (caolo[:, env_idx] @ orb_env[:, fo_idx]
                  if nfo_env > 0 else np.zeros((self.mol.nao, 0)))

        c_imp, bath_orb, self.fo_orb, self.fv_orb = _complete_global_orbitals(
            self.mol, self.imp_idx, bath_orb, fo_orb=fo_orb,
            svd_tol=threshold
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

        logger.info(self, 'rebuilding embedded space from CAHF density:')
        logger.info(self, '  impurity orbitals          %d',
                    len(self.imp_idx))
        logger.info(self, '  bath orbitals              %d', nbath)
        logger.info(self, '  appended FO orbitals       %d',
                    nappended_fo)
        if nfo_env > 0:
            logger.info(self,
                        '  (env natural occ range of FO: %.12g .. %.12g)',
                        occ_env[fo_idx].min(), occ_env[fo_idx].max())
        logger.info(self, '  kept FV orbitals           %d',
                    nkept_fv)
        logger.info(self, '  frozen occupied            %d',
                    self.nfo)
        logger.info(self, '  frozen virtual             %d',
                    self.nfv)
        logger.info(self, '  total embedded orbitals    %d',
                    self.nes)

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()

        # project converged Lowdin density into new embedded basis
        U_new = cloao @ self.es_orb
        dm_es_new = U_new.T.conj() @ ldm @ U_new
        dm_es_new = (dm_es_new + dm_es_new.T.conj()) * 0.5

        self.es_mf = self.CAHF(**self.fragment_scf_options)

        # override initial guess with projected converged density
        # diagonalize the FULL embedded density (cf. SSDMET build_embeded_subspace),
        # whose eigenvalues are strictly {2, f, 0}
        ncas = self.fragment_scf_options['ncas']
        nelecas = self.fragment_scf_options['nelecas']
        nelectron = self.mol.nelectron - 2 * self.nfo
        ncore = (nelectron - nelecas) // 2

        es_occ, es_nat_orb = np.linalg.eigh(dm_es_new)
        es_occ = es_occ[::-1]
        es_nat_orb = es_nat_orb[:, ::-1]

        mo_occ = np.zeros(self.nes)
        mo_occ[:ncore] = 2
        mo_occ[ncore:ncore + ncas] = nelecas / ncas
        es_dm = (es_nat_orb * mo_occ) @ es_nat_orb.T.conj()
        es_dm = (es_dm + es_dm.T.conj()) * 0.5

        self.es_init_guess_info['kind'] = 'rebuild_projected'
        self.es_init_guess_info['nat_occ'] = es_occ
        self.es_dm = es_dm
        self.es_mf.mo_coeff = es_nat_orb
        self.es_mf.mo_occ = mo_occ
        self.es_mf.get_init_guess = lambda *args, **kwargs: self.es_dm

        logger.info(self, 'energy from frozen occupied orbitals %s',
                    self.fo_ene(e_nuc=False))
        logger.info(self, 'nuclear repulsion energy %s',
                    self.mol.energy_nuc())
        return self.es_mf

    def density_fit(self, with_df=None):
        raise NotImplementedError(
            "density fitting for FDMET is not implemented yet"
        )
