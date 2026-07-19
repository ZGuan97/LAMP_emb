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
    mf : object, optional
        SCF/CAHF result.  ``None`` until ``run_scf`` is called.
    """
    mol: gto.Mole
    parent_atom_ids: list
    fragment_to_parent_idx: list
    fragment_imp_idx: list
    fragment_lig_idx: list
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
                         symmetry=False, verbose=verbose)

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
                   fragment_lig_idx=fragment_lig_idx)

    def run_scf(self, verbose=3, scf_runner=None, **kwargs):
        """Run the fragment reference calculation and store it in ``self.mf``.

        Parameters
        ----------
        verbose : int
            Verbosity for the fragment SCF output.
        scf_runner : callable, optional
            Future extension point for an alternative fragment reference.
            It receives this ``FragmentMolecule`` as its first argument and
            must return a converged mean-field object.
        **kwargs
            Additional SCF options.
        """
        if scf_runner is not None:
            mf = scf_runner(self, verbose=verbose, **kwargs)
            if mf is None:
                raise RuntimeError("fragment scf_runner must return a mean-field object")
            self.mf = mf
            return self.mf

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

        # These options are reserved for the embedded CAHF calculation.
        for key in ('spin', 'diis', 'rdiis_imp_idx', 'rdiis_prop',
                    'rdiis_power', 'rdiis_kernel', 'rdiis_mute',
                    'rdiis_ent_conv_tol', 'rdiis_space', 'newton'):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"unknown fragment SCF options: {sorted(kwargs)}")

        self.mf = cahf.CAHF_SOSCF(
            self.mol, ncas=ncas, nelecas=nelecas, spin=spin,
            avas_aolabels=avas_aolabels, avas_threshold=avas_threshold,
            init_guess=init_guess, pre_scf_max_cycle=pre_scf_max_cycle,
            max_cycle=max_cycle, conv_tol=conv_tol, level_shift=level_shift,
            verbose=verbose,
        )
        return self.mf

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

        # ``cloes`` columns are [embedded space | FO | FV].  The embedded
        # space has nimp + nbath columns, so FO/FV must be offset by the full
        # embedded-space size rather than by nbath alone.
        bath_orb = caolo[:, env_idx] @ cloes[nimp:, :nbath]
        fo_start = nimp + nbath
        fv_start = fo_start + nfo
        fo_orb = caolo[:, env_idx] @ cloes[nimp:, fo_start:fv_start]
        fv_orb = caolo[:, env_idx] @ cloes[nimp:, fv_start:]
        if bath_orb.shape[1] + fo_orb.shape[1] + fv_orb.shape[1] != len(env_idx):
            raise RuntimeError('fragment bath/FO/FV orbitals do not span the environment')
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


def complete_global_orbitals(mol, impurity_idx, bath_orb, fo_orb,
                              svd_tol=1e-10):
    caolo, cloao = ssdmet.lowdin_orth(
        mol, imp_idx=list(impurity_idx), preserve_imp=True
    )
    env_idx = [idx for idx in range(mol.nao) if idx not in impurity_idx]
    c_imp = caolo[:, impurity_idx]

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
    return dm_parent


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
                 fragment_scf_options=None, fragment_scf_runner=None,
                 threshold=1e-13,
                 keep_fv_orbitals=False,
                 embedded_init_guess='aufbau',
                 embedded_active_aolabels=None,
                 verbose=logger.INFO):

        self.imp_charge = imp_charge
        self.ligand_charges = None
        self.fragment_scf_options = (
            {} if fragment_scf_options is None else dict(fragment_scf_options)
        )
        self.fragment_scf_runner = fragment_scf_runner
        self.keep_fv_orbitals = keep_fv_orbitals
        self.embedded_init_guess = embedded_init_guess
        self.embedded_active_aolabels = embedded_active_aolabels

        self.mol = mol
        self.title = title
        self.verbose = verbose
        self.mf_or_cas = scf.rohf.ROHF(self.mol).x2c()
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

    def run_fragment_loop(self, verbose):
        """Run fragment SCF and bath construction for all ligands.

        Returns
        -------
        fragment_mols : list[FragmentMolecule]
        bath_blocks : list[ndarray]
            Bath orbital coefficients in parent AO basis.
        fo_blocks : list[ndarray]
            Frozen-occupied orbital coefficients in parent AO basis.
        """
        fragment_mols = []
        bath_blocks = []
        fo_blocks = []
        for ifrag, lig_atoms in enumerate(self.ligand_atoms):
            logger.info(self, 'build impurity-ligand fragment %d', ifrag)
            charge = self.imp_charge + self.ligand_charges[ifrag]
            frag = self.make_fragment_mol(lig_atoms, charge=charge, verbose=verbose)
            scf_options = dict(self.fragment_scf_options)
            frag.run_scf(verbose=verbose, scf_runner=self.fragment_scf_runner, **scf_options)
            fragment_mols.append(frag)
            bath_orb, fo_orb, _ = frag.build_bath(threshold=self.threshold)
            bath_blocks.append(frag.orbitals_to_parent(bath_orb, self.mol.nao))
            fo_blocks.append(frag.orbitals_to_parent(fo_orb, self.mol.nao))
        return fragment_mols, bath_blocks, fo_blocks

    def finalize_embedded_space(self, c_imp, bath_orb, fo_orb, fv_orb):
        """Assemble es_orb and set orbital count attributes.

        Handles FO/FV appending into the embedded space, sets
        ``es_orb``, ``nfo``, ``nfv``, ``nes``, ``nbath``,
        ``nappended_fo``, ``nkept_fv``.
        """
        nbath = bath_orb.shape[1]
        nappended_fo = fo_orb.shape[1]
        nkept_fv = fv_orb.shape[1] if self.keep_fv_orbitals else 0

        es_env_blocks = [bath_orb]
        if fo_orb.shape[1] > 0:
            es_env_blocks.append(fo_orb)
            fo_orb = np.zeros((self.mol.nao, 0))
        if self.keep_fv_orbitals and fv_orb.shape[1] > 0:
            es_env_blocks.append(fv_orb)
            fv_orb = np.zeros((self.mol.nao, 0))

        self.es_orb = np.hstack((c_imp, np.hstack(es_env_blocks)))
        self.fo_orb = fo_orb
        self.fv_orb = fv_orb
        self.nfo = self.fo_orb.shape[1]
        self.nfv = self.fv_orb.shape[1]
        self.nes = self.es_orb.shape[1]
        self.nbath = nbath
        self.nappended_fo = nappended_fo
        self.nkept_fv = nkept_fv

        logger.info(self, 'number of impurity orbitals %d',
                    len(self.imp_idx))
        logger.info(self, 'number of bath orbitals %d', nbath)
        logger.info(self,
                    'number of occupied environment orbitals appended '
                    'to embedded space %d', nappended_fo)
        logger.info(self,
                    'number of frozen virtual orbitals kept in '
                    'embedded space %d', nkept_fv)
        logger.info(self, 'number of frozen occupied orbitals %d',
                    self.nfo)
        logger.info(self, 'number of frozen virtual orbitals %d',
                    self.nfv)

    def build_embedded_hamiltonian(self):
        """Compute embedded-space one- and two-electron integrals.

        Sets ``es_int1e`` and ``es_int2e``.
        """
        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        logger.info(self,
                    'energy from frozen occupied orbitals %s',
                    self.fo_ene(e_nuc=False))
        logger.info(self, 'nuclear repulsion energy %s',
                    self.mol.energy_nuc())

    def set_embedded_mf(self):
        """Construct the embedded CAHF mean-field object.

        Requires ``es_int1e``, ``es_int2e``, ``nes``, ``nfo`` to be set.
        Initial guess is determined by ``_embedded_cahf_initial_guess``,
        which selects fragment-density or rebuild-projected path
        automatically.

        Returns
        -------
        es_mf : CAHF
        """
        self.es_mf = self.CAHF(**self.fragment_scf_options)
        return self.es_mf

    def run_embedded_scf(self):
        """Run the embedded SCF/CAHF to convergence.

        Dispatches ``kernel()`` appropriately:
        - Newton / SOSCF (default): uses ``mo_coeff`` / ``mo_occ`` from ``es_mf``
        - Standard CAHF (``newton=False``): uses ``es_dm`` as density-matrix guess

        Stores converged ``es_mf`` (Newton wraps the object) and
        ``es_occ``.
        """
        if getattr(self.es_mf, '_newton', False):
            self.es_mf = self.es_mf.newton()
            self.es_mf.kernel(
                mo_coeff=self.es_mf.mo_coeff,
                mo_occ=self.es_mf.mo_occ)
        else:
            self.es_mf.kernel(self.es_dm)
        if not self.es_mf.converged:
            raise RuntimeError("embedded CAHF did not converge; refusing to rebuild from its density")
        self.es_occ = self.es_mf.mo_occ

    def do_rebuild(self, threshold=None):
        """Rebuild embedded space from converged CAHF density.

        Works entirely within the first embedded space: diagonalizes the
        environment block of the converged embedded density, re-classifies
        bath / frozen-occupied orbitals, and projects the converged density
        into the new embedded basis.

        After this method, call ``build_embedded_hamiltonian()`` then
        ``set_embedded_mf()`` to construct a new embedded CAHF with the
        projected density as initial guess.

        Parameters
        ----------
        threshold : float, optional
            Defaults to ``self.threshold``.
        """
        if self.es_mf is None:
            raise RuntimeError(
                "embedded CAHF must be run before rebuild")
        if threshold is None:
            threshold = self.threshold

        nimp = len(self.imp_idx)

        # Converged density in the first embedded basis
        mo = self.es_mf.mo_coeff
        mo_occ = self.es_mf.mo_occ
        dm_es = (mo * mo_occ) @ mo.T.conj()

        # Diagonalize environment block to re-classify bath / FO
        env_idx = list(range(nimp, self.nes))
        dm_env = dm_es[np.ix_(env_idx, env_idx)]
        occ_env, orb_env = np.linalg.eigh(dm_env)

        bath_idx = np.nonzero(
            (occ_env >= threshold) & (occ_env <= 2 - threshold))[0]
        fo_idx = np.nonzero(occ_env > 2 - threshold)[0]
        nbath_new = len(bath_idx)

        if len(fo_idx) > 0:
            logger.info(self,
                        '  (env natural occ range of FO: '
                        '%.12g .. %.12g)',
                        occ_env[fo_idx].min(), occ_env[fo_idx].max())

        # Build transformation within the first embedded space:
        # new es_orb = old es_orb @ T,  T = block_diag(I_imp, bath)
        es_orb_old = self.es_orb
        bath_orb_env = orb_env[:, bath_idx]
        fo_orb_env = orb_env[:, fo_idx]
        T = block_diag(np.eye(nimp), bath_orb_env)
        self.es_orb = es_orb_old @ T

        # FO orbitals become external frozen (in AO basis)
        nfo_new = len(fo_idx)
        if nfo_new > 0:
            env_part = es_orb_old[:, env_idx]
            fo_orb_ao = env_part @ fo_orb_env
            self.fo_orb = np.hstack((self.fo_orb, fo_orb_ao))
        self.fv_orb = np.zeros((self.mol.nao, 0))
        self.nfo += nfo_new
        self.nfv = 0
        self.nbath = nbath_new
        self.nappended_fo = 0
        self.nkept_fv = 0
        self.nes = self.es_orb.shape[1]

        logger.info(self, 'number of impurity orbitals %d', nimp)
        logger.info(self, 'number of bath orbitals %d', nbath_new)
        logger.info(self, 'number of frozen occupied orbitals %d', self.nfo)
        logger.info(self, 'rebuild: total embedded orbitals %d', self.nes)

        # Project converged non-FO density into new embedded basis
        dm_es_new = T.T.conj() @ dm_es @ T

        # Diagonalize for CAHF MO structure
        ncas = self.fragment_scf_options['ncas']
        nelecas = self.fragment_scf_options['nelecas']
        nelectron = self.mol.nelectron - 2 * self.nfo
        ncore = (nelectron - nelecas) // 2

        es_occ, es_nat_orb = np.linalg.eigh(dm_es_new)
        es_occ = es_occ[::-1]
        es_nat_orb = es_nat_orb[:, ::-1]

        mo_occ_new = np.zeros(self.nes)
        mo_occ_new[:ncore] = 2
        mo_occ_new[ncore:ncore + ncas] = nelecas / ncas
        es_dm = (es_nat_orb * mo_occ_new) @ es_nat_orb.T.conj()

        # Store projected density for set_embedded_mf()
        self._rebuild_data = {
            'es_dm': es_dm,
            'mo_coeff': es_nat_orb,
            'mo_occ': mo_occ_new,
            'es_occ': es_occ,
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
        return FragmentMolecule.from_parent_mol(self.mol, self._imp_atoms, ligand_atoms, charge=charge, spin=spin, verbose=verbose)

    def dump_flags(self):
        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', self.__class__)
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

    def validate_atom_partition(self):
        """Validate that impurity and ligands form a complete partition.

        Checks:
        - ``ligand_atoms`` is not None
        - impurity ∩ ligand = ∅
        - impurity ∪ ligand = all parent atoms

        Raises
        ------
        ValueError
            If the partition is invalid.
        """
        all_lig = set()
        for lig in self.ligand_atoms:
            all_lig.update(lig)
        imp_set = set(self._imp_atoms)
        overlap = imp_set & all_lig
        if overlap:
            raise ValueError(
                f"impurity and ligand atoms overlap: "
                f"{sorted(overlap)}")
        all_atoms = set(range(self.mol.natm))
        covered = imp_set | all_lig
        if covered != all_atoms:
            missing = sorted(all_atoms - covered)
            extra = sorted(covered - all_atoms)
            parts = []
            if missing:
                parts.append(f"missing atoms: {missing}")
            if extra:
                parts.append(f"invalid atoms: {extra}")
            raise ValueError(
                "impurity + ligand atoms do not cover parent "
                "molecule: " + "; ".join(parts))

    def build(self, verbose=3):
        """Full fragment-DMET build pipeline.

        Executes the complete workflow:

        1. Fragment SCF + bath construction (per ligand)
        2. Global bath merge + embedded space assembly
        3. Embedded Hamiltonian construction
        4. First embedded CAHF run
        5. Rebuild embedded space from converged density
        6. Second embedded CAHF run (final)

        Parameters
        ----------
        verbose : int
            Verbosity for fragment SCF.

        Returns
        -------
        es_mf : CAHF
            Converged embedded CAHF object.
        """
        self.dump_flags()
        self.validate_atom_partition()

        # Phase 1: Fragment loop
        logger.info(self, '\n%s\nFDMET stage 1/4: fragment CAHF references and bath construction\n%s',
                    '=' * 78, '=' * 78)
        self.fragment_mols, bath_blocks, fo_blocks = self.run_fragment_loop(verbose)

        # Phase 2: Global merge
        logger.info(self, '\n%s\nFDMET stage 2/4: merge fragment bath/FO orbitals into the global space\n%s',
                    '=' * 78, '=' * 78)
        raw_bath = np.hstack(bath_blocks)
        raw_fo = np.hstack(fo_blocks)
        c_imp, bath_orb, fo_orb, fv_orb = \
            complete_global_orbitals(
                self.mol, self.imp_idx, raw_bath,
                raw_fo, svd_tol=self.threshold)
        self.finalize_embedded_space(
            c_imp, bath_orb, fo_orb, fv_orb)

        # Phase 3: Embedded Hamiltonian + first CAHF
        logger.info(self, '\n%s\nFDMET stage 3/4: first embedded CAHF\n%s',
                    '=' * 78, '=' * 78)
        self.build_embedded_hamiltonian()
        self.set_embedded_mf()
        self.run_embedded_scf()

        # Phase 4: Rebuild from converged density + second CAHF
        logger.info(self, '\n%s\nFDMET stage 4/4: rebuild and final embedded CAHF\n%s',
                    '=' * 78, '=' * 78)
        self.do_rebuild()
        self.build_embedded_hamiltonian()
        self.set_embedded_mf()
        self.run_embedded_scf()

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

        # The embedded space has a fixed active-orbital guess and no AO labels
        # for AVAS.  Use CAHF's second-order Newton solver by default instead
        # of the molecular CAHF_SOSCF wrapper used for fragment-local SCF.
        newton = scf_options.pop('newton', True)
        es_mf._newton = newton
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
            self.run_embedded_scf()
        return es_mf

    def _embedded_cahf_initial_guess(self, ncas, nelecas, nelectron):
        # Use rebuild projected density if available
        rebuild_data = getattr(self, '_rebuild_data', None)
        if rebuild_data is not None:
            del self._rebuild_data
            mo_coeff = rebuild_data['mo_coeff']
            mo_occ = rebuild_data['mo_occ']
            es_dm = rebuild_data['es_dm']
            self.es_init_guess_info = {
                'kind': 'rebuild_projected',
                'nat_occ': rebuild_data['es_occ'],
            }
            return mo_coeff, mo_occ, es_dm

        dm_parent = _fragment_density_to_parent_density(
            self.mol, self.fragment_mols
        )
        s = self.mol.intor_symmetric('int1e_ovlp')
        dm_es = self.es_orb.T.conj() @ s @ dm_parent @ s @ self.es_orb

        # Diagnose whether the stitched fragment density has the correct
        # electron number and whether its occupied weight is represented by
        # the global ES/FO/FV partition.  These orbitals together span the
        # parent AO space, so their projected traces must sum to Tr(S D).
        c_full = np.hstack((self.es_orb, self.fo_orb, self.fv_orb))
        dm_full = c_full.T.conj() @ s @ dm_parent @ s @ c_full
        n_parent = np.einsum('ij,ji->', s, dm_parent).real
        n_es = np.trace(dm_es).real
        n_fo = np.trace(dm_full[self.nes:self.nes + self.nfo,
                                 self.nes:self.nes + self.nfo]).real
        n_fv = np.trace(dm_full[self.nes + self.nfo:,
                                 self.nes + self.nfo:]).real
        metric_error = np.linalg.norm(c_full.T.conj() @ s @ c_full - np.eye(self.mol.nao))
        logger.info(
            self,
            'fragment-density electron diagnostic: Tr(SD) %.12g; '
            'ES %.12g; FO %.12g; FV %.12g; partition total %.12g; '
            'metric error %.3g',
            n_parent, n_es, n_fo, n_fv, n_es + n_fo + n_fv, metric_error)

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
        self.es_init_guess_info['n_parent'] = n_parent
        self.es_init_guess_info['n_es'] = n_es
        self.es_init_guess_info['n_fo'] = n_fo
        self.es_init_guess_info['n_fv'] = n_fv
        self.es_init_guess_info['metric_error'] = metric_error
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

        Convenience wrapper around :meth:`do_rebuild`,
        :meth:`build_embedded_hamiltonian`, and :meth:`set_embedded_mf`.
        Sets up a new embedded CAHF with the projected density as initial
        guess but does **not** run it.
        Call ``self.es_mf.kernel()`` afterwards.

        Parameters
        ----------
        threshold : float, optional
            Occupation-number threshold.  Defaults to ``self.threshold``.

        Returns
        -------
        es_mf : CAHF
            New embedded CAHF object (not converged).
        """
        self.do_rebuild(threshold)
        self.build_embedded_hamiltonian()
        self.set_embedded_mf()
        return self.es_mf

    def density_fit(self, with_df=None):
        raise NotImplementedError(
            "density fitting for FDMET is not implemented yet"
        )
