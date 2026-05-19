from dataclasses import dataclass

from pyscf import gto, scf
from pyscf.lib import logger

from embed_sim import ssdmet


@dataclass
class FragmentReference:
    """Container for one impurity-ligand fragment reference calculation."""
    mol: gto.mole.Mole
    mf: object
    atom_ids: list
    global_idx: list
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
    if all(isinstance(idx, int) for idx in indices):
        return indices
    return _parse_aolabels(mol, ao_idx, name)


def _make_fragment_mol(parent_mol, impurity_idx, fragment_idx, charge=None,
                       spin=None, verbose=None):
    global_idx = list(impurity_idx) + list(fragment_idx)
    if len(set(global_idx)) != len(global_idx):
        raise ValueError("impurity and ligand fragment AO indices overlap")

    atom_ids = _atom_ids_from_ao_indices(parent_mol, global_idx)
    if spin is None:
        spin = parent_mol.spin
    if charge is None:
        raise ValueError("fragment molecule charge must be assigned explicitly")
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

    imp_set = set(impurity_idx)
    frag_set = set(fragment_idx)
    old_atom_by_new = dict(enumerate(atom_ids))
    parent_ao_by_atom = {atom_id: [] for atom_id in atom_ids}
    for parent_ao, label in enumerate(parent_mol.ao_labels(fmt=False)):
        atom_id = label[0]
        if atom_id in parent_ao_by_atom:
            parent_ao_by_atom[atom_id].append(parent_ao)
    local_count_by_atom = {atom_id: 0 for atom_id in atom_ids}
    local_imp_idx = []
    local_fragment_idx = []
    for local_ao, label in enumerate(frag_mol.ao_labels(fmt=False)):
        old_atom = old_atom_by_new[label[0]]
        offset = local_count_by_atom[old_atom]
        parent_ao = parent_ao_by_atom[old_atom][offset]
        local_count_by_atom[old_atom] += 1
        if parent_ao in imp_set:
            local_imp_idx.append(local_ao)
        elif parent_ao in frag_set:
            local_fragment_idx.append(local_ao)

    return frag_mol, atom_ids, global_idx, local_imp_idx, local_fragment_idx


def build_fragment_baths(mol, impurity_idx, fragment_idx, fragment_scf="rohf",
                         threshold=1e-13):
    """
    Build DMET bath orbitals from fragment-local reference densities.

    This is the public functional entry point reserved for the next
    implementation step.  All quantities should eventually be returned in the
    global AO basis so the result can be passed into the existing SSDMET
    Hamiltonian projection machinery.
    """
    raise NotImplementedError(
        "fragment-local bath construction is not implemented yet"
    )


def run_fragment_scf(mol, impurity_idx, fragment_idx, charge,
                     fragment_scf="rohf", fragment_scf_verbose=3, **kwargs):
    """
    Run a low-level reference calculation for one impurity-fragment subsystem.
    """
    spin = kwargs.pop('spin', None)
    frag_mol, atom_ids, global_idx, local_imp_idx, local_fragment_idx = \
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
        atom_ids=atom_ids,
        global_idx=global_idx,
        local_imp_idx=local_imp_idx,
        local_fragment_idx=local_fragment_idx,
    )


def build_fragment(mol, impurity_idx, fragment_idx, charge,
                   fragment_scf="cahf", fragment_scf_verbose=3, **kwargs):
    """
    Build the first-stage impurity-ligand fragment reference.

    The returned object contains the fragment Mole, the converged local SCF
    object, and index maps between the parent AO basis and the local fragment
    AO basis.
    """
    return run_fragment_scf(mol, impurity_idx, fragment_idx,
                            charge=charge,
                            fragment_scf=fragment_scf,
                            fragment_scf_verbose=fragment_scf_verbose,
                            **kwargs)


def bath_from_fragment_density(dm, impurity_idx, fragment_idx, overlap=None,
                               threshold=1e-13):
    """
    Construct bath orbitals from one fragment density matrix.

    The intended convention is that dm is represented in the orthogonal local
    basis whose first block is the fixed impurity space and whose second block
    is the fragment environment space.  Bath orbitals are selected by
    diagonalizing the environment density block.
    """
    raise NotImplementedError(
        "bath construction from fragment density is not implemented yet"
    )


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
    def __init__(self, mol, title='untitled', imp_idx=None,
                 imp_charge=None, fragments=None, fragment_charges=None,
                 fragment_scf='cahf',
                 fragment_scf_options=None, threshold=1e-13,
                 verbose=logger.INFO):
        if not isinstance(mol, gto.mole.Mole):
            raise TypeError("FDMET requires a PySCF Mole object")
        if imp_idx is None:
            raise ValueError("FDMET requires impurity AO labels")
        self.input_obj = mol
        self.fragments = None
        self.imp_charge = imp_charge
        self.fragment_charges = None
        self.fragment_scf = fragment_scf
        self.fragment_scf_options = (
            {} if fragment_scf_options is None else dict(fragment_scf_options)
        )

        self.mol = mol
        self.max_mem = getattr(self.mol, 'max_memory', 4000)
        self.title = title
        self.verbose = verbose
        self.dm = None
        self._imp_idx = []
        self.imp_idx = imp_idx
        self.threshold = threshold

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
        self.fragment_refs = None
        self.fragments = _parse_fragments(self.mol, fragments)
        if self.fragments is not None:
            if fragment_charges is not None:
                self.fragment_charges = list(fragment_charges)
                if len(self.fragment_charges) != len(self.fragments):
                    raise ValueError(
                        "fragment_charges must have the same length as fragments"
                    )

    def dump_flags(self):
        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('fragment scf = %s', self.fragment_scf)
        log.info('fragment scf options = %s', self.fragment_scf_options)
        log.info('impurity charge = %s', self.imp_charge)
        if self.fragments is None:
            log.info('fragments = None')
        else:
            log.info('number of fragments = %d', len(self.fragments))
            log.info('fragment charges = %s', self.fragment_charges)

    def build(self, fragment_scf_verbose=3, chk_fname_load='', save_chk=False):
        return self.build_fragment(fragment_scf_verbose=fragment_scf_verbose,
                                   chk_fname_load=chk_fname_load,
                                   save_chk=save_chk)

    def build_fragment(self, fragment_scf_verbose=3,
                       chk_fname_load='', save_chk=False):
        self.dump_flags()
        if self.fragments is None:
            raise ValueError(
                "fragment bath construction requires fragments to be assigned"
            )
        self.fragment_refs = []
        for ifrag, fragment_idx in enumerate(self.fragments):
            logger.info(self, 'build impurity-ligand fragment %d', ifrag)
            scf_options = dict(self.fragment_scf_options)
            if self.imp_charge is None or self.fragment_charges is None:
                raise ValueError(
                    "imp_charge and fragment_charges must be assigned together"
                )
            charge = self.imp_charge + self.fragment_charges[ifrag]
            frag_ref = build_fragment(
                self.mol, self.imp_idx, fragment_idx,
                charge=charge,
                fragment_scf=self.fragment_scf,
                fragment_scf_verbose=fragment_scf_verbose,
                **scf_options,
            )
            self.fragment_refs.append(frag_ref)
        raise NotImplementedError(
            "fragment CAHF references are built; bath construction is not implemented yet"
        )

    def density_fit(self, with_df=None):
        raise NotImplementedError(
            "density fitting for FDMET is not implemented yet"
        )
