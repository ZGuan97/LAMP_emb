from pyscf import gto
from pyscf.lib import logger

from embed_sim import ssdmet


def _parse_aolabels(mol, aolabels, name):
    indices = [int(x) for x in gto.mole._aolabels2baslst(mol, aolabels, base=0)]
    if len(indices) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate AO indices")
    nao = mol.nao
    bad = [x for x in indices if x < 0 or x >= nao]
    if bad:
        raise ValueError(f"{name} contains AO indices outside [0, {nao})")
    return indices


def _validate_fragments(mol, fragments):
    if fragments is None:
        return None
    return [
        _parse_aolabels(mol, frag, f"fragment {ifrag}")
        for ifrag, frag in enumerate(fragments)
    ]


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


def run_fragment_scf(mol, impurity_idx, fragment_idx, fragment_scf="rohf",
                     **kwargs):
    """
    Run a low-level reference calculation for one impurity-fragment subsystem.
    """
    raise NotImplementedError(
        "fragment-local SCF calculations are not implemented yet"
    )


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
    fragments
        Ligand fragment AO labels.  Each item is one fragment.
    """
    def __init__(self, mol, title='untitled', imp_idx=None,
                 fragments=None, fragment_scf='rohf',
                 threshold=1e-13, verbose=logger.INFO):
        if not isinstance(mol, gto.mole.Mole):
            raise TypeError("FDMET requires a PySCF Mole object")
        if imp_idx is None:
            raise ValueError("FDMET requires impurity AO labels")
        self.input_obj = mol
        self.fragments = None
        self.fragment_scf = fragment_scf

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
        self.fragments = _validate_fragments(self.mol, fragments)

    def dump_flags(self):
        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('fragment scf = %s', self.fragment_scf)
        if self.fragments is None:
            log.info('fragments = None')
        else:
            log.info('number of fragments = %d', len(self.fragments))

    def build(self, chk_fname_load='', save_chk=False):
        return self.build_fragment(chk_fname_load=chk_fname_load,
                                   save_chk=save_chk)

    def build_fragment(self, chk_fname_load='', save_chk=False):
        self.dump_flags()
        if self.fragments is None:
            raise ValueError(
                "fragment bath construction requires fragments to be assigned"
            )
        raise NotImplementedError(
            "FDMET fragment bath construction is scaffolded but not implemented"
        )

    def set_fragments(self, fragments):
        self.fragments = _validate_fragments(self.mol, fragments)
        return self

    def density_fit(self, with_df=None):
        raise NotImplementedError(
            "density fitting for FDMET is not implemented yet"
        )
