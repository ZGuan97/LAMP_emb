from pyscf.geomopt.geometric_solver import PySCFEngine, NotConvergedError, \
    GeometryOptimizer, dump_mol_geometry
import numpy as np
import os
import tempfile
import geometric
import geometric.molecule
from pyscf import lib
from pyscf.geomopt.addons import dump_mol_geometry
from pyscf import __config__
from pyscf.grad.rhf import GradientsMixin

try:
    from geometric import internal, optimize, nifty, engine, molecule
except ImportError:
    msg = ('Geometry optimizer geomeTRIC not found.\ngeomeTRIC library '
           'can be found on github https://github.com/leeping/geomeTRIC.\n'
           'You can install geomeTRIC with "pip install geometric"')
    raise ImportError(msg)

# Overwrite units defined in geomeTRIC
internal.ang2bohr = optimize.ang2bohr = nifty.ang2bohr = 1./lib.param.BOHR
engine.bohr2ang = internal.bohr2ang = molecule.bohr2ang = nifty.bohr2ang = \
        optimize.bohr2ang = lib.param.BOHR
del(internal, optimize, nifty, engine, molecule)

# logging.config.fileConfig(logIni,defaults={'logfilename': logfilename},disable_existing_loggers=False)


INCLUDE_GHOST = getattr(__config__, 'geomopt_berny_solver_optimize_include_ghost', True)
ASSERT_CONV = getattr(__config__, 'geomopt_berny_solver_optimize_assert_convergence', True)

def constraint_parser(constraint_str:str):
    constraint_list = []
    nodes = constraint_str.split(",")
    for node in nodes:
        barsp = node.split("-")
        if len(barsp) == 1: 
            try: constraint_list.append(eval(barsp[0]))
            except: raise ValueError("Must be integers!")
        elif len(barsp) == 2:
            start, end = barsp
            try: start = eval(start); end = eval(end)
            except: raise ValueError("Must be integers!")
            constraint_list.extend(range(start, end+1))
        else:
            raise ValueError("Please check your form!")
    return np.array(constraint_list) - 1

class ConstraintEngine(PySCFEngine):
    def __init__(self, scanner, constraint_list):
        super().__init__(scanner)
        mask = np.ones(self.mol.natm, np.bool)
        mask[constraint_list] = False
        unmask = np.ones(self.mol.natm, np.bool)
        unmask[constraint_list] = True

        self.mask = mask
        self.unmask = unmask
        # fixed coordinates of the atom
        # Note: unit is Bohr now!
        self.fixed_coords = self.mol.atom_coords()[unmask]

        active_coords = self.mol.atom_coords()[mask]
        atms = np.array([self.mol.atom_symbol(i) for i in range(self.mol.natm)])
        active_atms = list(atms[mask])

        molecule = geometric.molecule.Molecule()
        molecule.elem = active_atms
        molecule.xyzs = [active_coords * lib.param.BOHR]
        super(PySCFEngine, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        if self.cycle >= self.maxsteps:
            raise NotConvergedError('Geometry optimization is not converged in '
                                    '%d iterations' % self.maxsteps)

        g_scanner = self.scanner
        mol = self.mol
        self.cycle += 1
        lib.logger.note(g_scanner, '\nGeometry optimization cycle %d', self.cycle)

        total_coords = np.zeros((mol.natm, 3))
        total_coords[self.unmask] = self.fixed_coords

        # geomeTRIC requires coords and gradients in atomic unit
        coords = coords.reshape(-1,3)

        total_coords[self.mask] = coords
        if g_scanner.verbose >= lib.logger.NOTE:
            dump_mol_geometry(mol, total_coords*lib.param.BOHR)

        # Temporately unsupport the symmetry
        #if mol.symmetry:
        #    coords = symmetrize(mol, total_coords)

        mol.set_geom_(total_coords, unit='Bohr')
        energy, gradients = g_scanner(mol)
        gradients = gradients[self.mask] # use the masked gradients
        print(gradients)
        lib.logger.note(g_scanner,
                        'cycle %d: E = %.12g  dE = %g  norm(grad) = %g', self.cycle,
                        energy, energy - self.e_last, np.linalg.norm(gradients))
        self.e_last = energy

        if callable(self.callback):
            self.callback(locals())

        if self.assert_convergence and not g_scanner.converged:
            raise RuntimeError('Nuclear gradients of %s not converged' % g_scanner.base)
        return {"energy": energy, "gradient": gradients.ravel()}

def kernel(method, constraint_list,
           assert_convergence=ASSERT_CONV,
           include_ghost=INCLUDE_GHOST, constraints=None, callback=None,
           maxsteps=100, **kwargs):

    # only support scanner and gradient for simplification
    if isinstance(method, lib.GradScanner):
        g_scanner = method
    elif isinstance(method, GradientsMixin):
        g_scanner = method.as_scanner()
    elif getattr(method, 'nuc_grad_method', None):
        g_scanner = method.nuc_grad_method().as_scanner()
    if not include_ghost:
        g_scanner.atmlst = np.where(method.mol.atom_charges() != 0)[0]
    
    tmpf = tempfile.mktemp(dir=lib.param.TMPDIR)
    engine = ConstraintEngine(g_scanner, constraint_list)
    engine.callback = callback
    engine.maxsteps = maxsteps
    # To avoid overwritting method.mol
    engine.mol = g_scanner.mol.copy()
    
    if not os.path.exists(os.path.abspath(
            os.path.join(geometric.optimize.__file__, '..', 'log.ini'))):
        kwargs['logIni'] = os.path.abspath(os.path.join(__file__, '..', 'log.ini'))

    engine.assert_convergence = assert_convergence
    try:
        geometric.optimize.run_optimizer(customengine=engine, input=tmpf,
                                         constraints=constraints, **kwargs)
        conv = True
        # method.mol.set_geom_(m.xyzs[-1], unit='Angstrom')
    except NotConvergedError as e:
        lib.logger.note(method, str(e))
        conv = False
    return conv, engine.mol

class ConstraintOptimizer(GeometryOptimizer):
    def __init__(self, method):
        super().__init__(method)

    def kernel(self, constraint_str=None, params=None):
        if params is not None:
            self.params.update(params)
        if constraint_str is not None:
            constraint_list = constraint_parser(constraint_str)
            self.converged, self.mol = \
                    kernel(self.method, constraint_list, callback=self.callback,
                        maxsteps=self.max_cycle, **self.params)
        else:
            self.converged
        return self.mol

    optimize = kernel


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.M(atom='''
        C       1.1879  -0.3829 0.0000
        C       0.0000  0.5526  0.0000
        O       -1.1867 -0.2472 0.0000
        H       -1.9237 0.3850  0.0000
        H       2.0985  0.2306  0.0000
        H       1.1184  -1.0093 0.8869
        H       1.1184  -1.0093 -0.8869
        H       -0.0227 1.1812  0.8852
        H       -0.0227 1.1812  -0.8852
                ''',
                basis='3-21g')

    mf = scf.RHF(mol)
    mf.kernel()
    constraint_str = "1-3, 5, 7"
    opt = ConstraintOptimizer(mf.nuc_grad_method())
    opt.kernel(constraint_str)
