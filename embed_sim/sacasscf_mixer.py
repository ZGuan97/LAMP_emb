from pyscf import mcscf, fci, mrpt, lib
from pyscf.lib import logger

import numpy as np

from embed_sim import spin_utils

def sacasscf_mixer(mf, ncas, nelec, statelis=None, weights = None, fix_spin_shift=0.5):
    # TODO wrap the solver by a class to have statelis as its property for SISO convenience
    solver = mcscf.CASSCF(mf,ncas,nelec)

    if statelis is None:
        logger.info(solver,'statelis is None')
        statelis = spin_utils.gen_statelis(ncas, nelec)
        logger.info(solver,'generate statelis %s', statelis)

    solvers = []
    logger.info(solver,'Attempting SA-CASSCF with')
    for i in range(len(statelis)):
        if i == 0 and statelis[0]:
            newsolver = fci.direct_spin1.FCI(mf)
            newsolver.spin = 0
            newsolver = fci.addons.fix_spin(newsolver,ss=(i/2)*(i/2+1),shift=fix_spin_shift)
            print('fix_spin parameter', fix_spin_shift, 'on spin multiplicity', 1)
            newsolver.nroots = statelis[0]
            solvers.append(newsolver)
            logger.info(solver,'%s states with spin multiplicity %s',statelis[0],0+1)
        elif statelis[i]:
            newsolver = fci.direct_spin1.FCI(mf)
            newsolver.spin = i
            newsolver = fci.addons.fix_spin(newsolver,ss=(i/2)*(i/2+1),shift=fix_spin_shift)
            print('fix_spin parameter', fix_spin_shift, 'on spin multiplicity', i+1)
            newsolver.nroots = statelis[i]
            solvers.append(newsolver)
            logger.info(solver,'%s states with spin multiplicity %s',statelis[i],i+1)

    statetot = np.sum(statelis)
    if weights is None:
        weights = np.ones(statetot)/statetot
    mcscf.state_average_mix_(solver, solvers, weights)
    return solver

def read_statelis(mc):
    spins = []
    nroots = []
    for solver in mc.fcisolver.fcisolvers:
        spins.append(solver.spin)
        nroots.append(solver.nroots)
    max_spin = np.max(np.array(spins))
    statelis = np.zeros(max_spin + 1, dtype=int)
    statelis[spins] = nroots
    return statelis

def sacasscf_nevpt2(mc, method='SC', **kwargs):
    if method.upper() == 'SC':
        return sacasscf_nevpt2_casci_ver(mc)
    elif method.upper() in ['PC','FIC','QD']:
        return sacasscf_nevpt2_prism(mc, method, **kwargs)
    else:
        raise NotImplementedError('Currently only SC-, PC-(FIC-) and QD- are supported')

from pyscf.fci.addons import _unpack_nelec
def sacasscf_nevpt2_undo_ver(mc):
    from pyscf.mcscf.addons import StateAverageFCISolver
    from pyscf.mcscf.df import _DFCAS
    if isinstance(mc.fcisolver, StateAverageFCISolver):
        spins = []
        nroots = []
        for solver in mc.fcisolver.fcisolvers:
            spins.append(solver.spin)
            nroots.append(solver.nroots)
        e_corrs = []
        print('undo state_average')
        sa_fcisolver = mc.fcisolver
        mc.fcisolver = mc.fcisolver.undo_state_average()
        for i, spin in enumerate(spins):
            mc.nelecas = _unpack_nelec(mc.nelecas, spin)
            mc.fcisolver.spin = spin
            nroot = nroots[i]
            for iroot in range(0, nroot):
                if isinstance(mc, _DFCAS):
                    from embed_sim.df import DFNEVPT
                    nevpt2 = DFNEVPT(mc, root=iroot+np.sum(nroots[:i],dtype=int), spin=spin)
                else:
                    print('spin', spin, 'iroot', iroot)
                    nevpt2 = mrpt.NEVPT(mc, root=iroot+np.sum(nroots[:i],dtype=int))
                nevpt2.verbose = logger.INFO-1 # when verbose=logger.INFO, meta-lowdin localization is called and cause error in DMET-NEVPT2
                e_corr = nevpt2.kernel()
                e_corrs.append(e_corr)
        print('redo state_average')
        mc.fcisolver = sa_fcisolver
    else:
        raise TypeError(mc.fcisolver, 'Not StateAverageFCISolver')
    return np.array(e_corrs)

def sacasscf_nevpt2_casci_ver(mc):
    print('sacasscf_nevpt2_casci_ver')
    from pyscf.mcscf.addons import StateAverageFCISolver
    from pyscf.mcscf.df import _DFCAS
    if isinstance(mc.fcisolver, StateAverageFCISolver):
        spins = []
        nroots = []
        for solver in mc.fcisolver.fcisolvers:
            spins.append(solver.spin)
            nroots.append(solver.nroots)
        e_corrs = []
        for i, spin in enumerate(spins):
            print('CASCI')
            mc_ci = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
            mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
            mc_ci.fcisolver.spin = spin
            mc_ci.fix_spin_(shift=0.5, ss=(spin/2)*(spin/2+1))
            mc_ci.fcisolver.nroots = nroots[i] # this is important for convergence of CASCI
            mc_ci.kernel(mc.mo_coeff)
            nroot = nroots[i]
            for iroot in range(0, nroot):
                if isinstance(mc, _DFCAS):
                    from embed_sim.df import DFNEVPT
                    nevpt2 = DFNEVPT(mc_ci, root=iroot, spin=spin)
                else:
                    print('spin', spin, 'iroot', iroot)
                    nevpt2 = mrpt.NEVPT(mc_ci, root=iroot)
                nevpt2.verbose = logger.INFO-1 # when verbose=logger.INFO, meta-lowdin localization is called and cause error in DMET-NEVPT2
                # nevpt2.verbose = 0 # when verbose=logger.INFO, meta-lowdin localization is 
                e_corr = nevpt2.kernel()
                e_corrs.append(e_corr)
    else:
        raise TypeError(mc.fcisolver, 'Not StateAverageFCISolver')
    return np.array(e_corrs)

def sacasscf_nevpt2_prism(mc, method, **kwargs):
    log = logger.new_logger(mc)
    try:
        from prism.interface import PYSCF as pyscf2prism
        from prism.nevpt import NEVPT as prism_nevpt2
    except ImportError:
        log.warn('Prism is not installed and SC-NEVPT2 is applied! Please see https://github.com/sokolov-group/prism')
        return sacasscf_nevpt2_casci_ver(mc)
    else:
        from pyscf.mcscf.addons import StateAverageFCISolver
        from pyscf.mcscf.df import _DFCAS
        if isinstance(mc.fcisolver, StateAverageFCISolver):
            spins = []
            nroots = []
            for solver in mc.fcisolver.fcisolvers:
                spins.append(solver.spin)
                nroots.append(solver.nroots)
            e_corrs = []
            for i, spin in enumerate(spins):
                log.info('CASCI for spin %s', spin)
                mc_ci = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
                mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
                mc_ci.fcisolver.spin = spin
                mc_ci.fix_spin_(shift=0.5, ss=(spin/2)*(spin/2+1))
                mc_ci.fcisolver.nroots = nroots[i] # this is important for convergence of CASCI
                mc_ci.kernel(mc.mo_coeff)

                interface = pyscf2prism(mc._scf, mc_ci, opt_einsum = True)
                if isinstance(mc, _DFCAS):
                    interface = pyscf2prism(mc._scf, mc_ci, opt_einsum = True).density_fit(with_df=mc._scf.with_df)
                    interface.get_naux = mc._scf.with_df.get_naoaux
                else:
                    interface = pyscf2prism(mc._scf, mc_ci, opt_einsum = True)
                def _charge_center(mol):
                    charges = mol.atom_charges()
                    coords  = mol.atom_coords()
                    return lib.einsum('z,zr->r', charges, coords)/charges.sum()
                mol = mc._scf.mol
                if mol.nao == 0:
                    try:
                        mydmet = kwargs['dmet']
                    except KeyError:
                        log.warn('No dmet object input, the transition dipole moments will be set to zeros')
                        interface.dip_mom_ao = np.zeros((3,mc_ci.mo_coeff.shape[-1],mc_ci.mo_coeff.shape[-1]))
                    else:
                        mol = mydmet.mol
                        with mol.with_common_orig(_charge_center(mol)):
                            interface.dip_mom_ao = lib.einsum('xij,ip,jq->xpq', mol.intor_symmetric('int1e_r', comp=3),
                                                              mydmet.es_orb, mydmet.es_orb)
                else:
                    interface.dip_mom_ao = lib.einsum('xij,ip,jq->xpq', mol.intor_symmetric('int1e_r', comp=3),
                                                      mydmet.es_orb, mydmet.es_orb)
                
                nevpt2 = prism_nevpt2(interface)
                if method.upper() in ['PC','FIC']:
                    nevpt2.method = "nevpt2"
                else:
                    nevpt2.method = 'qd-nevpt2'
                if 'expert_options' in kwargs.keys():
                    expert_options = kwargs['expert_options']
                    for option, value in expert_options.items():
                        setattr(nevpt2, option, value)
                e_tot, e_corr, osc = nevpt2.kernel()
                e_corrs += e_corr
        else:
            raise TypeError(mc.fcisolver, 'Not StateAverageFCISolver')
        return np.array(e_corrs)

def analysis(mc):
    from pyscf.mcscf.addons import StateAverageFCISolver
    if isinstance(mc.fcisolver, StateAverageFCISolver):
        spins = []
        nroots = []
        for solver in mc.fcisolver.fcisolvers:
            spins.append(solver.spin)
            nroots.append(solver.nroots)
        e_corrs = []
        for i, spin in enumerate(spins):
            mc_ci = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
            mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
            mc_ci.fcisolver.spin = spin
            mc_ci.fix_spin_(shift=0.5, ss=(spin/2)*(spin/2+1))
            mc_ci.fcisolver.nroots = nroots[i] # this is important for convergence of CASCI
            mc_ci.kernel(mc.mo_coeff)
            nroot = nroots[i]
            for iroot in range(0, nroot):
                print('analyze spin', spin, 'iroot', iroot)
                if nroot == 1:
                    mc_ci.analyze(ci=mc_ci.ci)
                else:
                    mc_ci.analyze(ci=mc_ci.ci[iroot])
    else:
        raise TypeError(mc.fcisolver, 'Not StateAverageFCISolver')
    return np.array(e_corrs)