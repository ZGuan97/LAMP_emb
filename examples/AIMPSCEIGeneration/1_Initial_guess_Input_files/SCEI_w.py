# User interface for Ab-initio Model Potential package

from email.policy import default
#from turtle import home
from embed_sim.AIMP3_Bare_Ion import AIMPEnvLoader, AIMP_RHF, AIMP_RKS, AIMP_ROHF, AIMP_ROKS, AIMP_UKS, AIMP_GKS
from embed_sim.pckit2 import OrganicPCLoader, PointChargeParams
from embed_sim.constraint_optimizer import ConstraintOptimizer
from embed_sim.EnvGenerator import XYZParser
from pyscf import gto, lib
from pyscf.tddft import TDA
from pyscf.tools.cubegen import orbital
import yaml, sys, getopt
import numpy as np
import basis_set_exchange as bse

Ha2eV = 27.211324570273

opts, args = getopt.getopt(sys.argv[1:], "i:f:")

for op, value in opts:
    if op == "-i": 
        workdir = value + "/"   # The working directory

logdir = workdir + "log.out"
logf = open(logdir, 'w')
sys.stdout = logf

inputdir = workdir + "input.yaml"
with open(inputdir, 'r') as f:
    inputdict = yaml.safe_load(f)
aimpdict = inputdict["aimp"]
clusterdict = inputdict["cluster"]
#add by ZhangTeng 2023.7.12
try: socdict = inputdict["soc"]
except KeyError: socdict = False

# Saving the interpolation results
qlst = []
gs_energy_lst = []
ex_energy_lst = []

#|ddm| as converge threshold for density matrix
DDM = 10
# ################################

# Load AIMP environment
try: aimpdir = workdir + aimpdict["dir"]
except KeyError: aimpdir = workdir + "aimp.xyz"
aimpdict['dir'] = aimpdir
aimpdict['workdir'] = workdir
AIMP_LOADER = AIMPEnvLoader(aimpdict,socdict)
try: orthoreg = aimpdict["orthoreg"]
except: orthoreg = 0

# Load Point Charge Environment
try: 
    pcdict = inputdict["pointcharge"]
    try: is_organic = pcdict['organic']
    except KeyError: is_organic = False

    try: rawxyzdir = workdir + pcdict["rawxyzdir"]
    except KeyError: rawxyzdir = workdir + "rawChgs.xyz"

    if not is_organic:
        try: rawchgdir = workdir + pcdict["rawchgdir"]
        except KeyError: rawchgdir = workdir + "rawCharges.dat"
        pcparam_raw = PointChargeParams(rawxyzdir, rawchgdir)
    else:
        pcdict = aimpdict
        pcdict["dir"] = rawxyzdir
        pcdict['workdir'] = workdir
        pc_loader = OrganicPCLoader(pcdict)
        pcparam_raw = pc_loader.make_param()

    try: surfxyzdir = workdir + pcdict["surfxyzdir"]
    except KeyError: surfxyzdir = workdir + "surfChgs.xyz"
    try: surfchgdir = workdir + pcdict["surfchgdir"]
    except KeyError: surfchgdir = workdir + "surfaceCharges.dat"
    pcparam_surf = PointChargeParams(surfxyzdir, surfchgdir)

    PCPARAM = pcparam_raw + pcparam_surf

except: PCPARAM = None


# Load cluster molecule
try: ecp = clusterdict["ecp"]
except KeyError: ecp = {}
try: spin = clusterdict["spin"]
except KeyError: spin = 0
try: verbose = inputdict["verbose"]
except KeyError: verbose = 3
try: cubes = inputdict["cube"]
except KeyError: cubes = -1
try: cluster_fake_name = clusterdict["cluster_fake_name"]
except KeyError: cluster_fake_name = False


def aimp_calc(CLUS_MOL, scfdict, interpolation=False):

    scftype = scfdict["calc"].upper()
    if socdict:
        if scftype in ['DFT', 'KS', 'RKS', 'TDDFT', 'TDKS']:
            try: xc = clusterdict['scf']['xc']
            except KeyError: xc = 'b3lyp'
            MF = AIMP_GKS(CLUS_MOL, AIMP_LOADER, xc=xc)
            MF.collinear = 'mcol'    
            dm = MF.get_init_guess()
            dm = dm.astype(np.complex128)
            dm = dm + 0j
            dm[0,:] += .1j
            dm[:,0] -= .1j
        else:
            raise NotImplementedError("Other tags have not been implemented yet!")

        if PCPARAM is not None: MF.addPCParam2(PCPARAM)
        MF.set_orthoreg_param(orthoreg)
    else:
        if scftype in ['HF', 'RHF', 'TDHF', 'TDAHF']:
            MF = AIMP_RHF(CLUS_MOL, AIMP_LOADER)
        elif scftype in ['ROHF']:
            MF = AIMP_ROHF(CLUS_MOL, AIMP_LOADER)
        elif scftype in ['DFT', 'KS', 'RKS', 'TDDFT', 'TDKS']:
            if spin == 0:
                try: xc = clusterdict['scf']['xc']
                except KeyError: xc = 'b3lyp'
                MF = AIMP_RKS(CLUS_MOL, AIMP_LOADER, xc=xc)
            else:
                try: xc = clusterdict['scf']['xc']
                except KeyError: xc = 'b3lyp'
                MF = AIMP_UKS(CLUS_MOL, AIMP_LOADER, xc=xc)
        elif scftype in ['ROKS']:
            try: xc = clusterdict['scf']['xc']
            except KeyError: xc = "b3lyp"
            MF = AIMP_ROKS(CLUS_MOL, AIMP_LOADER, xc=xc)
        else:
            raise NotImplementedError("Other tags have not been implemented yet!")

        if PCPARAM is not None: MF.addPCParam2(PCPARAM)
        MF.set_orthoreg_param(orthoreg)

    # checkpoint file for TDA method
    try: MF.chkfile = workdir + inputdict['chkfile']
    except KeyError: pass

    if socdict:
        MF.kernel(dm0=dm)
    else:
        MF.kernel()
    if interpolation: gs_energy_lst.append(MF.e_tot)

    if 'TD' in scftype: # Do time-dependent calculation using Tam-dancoff approx
        MF = TDA(MF)
        try: 
            tdopt = scfdict['tdopt']
            for key in tdopt.keys():
                setattr(MF, key, tdopt[key])
        except KeyError: pass

        # checkpoint file for TDA method
        try: MF.chkfile = workdir + inputdict['chkfile']
        except KeyError: pass

        MF.kernel()
        if interpolation:
            ex_energy_lst.append(MF.e_tot + MF.e[0] / Ha2eV)

    return MF

# AIMP Energy calculation
if inputdict["type"].upper() in ["GEO_OPT", "GEOM_OPT", "RELAX", "GEOMOPT", "ENERGY"]:
    try: clusterdir = workdir + clusterdict["dir"]
    except KeyError: clusterdir = workdir + "cluster.xyz"

    # added for custom basis sets in the NWChem format
    for basis in clusterdict["basis"]:
        if 'parse' in clusterdict["basis"][basis] or 'load' in clusterdict["basis"][basis]:
            clusterdict["basis"][basis] = eval(clusterdict["basis"][basis])

    CLUS_MOL = gto.M(atom=clusterdir, basis=clusterdict["basis"], 
    ecp=ecp, charge=clusterdict["charge"], spin=spin, verbose=verbose)
    if socdict:
        NELEC = CLUS_MOL.nelec[0] + CLUS_MOL.nelec[1]
    else:
        NELEC = CLUS_MOL.nelec[0]
    NAMELINE = np.loadtxt(clusterdir, skiprows=2, max_rows=1, dtype=str)
    if cluster_fake_name:
        NAME = cluster_fake_name
    else:
        NAME = NAMELINE[0]
    scfdict = clusterdict["scf"]
    MF = aimp_calc(CLUS_MOL, scfdict)
    MO_COEFF = MF.mo_coeff[:,:NELEC]
    MO_ENERGY = MF.mo_energy[:NELEC]
    RDM = MF.make_rdm1()
    E_TOT = MF.e_tot

    file_name_mo_coeff = 'MO_COEFF_' + NAME + '.txt'
    file_name_mo_energy = 'MO_ENERGY_' + NAME + '.txt'
    file_name_rdm = 'RDM_' + NAME + '.txt'
    file_name_etot = 'E_TOT_' + NAME + '.txt'

    #Output new results
    np.savetxt(file_name_mo_coeff, MO_COEFF)
    np.savetxt(file_name_mo_energy, MO_ENERGY)
    np.savetxt(file_name_rdm, RDM)
    np.savetxt(file_name_etot, np.array([E_TOT]))

    # Visualizing orbitals
    if cubes!= -1:
        for cube in cubes:
            cube_name = str(cube)
            coeff = MF.mo_coeff[:,cube]
            orbital(CLUS_MOL, 'orbital'+cube_name+'.cube', coeff)

# Geometry Optimization
#add by ZhangTeng 2024.4.10
if inputdict['type'].upper() in ["GEO_OPT", "GEOM_OPT", "RELAX", "GEOMOPT"]:
    optdict = inputdict["geomopt"]
    try: constraint = optdict["constraint"]
    except KeyError: constraint = False
    try: 
        td_state = optdict["td_state"]
        scanner = MF.nuc_grad_method().as_scanner(state=td_state)
    except KeyError:
        scanner = MF.nuc_grad_method().as_scanner()
    if constraint:
        params = {"constraints": "constraints.txt",}  
        OPT = MF.Gradients().optimizer(solver='geomeTRIC')
        OPT.kernel(params)
    else:
        OPT = scanner.optimizer(); OPT.kernel()
    optmol = OPT.mol
    optmol.tofile(workdir + optdict["xyzdir"], format="xyz")


# Interpolation
if inputdict["type"].upper() in ['INTERPOLATION']:
    interdict = inputdict["interpolation"]
    try: ngrid = interdict["ngrid"]
    except KeyError: ngrid = 25
    scfdict = clusterdict["scf"]
    xlst = np.linspace(-1, 2, ngrid)
    gsdir, exdir = interdict["xyzdir"]
    gs = XYZParser(file=workdir + gsdir)
    ex = XYZParser(file=workdir + exdir)
    delta = gs + ex * (-1)
    dQ = delta.calc_dQ()
    for x in xlst:
        q = dQ * x
        qlst.append(q)
        molparser = gs * (1. - x) + ex * x
        CLUS_MOL = gto.M(atom=molparser.__str__(), basis=clusterdict["basis"], 
        ecp=ecp, charge=clusterdict["charge"], spin=spin)
        aimp_calc(CLUS_MOL, scfdict=scfdict, interpolation=True)
    
    q_and_gs = np.array([qlst, gs_energy_lst])
    e_tda = np.array(ex_energy_lst).T
    totlst = np.vstack((q_and_gs, e_tda))
    np.savetxt(workdir + interdict["outdir"], totlst)

# close log file
logf.close()
