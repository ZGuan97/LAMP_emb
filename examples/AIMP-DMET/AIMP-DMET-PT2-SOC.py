import sys, os

from embed_sim.AIMP3_DMET_SCEI import AIMPEnvLoader, AIMP_RHF, AIMP_RKS, AIMP_ROHF, AIMP_ROKS, AIMP_CAHF
from embed_sim.pckit2 import OrganicPCLoader, PointChargeParams
from embed_sim.EnvGenerator import XYZParser
from pyscf import gto, lib, scf, ao2mo, lo
from pyscf.lib import chkfile
from pyscf.lib import logger
import yaml, getopt
import numpy as np
from embed_sim import cahf, rdiis, siso
import basis_set_exchange as bse
from functools import reduce
from pyscf.tools import molden
import scipy
import h5py


Ha2eV = 27.211324570273

workdir = "./"
inputdir = workdir + "input.yaml"
with open(inputdir, 'r') as f:
    inputdict = yaml.safe_load(f)
aimpdict = inputdict["aimp"]
clusterdict = inputdict["cluster"]

# Load AIMP environment
try: aimpdir = workdir + aimpdict["dir"]
except KeyError: aimpdir = workdir + "aimp.xyz"
aimpdict['dir'] = aimpdir
aimpdict['workdir'] = workdir
AIMP_LOADER = AIMPEnvLoader(aimpdict)
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

def aimp_calc(CLUS_MOL, scfdict, interpolation=False):

    scftype = scfdict["calc"].upper()
    print(scftype)

    if scftype in ['HF', 'RHF', 'TDHF', 'TDAHF']:
        MF = AIMP_RHF(CLUS_MOL, AIMP_LOADER).density_fit()
    elif scftype in ['ROHF']:
        MF = AIMP_ROHF(CLUS_MOL, AIMP_LOADER).density_fit()
    elif scftype in ['CAHF']:
        MF = AIMP_CAHF(CLUS_MOL, AIMP_LOADER).x2c().density_fit()
    elif scftype in ['DFT', 'KS', 'RKS', 'TDDFT', 'TDKS']:
        try: xc = clusterdict['scf']['xc']
        except KeyError: xc = 'b3lyp'
        MF = AIMP_RKS(CLUS_MOL, AIMP_LOADER, xc=xc)
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
    
    return MF

# AIMP Energy calculation
if inputdict["type"].upper() in ["GEO_OPT", "GEOM_OPT", "RELAX", "GEOMOPT", "ENERGY"]:
    try: clusterdir = workdir + clusterdict["dir"]
    except KeyError: clusterdir = workdir + "cluster.xyz"

    # added for custom basis sets in the NWChem format
    for basis in clusterdict["basis"]:
        if 'parse' in clusterdict["basis"][basis] or 'load' in clusterdict["basis"][basis]:
            clusterdict["basis"][basis] = eval(clusterdict["basis"][basis])
    
    CLUS_MOL = gto.M(atom=clusterdir, basis=clusterdict["basis"], charge=clusterdict["charge"], spin=spin, verbose=4)
    scfdict = clusterdict["scf"]
    MF = aimp_calc(CLUS_MOL, scfdict)
    # rdiis_ref = np.diag([int(x in CLUS_MOL.search_ao_label(['Ce.*'])) for x in range(MF.get_ovlp().shape[0])])
    #MF.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=CLUS_MOL.search_ao_label(['Ce.*']),power=0.2)
    # rdiis.tag_rdiis_(MF,reg='dS',imp_inds=CLUS_MOL.search_ao_label(['Ce.*']),power=0.2)
    MF.max_cycle = 3000
    MF.conv_tol = 1e-07
    #MF.level_shift = 2
    if os.path.exists(MF.chkfile):
        print("Load from chk file.")
        MF.init_guess = 'chk'
        scfdat = chkfile.load(MF.chkfile,'scf')
        MF.e_tot = scfdat['e_tot']
        MF.mo_coeff = scfdat['mo_coeff']
        MF.mo_occ = scfdat['mo_occ']
        MF.mo_energy = scfdat['mo_energy']
        #MF.kernel()
    else:
        MF.init_guess = 'minao'
        print()
        print("Ready for MF.kernel().")
        MF.kernel()

print()
print("Enter to AO-DMET procedure.")
title = 'Ce'

imp_inds = CLUS_MOL.search_ao_label(['Ce.*'])
thres = 1e-12

from embed_sim import ssdmet, sacasscf_mixer, siso
mydmet = ssdmet.SSDMET(MF, title=title, imp_idx=imp_inds, threshold=thres, readmp2 = False, es_natorb=False, bath_option={'ROMP2':254}).density_fit()
mydmet.readmp2 = False
mydmet.build(restore_imp = True)
#es_mf = mydmet.ROHF()
#es_mf.kernel(mydmet.es_dm)
es_mf = mydmet.es_mf

'''
from pyscf.tools import molden
with open(mydmet.title+' imp_rohf_orbs.molden', 'w')as f1:
    molden.header(MF.mol, f1)
    molden.orbital_coeff(MF.mol, f1, mydmet.es_orb @ es_mf.mo_coeff, ene=es_mf.mo_energy, occ=es_mf.mo_occ)


'''
#ncas, nelec, es_mo = mydmet.avas(['Ce 4f','Ce 5d'], minao=CLUS_MOL._basis['Ce'], threshold=0.5, openshell_option=3)

title = 'Ce'
ncas, nelec, es_mo = mydmet.avas(['Ce 4f','Ce 5d'], minao=CLUS_MOL._basis['Ce'], threshold=0.5, openshell_option=2)
ncas=12
nelec=1

es_cas = sacasscf_mixer.sacasscf_mixer(es_mf, ncas, nelec)
es_cas.kernel(es_mo)
#===================PT2========================
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas, method='SC')
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
#===================PT2========================
Ha2cm = 219474.63
np.savetxt(mydmet.title+'_opt.txt',(es_cas.fcisolver.e_states-np.min(es_cas.fcisolver.e_states))*Ha2cm,fmt='%.6f')
total_cas = mydmet.total_cas(es_cas)

mysiso = siso.SISO(title, total_cas, verbose=6).density_fit()
mysiso.kernel()