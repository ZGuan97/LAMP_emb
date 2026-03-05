import sys, os
from embed_sim import rdiis

from embed_sim.AIMP3_DMET_SCEI import AIMPEnvLoader, AIMP_RHF, AIMP_RKS, AIMP_ROHF, AIMP_ROKS
from embed_sim.pckit2 import OrganicPCLoader, PointChargeParams
from embed_sim.EnvGenerator import XYZParser
from pyscf import gto, lib
from pyscf.lib import chkfile
import yaml, getopt
import numpy as np
import basis_set_exchange as bse






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

    if scftype in ['HF', 'RHF', 'TDHF', 'TDAHF']:
        MF = AIMP_RHF(CLUS_MOL, AIMP_LOADER)
    elif scftype in ['ROHF']:
        MF = AIMP_ROHF(CLUS_MOL, AIMP_LOADER).density_fit()
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
    MF.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=CLUS_MOL.search_ao_label(['Ce.*']),power=0.2)
    MF.max_cycle = 3000
    MF.conv_tol = 1e-07
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
        MF.init_guess = 'atom'
        print()
        print("Ready for MF.kernel().")
        MF.kernel()

print()

'''
from pyscf.tools import molden
with open('Ce' + '_imp_rohf_orbs.molden', 'w')as f1:
    molden.header(MF.mol, f1)
    molden.orbital_coeff(MF.mol, f1, MF.mo_coeff, ene=MF.mo_energy, occ=MF.mo_occ)


'''
# All-electron CASSCF+NEVPT2
from embed_sim import myavas, sacasscf_mixer, siso

title = 'Ce'
ncas, nelec, mo = myavas.avas(MF, ['Ce 4f','Ce 5d','Ce 6s'], minao=CLUS_MOL._basis['Ce'], threshold=0.5, openshell_option=2)
ncas=13
nelec=1
mycas = sacasscf_mixer.sacasscf_mixer(MF, ncas, nelec, statelis=[0, 13, 0])
mycas.kernel(mo)
Ha2cm = 219474.63
np.savetxt(title+'_cas_NO_SOC.txt',(mycas.fcisolver.e_states-np.min(mycas.fcisolver.e_states))*Ha2cm,fmt='%.6f')


#NVEPT2

ecorr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='SC')
mycas.fcisolver.e_states = mycas.fcisolver.e_states + ecorr
np.savetxt(title+'_nevpt2.txt',ecorr)

Ha2cm = 219474.63
np.savetxt(title+'_opt.txt',(mycas.fcisolver.e_states-np.min(mycas.fcisolver.e_states))*Ha2cm,fmt='%.6f')


mysiso = siso.SISO(title, mycas, amfi=True, verbose=6).density_fit()
mysiso.kernel()
#mysiso.analyze(states=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
