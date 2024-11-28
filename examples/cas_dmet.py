import numpy as np
from pyscf import scf
from embed_sim import ssdmet, myavas, sacasscf_mixer, siso

title = 'CoSPh4'
    
from pyscf import gto
mol = gto.M(atom='''
        Co   1.506499   10.764206    0.414024
        S    0.900702   12.969251    0.863740
        S   -0.630821    9.850987    0.263627
        S    2.753892   10.415545   -1.502746
        S    3.027943    9.878195    1.923416
        C    2.377311   13.975173    0.913216
        C    2.276518   15.365718    0.836418
        H    1.444112   15.762234    0.713835
        C    3.399137   16.154649    0.937832
        H    3.312247   17.078944    0.888601
        C    4.645140   15.606362    1.112844
        H    5.397607   16.149181    1.179058
        C    4.758096   14.244532    1.186689
        H    5.597454   13.867157    1.321825
        C    3.649380   13.407744    1.065829
        H    3.753648   12.483449    1.085522
        C   -0.483108    8.032887    0.187074
        C   -1.603989    7.293178    0.071383
        H   -2.422493    7.725245   -0.019692
        C   -1.574447    5.919042    0.086153
        H   -2.366884    5.440487    0.000000
        C   -0.364938    5.235392    0.231381
        H   -0.335395    4.306995    0.285534
        C    0.776797    5.988774    0.292919
        H    1.598776    5.558075    0.347072
        C    0.738565    7.384787    0.273226
        H    1.527526    7.875648    0.319995
        C    2.008897   11.246043   -2.857063
        C    0.792437   11.947467   -2.723896
        H    0.359725   11.969345   -1.900278
        C    0.238079   12.605139   -3.809910
        H   -0.559572   13.071388   -3.704557
        C    0.851522   12.577793   -5.050260
        H    0.465730   13.018063   -5.772218
        C    2.036702   11.896878   -5.208288
        H    2.450298   11.872266   -6.040521
        C    2.615389   11.247410   -4.127935
        H    3.423466   10.801671   -4.246087
        C    2.286945    9.211500    3.375947
        C    3.044626    9.114422    4.556237
        H    3.906575    9.464451    4.578390
        C    2.530237    8.503239    5.693449
        H    3.049839    8.432139    6.461437
        C    1.239051    8.001440    5.679173
        H    0.886278    7.585780    6.431899
        C    0.479633    8.125864    4.519314
        H   -0.391005    7.797712    4.514391
        C    0.962741    8.709702    3.404255
        H    0.424023    8.782168    2.648574
    ''',
    basis={'default':'def2tzvp','C':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 3,charge = -2,verbose= 4)

mf = scf.rohf.ROHF(mol).x2c().density_fit()
chk_fname = title + '_rohf.chk'

mf.chkfile = chk_fname
mf.init_guess = 'chk'
mf.level_shift = .1
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()

# DMET from ROHF CAS(10o, 7e)
print('******')
print('DMET from ROHF CAS(10o, 7e)')
print('******')
hf_dmet = ssdmet.SSDMET(mf, title=title, imp_idx='Co *')
hf_dmet.build(chk_fname_load=title + '_dmet_chk.h5')
ncas, nelec, es_mo = hf_dmet.avas(['Co 3d', 'Co 4d'], minao='def2tzvp', threshold=0.5)
es_cas = sacasscf_mixer.sacasscf_mixer(hf_dmet.es_mf, ncas, nelec, statelis=[0, 40, 0, 10])
es_cas.kernel(es_mo)

hf_dmet_pre_soc_ene = np.sort(es_cas.e_states) - np.min(es_cas.e_states)
print('ROHF DMET pre SOC energy', hf_dmet_pre_soc_ene)

# All-electron CASSCF CAS(10o, 7e)
print('******')
print('All-electron CASSCF CAS(10o, 7e)')
print('******')
ncas, nelec, mo = myavas.avas(mf, ['Co 3d', 'Co 4d'])
mycas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec, statelis=[0, 40, 0, 10])
cas_chk = title + '_cas.chk'
mycas.chkfile = cas_chk

from pyscf import lib
try:
    mo = lib.chkfile.load(cas_chk, 'mcscf/mo_coeff')
except IOError:
    pass
mycas.natorb = True
mycas.kernel(mo)
ae_pre_soc_ene = np.sort(mycas.e_states) - np.min(mycas.e_states)
print('All-electron pre SOC energy', ae_pre_soc_ene)

# All-electron CASSCF CAS(5o, 7e)
print('******')
print('All-electron CASSCF CAS(5o, 7e)')
print('******')
ncas, nelec, mo = myavas.avas(mf, 'Co 3d', canonicalize=False)
mycas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec, statelis=[0, 40, 0, 10])
cas_chk = title + '_cas_5_7.chk'
mycas.chkfile = cas_chk

from pyscf import lib
try:
    mo = lib.chkfile.load(cas_chk, 'mcscf/mo_coeff')
except IOError:
    pass
mycas.natorb = True
mycas.kernel(mo)
ae_pre_soc_ene = np.sort(mycas.e_states) - np.min(mycas.e_states)
print('All-electron (5,7) pre SOC energy', ae_pre_soc_ene)

# DMET from CAS(5o, 7e) CAS(10o, 7e)
print('******')
print('DMET from CAS(5o, 7e) CAS(10o, 7e)')
print('******')
cas_dmet = ssdmet.SSDMET(mycas, title=title+'_cas', imp_idx='Co *')
cas_dmet.build()
ncas, nelec, es_mo = cas_dmet.avas(['Co 3d', 'Co 4d'], minao='def2tzvp', threshold=0.5)

es_cas = sacasscf_mixer.sacasscf_mixer(cas_dmet.es_mf, ncas, nelec, statelis=[0, 40, 0, 10])
es_cas.kernel(es_mo)

cas_dmet_pre_soc_ene = np.sort(es_cas.e_states) - np.min(es_cas.e_states)
print('CAS DMET pre SOC energy', cas_dmet_pre_soc_ene)