from pyscf import gto, scf
from embed_sim import rdiis, ssdmet
import basis_set_exchange as bse
import numpy as np

mol = gto.M(atom='BCpErCOT.xyz',
        basis={
            'Er':gto.basis.parse(bse.get_basis('ANO-R2',elements='Er',fmt='nwchem')),
            'B' :gto.basis.parse(bse.get_basis('ANO-R2',elements='B',fmt='nwchem')),
            'C' :gto.basis.parse(bse.get_basis('ANO-R1',elements='C',fmt='nwchem')),
            'H' :gto.basis.parse(bse.get_basis('ANO-R1',elements='H',fmt='nwchem'))
            },
        symmetry=0,spin=3,charge=0,nucmod='G',verbose=4)
title = 'BCpErCOT'
mf = scf.rohf.ROHF(mol).x2c().density_fit()
mf.chkfile = title+'.chk'
mf.init_guess = 'atom'
mf.level_shift = 2.0
mf.conv_tol = 1e-2
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Er.*f']),power=0.)
mf.diis.filename = title+'_diis.h5'
mf.max_cycle = 200
mf.conv_check = False
mf.kernel()

mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Er.*')
mydmet.build(save_chk=False)
es_mf = mydmet.ROHF()
es_mf.level_shift = 2.0
es_mf.max_cycle = 0
es_mf.verbose = 0
es_mf.kernel(mydmet.es_dm)
es_mf.mo_occ = es_mf.get_occ()
es_mf = es_mf.newton()
es_mf.verbose = 4
es_mf.max_cycle = 100
es_mf.kernel()

mf.conv_tol = 1e-5
dm_core = mydmet.fo_orb@mydmet.fo_orb.conj().T
dm = (mydmet.es_orb@es_mf.make_rdm1()[0]@mydmet.es_orb.conj().T+dm_core,
      mydmet.es_orb@es_mf.make_rdm1()[1]@mydmet.es_orb.conj().T+dm_core)
mf.max_cycle = 100
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Er.*f']),power=0.)
mf.kernel(dm)

mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Er.*')
mydmet.build(save_chk=False)
es_mf = mydmet.ROHF()
es_mf.level_shift = 2.0
es_mf.max_cycle = 0
es_mf.verbose = 0
es_mf.kernel(mydmet.es_dm)
es_mf.mo_occ = es_mf.get_occ()
es_mf = es_mf.newton()
es_mf.verbose = 4
es_mf.max_cycle = 100
es_mf.kernel()

mf.conv_tol = 1e-8
dm_core = mydmet.fo_orb@mydmet.fo_orb.conj().T
dm = (mydmet.es_orb@es_mf.make_rdm1()[0]@mydmet.es_orb.conj().T+dm_core,
      mydmet.es_orb@es_mf.make_rdm1()[1]@mydmet.es_orb.conj().T+dm_core)
mf.max_cycle = 200
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Er.*f']),power=0.)
mf.conv_check = True
mf.kernel(dm)
