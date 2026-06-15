import numpy as np
from pyscf import gto, scf
from embed_sim import sacasscf_mixer, siso, fragment

title = 'CoSH4'

mol = gto.M(atom = '''
        Co             
        S                  1            2.30186590
        S                  1            2.30186590    2            109.47122060
        S                  1            2.30186590    3            109.47122065    2            -120.00000001                  0
        S                  1            2.30186590    4            109.47122060    3            120.00000001                   0
        H                  2            1.30714645    1            109.47121982    4            -60                            0
        H                  4            1.30714645    1            109.47121982    3            60                             0
        H                  5            1.30714645    1            109.47121982    4            -180                           0
        H                  3            1.30714645    1            109.47121982    4            60                             0
    ''',
    basis={'default':'def2tzvp','s':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 3,charge = -2,verbose= 4)



# Atom order in the Z-matrix above:
#   0 Co
#   1 S(L1), 2 S(L2), 3 S(L3), 4 S(L4)
#   5 H(L1), 6 H(L3), 7 H(L4), 8 H(L2)
#
# Impurity and ligand fragments are defined by atom IDs.
impurity = {
    'name': 'Co',
    'atoms': 'Co',
    'charge': 2,
}

# ligands = [
#     {'name': 'L1', 'atoms': [1, 5], 'charge': -1},
#     {'name': 'L2', 'atoms': [2, 8], 'charge': -1},
#     {'name': 'L3', 'atoms': [3, 6], 'charge': -1},
#     {'name': 'L4', 'atoms': [4, 7], 'charge': -1},
# ]

ligands = [
    {'name': 'L1', 'atoms': [1, 5, 2, 8], 'charge': -2},
    {'name': 'L2', 'atoms': [3, 6, 4, 7], 'charge': -2},
]

ligand_atoms = [ligand['atoms'] for ligand in ligands]
ligand_charges = [ligand['charge'] for ligand in ligands]

mydmet = fragment.FDMET(
    mol,
    title=title,
    imp_atoms=impurity['atoms'],
    imp_charge=impurity['charge'],
    ligand_atoms=ligand_atoms,
    ligand_charges=ligand_charges,
    fragment_scf='cahf',
    keep_fv_orbitals=False,
    embedded_init_guess='fragment_density',
    embedded_active_aolabels='Co 3d',
    fragment_scf_options={
        'ncas': 5,
        'nelecas': 7,
        'cahf_spin': 3,
        'diis': 'rdiis',
        'rdiis_prop': 'dS',
        'rdiis_imp_idx': ['Co.*d'],
        'rdiis_power': 0.2,
        'max_cycle': 500,
        'level_shift': 4.0,
    },
)
mydmet.build(verbose=3)
mydmet.es_mf.run()

fv0 = len(mydmet.imp_idx) + mydmet.nbath + mydmet.nappended_fo
fv1 = fv0 + mydmet.nkept_fv
mo = mydmet.es_mf.mo_coeff
mo_occ = mydmet.es_mf.mo_occ
fv_weight = np.sum(np.abs(mo[fv0:fv1, :])**2, axis=0)
occ_mask = mo_occ > 1e-8
ncas = mydmet.fragment_scf_options['ncas']
nelecas = mydmet.fragment_scf_options['nelecas']
ncore = (mydmet.es_mf.mol.nelectron - nelecas) // 2
active_slice = slice(ncore, ncore+ncas)
dm = mydmet.es_mf.make_rdm1(mo, mo_occ)
if dm.ndim == 3:
    dm = dm[0] + dm[1]

print('\n=== FV projection diagnostic ===')
print('embedded dimensions: nimp = %d, nbath = %d, nappended_fo = %d, nkept_fv = %d, nes = %d' %
      (len(mydmet.imp_idx), mydmet.nbath, mydmet.nappended_fo,
       mydmet.nkept_fv, mydmet.nes))
print('max MO FV weight = %.12e at MO %d' %
      (fv_weight.max(), int(np.argmax(fv_weight))))
print('max occupied/active MO FV weight = %.12e' %
      fv_weight[occ_mask].max())
print('sum occupied/active occ-weighted FV weight = %.12e' %
      np.dot(mo_occ, fv_weight))
print('sum active MO FV weight = %.12e' %
      np.sum(fv_weight[active_slice]))
print('density trace on kept-FV block = %.12e' %
      np.trace(dm[fv0:fv1, fv0:fv1]).real)
print('largest 10 occupied/active FV weights:')
for idx in np.argsort(fv_weight[occ_mask])[-10:][::-1]:
    mo_idx = np.nonzero(occ_mask)[0][idx]
    print('  MO %4d occ %12.8f FV weight %.12e' %
          (mo_idx, mo_occ[mo_idx], fv_weight[mo_idx]))

# ncas, nelec, es_mo = mydmet.avas('Co 3d', threshold=0.5)

es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas=5, nelec=7, statelis=[0,0,0,10])
es_cas.kernel()
raise
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
mysiso = siso.SISO(title, total_cas)
mysiso.kernel()
