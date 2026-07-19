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
    keep_fv_orbitals=False,
    embedded_init_guess='fragment_density',
    embedded_active_aolabels='Co 3d',
    fragment_scf_options={
        'ncas': 5,
        'nelecas': 7,
        'cahf_spin': 1,
        'init_guess': 'atom',
        'pre_scf_max_cycle': 0,
        'avas_aolabels': ['Co 3d'],
        'avas_threshold': 0.5,
        'max_cycle': 200,
        'conv_tol': 1e-9,
        'level_shift': 4.0,
        'diis': 'rdiis',
        'rdiis_prop': 'dS',
        'rdiis_imp_idx': ['Co.*d'],
        'rdiis_power': 0.2,
    },
)
mydmet.build(verbose=4)
mydmet.es_mf.run()

print('\n=== Rebuilding embedded space from CAHF density ===')
old_nbath = mydmet.nbath
old_nappended_fo = mydmet.nappended_fo
old_nes = mydmet.nes
mydmet.rebuild_from_embedded_density()
print('nbath:        %d -> %d' % (old_nbath, mydmet.nbath))
print('nappended_fo: %d -> %d' % (old_nappended_fo, mydmet.nappended_fo))
print('nes:          %d -> %d' % (old_nes, mydmet.nes))
mydmet.es_mf.run()
print('E_rebuilt = %.12f' % mydmet.es_mf.e_tot)

es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas=5, nelec=7, statelis=[0,40,0,10])
es_cas.kernel()
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
mysiso = siso.SISO(title, total_cas)
mysiso.kernel()
