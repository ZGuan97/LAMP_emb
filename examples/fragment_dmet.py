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
    basis={'default':'ccpvtz','S':'ccpvdz','H':'ccpvdz'}, symmetry=0 ,spin = 3,charge = -2,verbose= 4)



# Atom order in the Z-matrix above:
#   0 Co
#   1 S(L1), 2 S(L2), 3 S(L3), 4 S(L4)
#   5 H(L1), 6 H(L3), 7 H(L4), 8 H(L2)
#
# Impurity and ligand fragments are defined with PySCF AO labels.
impurity = {
    'name': 'Co',
    'aolabels': 'Co.*',
    'charge': 2,
}

ligands = [
    {'name': 'L1', 'aolabels': ['1 S.*', '5 H.*'], 'charge': -1},
    {'name': 'L2', 'aolabels': ['2 S.*', '8 H.*'], 'charge': -1},
    {'name': 'L3', 'aolabels': ['3 S.*', '6 H.*'], 'charge': -1},
    {'name': 'L4', 'aolabels': ['4 S.*', '7 H.*'], 'charge': -1},
]

fragments = [ligand['aolabels'] for ligand in ligands]
fragment_charges = [ligand['charge'] for ligand in ligands]

mydmet = fragment.FDMET(
    mol,
    title=title,
    imp_idx=impurity['aolabels'],
    imp_charge=impurity['charge'],
    fragments=fragments,
    fragment_charges=fragment_charges,
    fragment_scf='cahf',
    fragment_scf_options={
        'ncas': 5,
        'nelecas': 7,
        'cahf_spin': 3,
        'diis': 'rdiis',
        'rdiis_prop': 'dS',
        'rdiis_imp_idx': ['Co.*d'],
        'rdiis_power': 0.2,
        'max_cycle': 200,
        'level_shift': 2.0,
    },
)
mydmet.build(fragment_scf_verbose=3)

ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='ccpvtz', threshold=0.5)

es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
es_cas.kernel(es_mo)

es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
mysiso = siso.SISO(title, total_cas)
mysiso.kernel()
