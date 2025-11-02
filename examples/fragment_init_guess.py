from pyscf import gto, scf, lib
from embed_sim import ssdmet
import basis_set_exchange as bse
import scipy
import numpy
from functools import reduce
from scipy.linalg import block_diag

def generate_fragment_guess(mol, fragments_info):
    lib.logger.info(mol, "--- Initial guess by frag ---")

    all_frag_indices = []
    all_frag_atom_specs = []

    unassigned_atom_indices = set(range(mol.natm))

    for i, frag_info in enumerate(fragments_info):
        frag_symbols = frag_info['symbols']
        target_symbols_set = {s.strip().capitalize() for s in frag_symbols}

        current_frag_indices = set()
        for atom_idx in list(unassigned_atom_indices):
            atom_entry = mol.atom[atom_idx]
            symbol = atom_entry[0].strip().capitalize()
            if symbol in target_symbols_set:
                current_frag_indices.add(atom_idx)

        if not current_frag_indices:
            raise ValueError(f"Frag {i} (symbol: {frag_symbols}) couldn't find any atom that can be matched with")

        lib.logger.info(mol, 'Frag %d (symbol: %s) Atom index: %s', i, frag_symbols, sorted(current_frag_indices))

        sorted_indices = sorted(list(current_frag_indices))
        all_frag_indices.append(sorted_indices)
        unassigned_atom_indices -= current_frag_indices

    if unassigned_atom_indices:
        raise ValueError(f"Error: These atoms are not assigned to any frag: {unassigned_atom_indices}")
    lib.logger.info(mol, '\nAll atoms have been assigned to certain frag')

    mf_fragments = []
    total_spin_check = 0
    total_charge_check = 0

    mf_dm = []
    for i, (frag_info, frag_indices) in enumerate(zip(fragments_info, all_frag_indices)):
        lib.logger.info(mol, '\n--- Processing frag %d ---', i)
        frag_spin = frag_info['spin']
        frag_charge = frag_info['charge']
        total_spin_check += frag_spin
        total_charge_check += frag_charge

        frag_atom_spec = []
        for j in frag_indices:
            atom_entry = mol.atom[j]
            symbol = atom_entry[0]
            coord_original = numpy.array(atom_entry[1])
            frag_atom_spec.append([symbol, coord_original.tolist()])
        
        all_frag_atom_specs.append(frag_atom_spec)

        frag_mol = gto.Mole()
        frag_mol.atom = frag_atom_spec
        frag_mol.basis = mol.basis
        frag_mol.spin = frag_spin
        frag_mol.charge = frag_charge
        frag_mol.build()

        lib.logger.info(mol, '\n Carrying out ROHF calculations for frag %d ...", i)
        mf = scf.rohf.ROHF(frag_mol).x2c().density_fit()
        mf.conv_tol = 0.1
        mf.level_shift = 1
        mf.conv_check = False
        mf.run(init_guess='minao', verbose=4)
        lib.logger.info(mol, 'Calculation for frag %d completed. E = %.8f', i, mf.e_tot)
        dma,dmb = mf.make_rdm1()
        dm = dma + dmb
        mf_dm.append(dm)
        mf_fragments.append(mf)

    if total_charge_check != mol.charge or total_spin_check != mol.spin:
        lib.logger.warn(mol, 'Warning: Sum of frag charge/frag spin ≠ system charge/spin!')
        lib.logger.info(mol, 'Sum of frag charge: %s, system charge: %s', total_charge_check, mol.charge)
        lib.logger.info(mol, 'Sum of frag spin: %s, system spin: %s', total_spin_check, mol.spin)
    combined_atom_spec = []
    for spec in all_frag_atom_specs:
        for atom in spec:
            symbol = atom[0]
            coord_original = numpy.array(atom[1])
            combined_atom_spec.append([symbol, coord_original.tolist()])

    combined_mol = gto.Mole()
    combined_mol.atom = combined_atom_spec
    combined_mol.basis = mol.basis
    combined_mol.spin = mol.spin
    combined_mol.charge = mol.charge
    combined_mol.nucmod = mol.nucmod
    combined_mol.build()
    lib.logger.info(mol, '\nNew molecule has been generated in the order of fragments')
    list_of_mo_coeffs = [mf.mo_coeff for mf in mf_fragments]
    list_of_mo_occs = [mf.mo_occ for mf in mf_fragments]
    dm_guess = block_diag(*mf_dm)
    mo_coeff_guess = block_diag(*list_of_mo_coeffs)
    mo_occ_guess = numpy.concatenate(list_of_mo_occs)

    lib.logger.info(mol, '\nDensity matrix concatenated successfully')

    return combined_mol, dm_guess, mo_coeff_guess, mo_occ_guess

if __name__ == '__main__':
    def read_xyz_robust(filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                num_atoms = int(lines[0].strip())
                atom_coords = []
                for line in lines[2:2+num_atoms]:
                    parts = line.strip().split()
                    symbol = parts[0]
                    coords = tuple(float(c) for c in parts[1:4])
                    atom_coords.append([symbol, coords])
                return atom_coords
        except (IOError, IndexError, ValueError) as e:
            lib.logger.info(mol, "Errors occur when reading '%s': %s", filename, e)
            lib.logger.info(mol, "Please make sure that the file exists and it is in correct format (first row: number of atoms; second row: comment).")
            return None
    xyz_filename = 'DyCpBz.xyz' 
    atom_list = read_xyz_robust(xyz_filename)
    mol = gto.M(atom=atom_list,
            basis={
                'Dy2':'ANO-R2',
                'Dy1':'ANO-R2',
                'C1':'ANO-R1',
                'C0':'ANO-R1',
                'C':'ANO-R0',
                'H':'ANO-R0',
                'H0':'ANO-R1'
                },nucmod='G',
            symmetry=0,spin=10,charge=0,verbose=4)
    title = 'DyCpBz_frag'
    # generate fragment init guess
    new_mol, dm, mo, occ = generate_fragment_guess(mol, fragments_info=[
            {
                'symbols': ['Dy1'], 
                'charge': 3, 
                'spin': 5
            },
            {
                'symbols': ['Dy2'], 
                'charge': 3, 
                'spin': 5
            },
            {
                'symbols': ['C0','H0'], 
                'charge': -4, 
                'spin': 0
            },
            {
                'symbols': ['C1','C','H'], 
                'charge': -2, 
                'spin': 0
            }             
        ])
    new_mol.verbose = 4
    method = 'default'
    # loose convergence
    mf = scf.rohf.ROHF(new_mol).x2c().density_fit()
    mf.level_shift = 1
    mf.conv_tol = 1e-2
    mf.conv_check = False
    mf.chkfile = method+'.chk'
    mf.max_cycle = 1000
    mf.kernel(dm0=dm)
  
    # sSOSCF for each fragment
    mydmet = ssdmet.SSDMET(mf, title=title, imp_idx=['Dy1.*'])
    mydmet.build(save_chk=False)
    es_mf = mydmet.ROHF()
    es_mf = es_mf.newton()
    es_mf.max_cycle = 100
    es_mf.conv_tol = 1e-7
    es_mf.kernel()

    dm_core = mydmet.fo_orb@mydmet.fo_orb.conj().T
    dm = (mydmet.es_orb@es_mf.make_rdm1()[0]@mydmet.es_orb.conj().T+dm_core,
        mydmet.es_orb@es_mf.make_rdm1()[1]@mydmet.es_orb.conj().T+dm_core)
    mf.conv_tol = 1e-6
    mf.max_cycle = 1
    mf.kernel(dm0=dm)

    mydmet = ssdmet.SSDMET(mf, title=title, imp_idx=['Dy2.*'])

    mydmet.build(save_chk=False)
    es_mf = mydmet.ROHF()
    es_mf = es_mf.newton()
    es_mf.max_cycle = 100
    es_mf.conv_tol = 1e-7
    es_mf.kernel()

    dm_core = mydmet.fo_orb@mydmet.fo_orb.conj().T
    dm = (mydmet.es_orb@es_mf.make_rdm1()[0]@mydmet.es_orb.conj().T+dm_core,
        mydmet.es_orb@es_mf.make_rdm1()[1]@mydmet.es_orb.conj().T+dm_core)
    mf.max_cycle = 1
    mf.kernel(dm0=dm)

    mydmet = ssdmet.SSDMET(mf, title=title, imp_idx=['C0.*p'])
    mydmet.build(save_chk=False)
    es_mf = mydmet.ROHF()
    es_mf = es_mf.newton()
    es_mf.max_cycle = 100
    es_mf.conv_tol = 1e-7
    es_mf.kernel()

    dm_core = mydmet.fo_orb@mydmet.fo_orb.conj().T
    dm = (mydmet.es_orb@es_mf.make_rdm1()[0]@mydmet.es_orb.conj().T+dm_core,
        mydmet.es_orb@es_mf.make_rdm1()[1]@mydmet.es_orb.conj().T+dm_core)
    mf.conv_tol = 1e-6
    mf.max_cycle = 1000
    mf.conv_check = False 
    mf.kernel(dm0=dm)
  
    # It is recommended to first converge with a small basis set, then perform calculation with a larger one.
    new_mol.basis = {
                'Dy2':'ANO-R2',
                'Dy1':'ANO-R2',
                'C1':'ANO-R2',
                'C0':'ANO-R2',
                'C':'ANO-R1',
                'H':'ANO-R1',
                'H0':'ANO-R2'
                }
    new_mol.build()
    mf = scf.rohf.ROHF(new_mol).x2c().density_fit()
    mf.level_shift = 1
    mf.conv_tol = 1e-6
    mf.chkfile = method+'.chk'
    mf.init_guess = 'chk'
    mf.max_cycle = 1000
    mf.conv_check = False
    mf.kernel()
