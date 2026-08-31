import numpy as np
from pyscf import lib, gto
def calc_fo_ene(dmet, e_nuc = True):
    # energy of frozen occupied orbitals and nuclear-nuclear repulsion
    dm_fo = dmet.fo_orb @ dmet.fo_orb.T.conj()*2

    h1e = dmet.mf_or_cas.get_hcore()
    if isinstance(dm_fo, np.ndarray) and dm_fo.ndim == 2:
        dm_fo = np.array((dm_fo*.5, dm_fo*.5))
    # get_veff in casci and rohf differ by a factor 2: rohf.get_veff = casci.get_veff * 2
    # we manually build vhf
    vj, vk = dmet.mf_or_cas.get_jk(dmet.mol, dm_fo)
    vhf = vj[0] + vj[1] - vk
    
    if h1e[0].ndim < dm_fo[0].ndim:  # get [0] because h1e and dm may not be ndarrays
        h1e = (h1e, h1e)
    e1 = lib.einsum('ij,ji->', h1e[0], dm_fo[0])
    e1+= lib.einsum('ij,ji->', h1e[1], dm_fo[1])
    e_coul =(lib.einsum('ij,ji->', vhf[0], dm_fo[0]) +
            lib.einsum('ij,ji->', vhf[1], dm_fo[1])) * .5
    e_elec = (e1 + e_coul).real
    fo_ene = e_elec
    if e_nuc:
        e_nuc = dmet.mf_or_cas.energy_nuc()
        fo_ene += e_nuc
    dmet.fo_ene = fo_ene
    return fo_ene  
def concentric_localization(dmet, proj_bas, n_shell, atoms_A, couple_op='hcore', ele_density = False, threshold=1e-6):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")
    if atoms_A is None or len(atoms_A) == 0:
        raise ValueError("atoms_A must be a non-empty list of atom indices")

    atoms_A = [int(i) for i in atoms_A]
    natm = dmet.mol.natm
    if any(i < 0 or i >= natm for i in atoms_A):
        raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

    # Build a fake molecule from selected atoms with a small projection basis.
    fake_mol = gto.Mole()
    fake_mol.verbose = dmet.verbose
    fake_mol.unit = 'Bohr'
    fake_mol.symmetry = False
    fake_mol.atom = [(dmet.mol.atom_symbol(i), dmet.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
    fake_mol.basis = proj_bas
    fake_mol.spin = 0 if fake_mol.nelectron % 2 == 0 else 1
    fake_mol.charge = 0 # here may be error for the sake of open shell systems
    fake_mol.build(False, False)
    
    slices = fake_mol.aoslice_by_atom()
    for ia in range(fake_mol.natm):
        ao_start = slices[ia][2]
        ao_end = slices[ia][3]
        num_ao = ao_end - ao_start  # AO number for this atom
    
        symbol = fake_mol.atom_symbol(ia)
        print(f" atom {ia} ({symbol}) with {num_ao:2d} basis functions from ({ao_start:2d} to {ao_end-1:2d})")

    print(f"Total basis functions (fake_mol.nao): {fake_mol.nao}")

    print(f"Fake molecule built with {fake_mol.natm} atoms and {fake_mol.nao} AOs.")
    print(f"Fake molecule: {fake_mol.atom}")
    #s_wb = self.mol.intor_symmetric('int1e_ovlp')
    s_pb = fake_mol.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake_mol, dmet.mol) 
    s_pb_inv = np.linalg.inv(s_pb)
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    fv_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath+dmet.nfo :]

    def svd(coeff, couple_op, coeff_ker):
        ovlp = coeff.T.conj() @ couple_op @ coeff_ker #  ovlp may use other types of coupling operators, not limited to h_core
        # MUST use full_matrices=True here to get the complete right singular vectors 

        U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
        dmet.log.info(f"Singular values for shell {i}: {sigma}")
        r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span  = Vh[:r, :].T.conj()
        V_ker = Vh[r:, :].T.conj()
        coeff_n1 = coeff_ker @ V_span
        coeff_ker1 = coeff_ker @ V_ker
        return coeff_n1, coeff_ker1
    c_fv_prime = s_pb_inv @ s_cross @ fv_AO
    # space sizes
    dmet.log.info(f"fv_AO shape: {fv_AO.shape}")
    dmet.log.info(f"s_cross shape: {s_cross.shape}")
    dmet.log.info(f"c_fv_prime shape: {c_fv_prime.shape}")

    U, sigma, Vh = np.linalg.svd(c_fv_prime.T.conj() @ s_cross @ fv_AO, full_matrices=True)
    r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
    dmet.log.info(f"SVD on projector rank r0: {r0}, sigma size: {len(sigma)}")
    dmet.log.info(f"Singular values: {sigma}")
    V_span = Vh[:r0, :].T.conj()
    V_ker  = Vh[r0:, :].T.conj()
    C_0 = fv_AO @ V_span
    C_ker0 = fv_AO @ V_ker
    C_vir = []
    C_vir.append(C_0)
    C_ker = []
    C_ker.append(C_ker0)

    # pseudo-canonicalize the vir space by diagonalizing the Fock matrix in this subspace
    dm = dmet.mf_or_cas.make_rdm1()
    fock_ao = dmet.mf_or_cas.get_fock(dm=dm) 

    dmet.log.info(f"Shell 0: {C_0.shape[1]} vectors in vir space, {C_ker0.shape[1]} vectors in ker space.")
    for i in range(n_shell):
        if couple_op == 'hcore':
            couple_matrix = dmet.mf_or_cas.get_hcore()
            dmet.log.info(f"Using Hcore as coupling operator for shell {i+1}")
        elif couple_op == 'fock':
            couple_matrix = fock_ao
            dmet.log.info(f"Using Fock matrix as coupling operator for shell {i+1}")
        #elif couple_op == ''
        new_vir, new_ker = svd(C_vir[i], couple_matrix, C_ker[i])  
        C_vir.append(new_vir)
        C_ker.append(new_ker)
        dmet.log.info(f"Shell {i+1}: {new_vir.shape[1]} new vectors added to vir space, {new_ker.shape[1]} vectors remain in ker space.")
        dmet.log.info(f"======Shell {i+1} concentric localized======")
    C_vir_matrix = np.hstack(C_vir)
    
    # Export density cube files for each shell. Noting that the density is calculated considering the orbs are occupied just for visualization.
    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_vir):
                if c_shell.shape[1] > 0:
                    dm_shell = 2.0 * (c_shell @ c_shell.T.conj())
                    cube_name = f"{dmet.title}_shell_{idx}_density.cube"
                    dmet.log.info(f"Exporting electron density of Shell {idx} ({c_shell.shape[1]} orbitals) to {cube_name}")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    # Project Fock matrix into the vir subspace defined by C_vir_matrix
    # (N_vir, N_AO) @ (N_AO, N_AO) @ (N_AO, N_vir) -> (N_vir, N_vir)
    fock_sub = C_vir_matrix.T.conj() @ fock_ao @ C_vir_matrix

    # diagonalize the Fock matrix
    mo_energy, U = np.linalg.eigh(fock_sub)
    C_vir_canonical = C_vir_matrix @ U

    C_fv_new = C_ker[-1]
    fock_fv = C_fv_new.T.conj() @ fock_ao @ C_fv_new
    mo_energy_fv, U_fv = np.linalg.eigh(fock_fv)
    C_fv_canonical = C_fv_new @ U_fv

    Q_emb = dmet.lo_cloes[:, :nimp+nbath]
    Q_fo  = dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]
    
    lo2New_bath = dmet.cloao @ C_vir_canonical
    lo2New_fv   = dmet.cloao @ C_fv_canonical
    
    dmet.lo_cloes = np.hstack([Q_emb, lo2New_bath, Q_fo, lo2New_fv])
    
    n_shifted_fv = lo2New_bath.shape[1]
    dmet.nes += n_shifted_fv
    dmet.nfv -= n_shifted_fv
    
    dmet.es_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, :dmet.nes])
    dmet.fo_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes : dmet.nes+dmet.nfo])
    dmet.fv_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes+dmet.nfo :])

    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi'):
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()
    dm_arg = dmet.dm_pair if (dmet.open_shell and dmet.dm_pair is not None) else dmet.dm
    dmet.es_dm = dmet.make_es_dm(dmet.open_shell, dmet.lo_cloes[:, :dmet.nes], dmet.cloao, dm_arg)

    dmet.es_mf = dmet.ROHF()
    calc_fo_ene(dmet, e_nuc=True)

    dmet.log.info(f"Concentric Shell appended. Added {n_shifted_fv} vir orbitals to bath.")
    dmet.log.info(f"New sizes: NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; NImp={nimp}, NBATH = {dmet.nes - nimp}")
    return dmet
'''
    def concentric_occ_spade(self, atoms_A, threshold=1e-6):
        nimp = len(dmet.imp_idx)
        nbath = dmet.nes - nimp
        c_fo_lo = dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]
        c_fo_ao = dmet.caolo @ c_fo_lo
        ao_indices_A = []
        for ia in atoms_A:
            atom_id, atom_symbol, start, end = dmet.mol.aoslice_by_atom()[ia]
            ao_indices_A.extend(range(start, end))
        Q_A = np.zeros((dmet.mol.nao, dmet.mol.nao))
        for idx in ao_indices_A:
            Q_A[idx, idx] = 1.0
        
        c_fo_lo_A = Q_A @ c_fo_lo
        u_A, sigma_A, vh_A = np.linalg.svd(c_fo_lo_A, full_matrices=True)
        print(f"Shape of c_fo_lo_A: {c_fo_lo_A.shape}")
        print(f"Singular values for c_fo_lo_A: {sigma_A}")
        print(f"U_A shape: {u_A.shape}, vh_A shape: {vh_A.shape}")
        print(f"Shape of V_A: {vh_A.T.conj().shape}")
        C_spade = c_fo_lo @ vh_A.T.conj()
        print(f"Shape of C_spade: {C_spade.shape}")
        
'''
def concentric_occ_localization(dmet, proj_bas, n_shell, atoms_A, couple_op='hcore', ele_density = False, threshold=1e-6):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")
    if atoms_A is None or len(atoms_A) == 0:
        raise ValueError("atoms_A must be a non-empty list of atom indices")

    atoms_A = [int(i) for i in atoms_A]
    natm = dmet.mol.natm
    if any(i < 0 or i >= natm for i in atoms_A):
        raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

    # Build a fake molecule from selected atoms with a small projection basis.
    fake_mol = gto.Mole()
    fake_mol.verbose = dmet.verbose
    fake_mol.unit = 'Bohr'
    fake_mol.symmetry = False
    fake_mol.atom = [(dmet.mol.atom_symbol(i), dmet.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
    fake_mol.basis = proj_bas
    fake_mol.spin = 0
    fake_mol.charge = 0 # here may be error for the sake of open shell systems
    if fake_mol.nelectron % 2 != 0:
        fake_mol.spin = 1 # a compromise but not affect the basis of fakemol
    fake_mol.build(False, False)

    slices = fake_mol.aoslice_by_atom()
    for ia in range(fake_mol.natm):
        ao_start = slices[ia][2]
        ao_end = slices[ia][3]
        num_ao = ao_end - ao_start  # AO number for this atom
    
        symbol = fake_mol.atom_symbol(ia)
        print(f" atom {ia} ({symbol}) with {num_ao:2d} basis functions from ({ao_start:2d} to {ao_end-1:2d})")

    print(f"Total basis functions (fake_mol.nao): {fake_mol.nao}")
    print(f"Fake molecule built with {fake_mol.natm} atoms and {fake_mol.nao} AOs.")
    print(f"Fake molecule: {fake_mol.atom}")

    s_pb = fake_mol.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake_mol, dmet.mol) 
    s_pb_inv = np.linalg.inv(s_pb)
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    # NOTE: Here we operate on the frozen occupied (FO) orbitals instead of FV
    fo_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]

    def svd(coeff, couple_op, coeff_ker):
        ovlp = coeff.T.conj() @ couple_op @ coeff_ker
        U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
        r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span  = Vh[:r, :].T.conj()
        V_ker = Vh[r:, :].T.conj()
        coeff_n1 = coeff_ker @ V_span
        coeff_ker1 = coeff_ker @ V_ker
        return coeff_n1, coeff_ker1
        
    c_fo_prime = s_pb_inv @ s_cross @ fo_AO

    U, sigma, Vh = np.linalg.svd(c_fo_prime.T.conj() @ s_cross @ fo_AO, full_matrices=True)
    r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
    V_span = Vh[:r0, :].T.conj()
    V_ker  = Vh[r0:, :].T.conj()
    C_0 = fo_AO @ V_span
    C_ker0 = fo_AO @ V_ker
    C_occ = [C_0]
    C_ker = [C_ker0]

    dm = dmet.mf_or_cas.make_rdm1()
    fock_ao = dmet.mf_or_cas.get_fock(dm=dm) 

    for i in range(n_shell):
        if couple_op == 'hcore':
            couple_matrix = dmet.mf_or_cas.get_hcore()
        elif couple_op == 'fock':
            couple_matrix = fock_ao
            
        new_occ, new_ker = svd(C_occ[i], couple_matrix, C_ker[i])  
        C_occ.append(new_occ)
        C_ker.append(new_ker)
        print(f"Shell {i+1}: {new_occ.shape[1]} new vectors added to occ space")
        
    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_occ):
                if c_shell.shape[1] > 0:
                    dm_shell = 2.0 * (c_shell @ c_shell.T.conj())
                    cube_name = f"{dmet.title}_shell_{idx}_density_occ.cube"
                    dmet.log.info(f"Exporting electron density of Shell {idx} ({c_shell.shape[1]} orbitals) to {cube_name}")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    C_occ_matrix = np.hstack(C_occ)
    
    fock_sub = C_occ_matrix.T.conj() @ fock_ao @ C_occ_matrix
    mo_energy, U = np.linalg.eigh(fock_sub)
    C_occ_canonical = C_occ_matrix @ U

    C_fo_new = C_ker[-1]
    fock_fo = C_fo_new.T.conj() @ fock_ao @ C_fo_new
    mo_energy_fo, U_fo = np.linalg.eigh(fock_fo)
    C_fo_canonical = C_fo_new @ U_fo

    Q_emb = dmet.lo_cloes[:, :nimp+nbath]
    Q_fv  = dmet.lo_cloes[:, nimp+nbath+dmet.nfo :]
    
    lo2New_bath = dmet.cloao @ C_occ_canonical
    lo2New_fo   = dmet.cloao @ C_fo_canonical
    
    # Reassemble logic for FO
    # Sequence: [Emb, Target_FO (shifted to bath), Remaining_FO, FV]
    dmet.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_fo, Q_fv])
    
    n_shifted_fo = lo2New_bath.shape[1]
    dmet.nes += n_shifted_fo
    dmet.nfo -= n_shifted_fo
    
    dmet.es_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, :dmet.nes])
    dmet.fo_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes : dmet.nes+dmet.nfo])
    dmet.fv_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes+dmet.nfo :])

    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi'):
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()
    dm_arg = dmet.dm_pair if (dmet.open_shell and dmet.dm_pair is not None) else dmet.dm
    dmet.es_dm = dmet.make_es_dm(dmet.open_shell, dmet.lo_cloes[:, :dmet.nes], dmet.cloao, dm_arg)
    
    dmet.es_mf = dmet.ROHF()
    calc_fo_ene(dmet, e_nuc=True)
    
    dmet.log.info(f"Concentric FO Shell appended. Added {n_shifted_fo} occ orbitals to bath.")
    dmet.log.info(f"New sizes: NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; NImp={nimp}, NBATH = {dmet.nes - nimp}")

    return dmet

def localize_environment_spaces(dmet, method='boys'):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")            
    dmet.log.info(f"Performing {method.upper()} localization on Env subspaces (Bath, FO, FV)...")
    
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    # MO coeff in AO bases
    bath_AO = dmet.caolo @ dmet.lo_cloes[:, nimp : nimp+nbath]
    fo_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]
    fv_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath+dmet.nfo :]
    # note that, bath space may have some occ orbitals, may be some issues.
    from pyscf import lo
    def localize_subspace(coeff_AO, name):
        if coeff_AO.shape[1] <= 1:
            return coeff_AO  
        try:
            if method.lower() == 'boys':
                loc_obj = lo.Boys(dmet.mol, coeff_AO)
            elif method.lower() == 'pm':
                loc_obj = lo.PipekMezey(dmet.mol, coeff_AO)
            elif method.lower() == 'er':
                loc_obj = lo.EdmistonRuedenberg(dmet.mol, coeff_AO)
            else:
                dmet.log.warn(f"Unknown localization method {method}, skipping localization for {name}.")
                return coeff_AO
                
            loc_obj.verbose = 0
            return loc_obj.kernel()
        except Exception as e:
            dmet.log.warn(f"Localization failed for {name} subspace using {method}: {str(e)}")
            return coeff_AO
    
    bath_loc_AO = localize_subspace(bath_AO, "Bath")
    fo_loc_AO   = localize_subspace(fo_AO, "FO")
    fv_loc_AO   = localize_subspace(fv_AO, "FV")
    # from AO basis to LO basis
    dmet.lo_cloes[:, nimp : nimp+nbath] = dmet.cloao @ bath_loc_AO
    dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo] = dmet.cloao @ fo_loc_AO
    dmet.lo_cloes[:, nimp+nbath+dmet.nfo :] = dmet.cloao @ fv_loc_AO
    # and refresh the fo fv orbitals in AO basis
    dmet.es_orb = dmet.caolo @ dmet.lo_cloes[:, :dmet.nes]
    dmet.fo_orb = fo_loc_AO
    dmet.fv_orb = fv_loc_AO
    
    dmet.log.info("Environment subspaces localized successfully.")
    return dmet

