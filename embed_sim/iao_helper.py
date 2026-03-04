'''
IAOPAO generator
Author: Teng Zhang <zhangtchem@stu.pku.edu.cn>
This code is modified from:

QC-DMET: A Python implementation of density matrix embedding theory for
ab initio quantum chemistry.

Copyright (C) 2015 Sebastian Wouters
'''

import re
from pyscf import gto
import numpy as np
import scipy
from pyscf.tools.cubegen import orbital



def subset_by_principal_angles(A, B, atol=1e-12):
    """
    Determine whether span(A) ⊆ span(B) using canonical angles (Euclidean metric).

    Returns:
      is_subset: whether it is a subset
      s:         cosines of the canonical angles (singular values, descending)
      rA, rB:    numerical ranks
    """

    # Orthonormal basis
    QA, _ = np.linalg.qr(A)  # m×kA
    QB, _ = np.linalg.qr(B)  # m×kB
    rA = np.linalg.matrix_rank(A, tol=atol)
    rB = np.linalg.matrix_rank(B, tol=atol)
    QA = QA[:, :rA]
    QB = QB[:, :rB]

    # cos = svd(QA^H QB)
    s = np.linalg.svd(QA.T.conj() @ QB, compute_uv=False)
    # Subset criterion: rA ≤ rB and the first rA singular values s_i ≈ 1
    is_subset = (rA <= rB) and np.allclose(s[:rA], 1.0, atol=10*atol)
    return is_subset, s, rA, rB


def check_full_orthonormal_metric(A, S=None, atol=1e-8):
    """
    Check whether A and B are orthonormal under the given metric S.
    The output format exactly follows the original check_full_orthonormal.
    """

    if S is None:
        S = np.eye(A.shape[0])

    # Metric overlap
    O_A = A.T.conj() @ S @ A
    #O_B = B.T.conj() @ S @ B
    #O_AB = A.T.conj() @ S @ B

    # A itself
    diag_A = np.diag(O_A)
    off_A = O_A - np.diag(diag_A)

    # B itself
    #diag_B = np.diag(O_B)
    #off_B = O_B - np.diag(diag_B)

    # Check conditions
    is_A_normalized = np.allclose(diag_A, 1, atol=atol)
    is_A_orthogonal = np.allclose(off_A, 0, atol=atol)
    #is_B_normalized = np.allclose(diag_B, 1, atol=atol)
    #is_B_orthogonal = np.allclose(off_B, 0, atol=atol)
    #is_AB_orthogonal = np.allclose(O_AB, 0, atol=atol)


    # ---------- Print ----------
    #print("===== A self-check (with metric) =====")
    print("A_normalized", is_A_normalized)
    print("A_orthogonal", is_A_orthogonal)

    #print("\n===== B self-check (with metric) =====")
    #print("Normalization:", is_B_normalized)
    #print("Orthogonality:", is_B_orthogonal)

    #print("\n===== Cross check between A and B (with metric) =====")
    #print("Mutually orthogonal:", is_AB_orthogonal)

    # Deviation metrics
    dev_A = np.linalg.norm(O_A - np.eye(O_A.shape[0]), 'fro')
    #dev_B = np.linalg.norm(O_B - np.eye(O_B.shape[0]), 'fro')
    #dev_AB = np.linalg.norm(O_AB, 'fro')

    print(f"\nA deviation from identity matrix: {dev_A:.3e}")
    #print(f"B deviation from identity matrix: {dev_B:.3e}")
    #print(f"A-B cross deviation: {dev_AB:.3e}")

    return {
        "A_normalized": is_A_normalized,
        "A_orthogonal": is_A_orthogonal,
        #"B_normalized": is_B_normalized,
        #"B_orthogonal": is_B_orthogonal,
        #"A_B_orthogonal": is_AB_orthogonal,
        "A_dev": dev_A,
        #"B_dev": dev_B,
        #"AB_dev": dev_AB
    }

def build_mol_with_mixed_basis(atom_str):

    # Extract the first "atom name (possibly with index)" in each line → extract the letter part → normalize capitalization
    tokens = re.findall(r'^\s*([A-Za-z][A-Za-z0-9]*)', atom_str, flags=re.M)
    elements = set(map(lambda s: re.match(r'[A-Za-z]+', s).group(0).capitalize(), tokens))

    # Construct basis dictionary: default minao; if lanthanides are present override with ANO-R0
    basis = dict.fromkeys(elements, 'minao')
    if 'Ce' in elements:
        basis.update({'Ce': 'ANO-R0'})
    if 'Er' in elements:
        basis.update({'Er': 'ANO-R0'})
    if 'Co' in elements:
        basis.update({'Co': 'ANO-R0'})
    if 'Dy' in elements:
        basis.update({'Dy': 'ANO-R0'})

    return basis


# Return the p_list array indicating positions of atomic orbitals in mol that match those in pmol (1 for match, 0 otherwise)
def construct_p_list( mol, pmol ):

    Norbs       = mol.nao_nr()
    p_list = np.isin(mol.ao_labels(),pmol.ao_labels()).astype(int)
    assert( np.sum( p_list ) == pmol.nao_nr() )
    return p_list


#Lowdin
def orthogonalize_iao( coeff, ovlp ):

    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C, third equation
    eigs, vecs = scipy.linalg.eigh( np.dot( coeff.T, np.dot( ovlp, coeff ) ) )
    coeff      = np.dot( coeff, np.dot( np.dot( vecs, np.diag( np.power( eigs, -0.5 ) ) ), vecs.T ) )
    return coeff


# This code defines a function resort_orbitals to reorder atomic orbitals according to atomic coordinates.
# Specifically, the code matches each orbital with the nearest atom based on spatial position,
# thereby reordering orbitals according to atomic order.
def resort_orbitals( mol, ao2loc ):

    # Sort the orbitals according to the atom list
    Norbs  = mol.nao_nr()
    coords = np.zeros( [ Norbs, 3 ], dtype=float )
    rvec   = mol.intor( 'cint1e_r_sph', 3 )
    for cart in range(3):
        coords[ :, cart ] = np.diag( np.dot( np.dot( ao2loc.T, rvec[cart] ) , ao2loc ) )
    atomid = np.zeros( [ Norbs ], dtype=int )
    for orb in range( Norbs ):
        min_id = 0
        min_distance = np.linalg.norm( coords[ orb, : ] - mol.atom_coord( 0 ) )
        for atom in range( 1, mol.natm ):
            current_distance = np.linalg.norm( coords[ orb, : ] - mol.atom_coord( atom ) )
            if ( current_distance < min_distance ):
                min_distance = current_distance
                min_id = atom
        atomid[ orb ] = min_id
    resort = []
    for atom in range( 0, mol.natm ):
        for orb in range( Norbs ):
            if ( atomid[ orb ] == atom ):
                resort.append( orb )
    resort = np.array( resort )
    ao2loc = ao2loc[ :, resort ]
    return ao2loc
    
def construct_iao( mol, mf ):
    # Number of AOs in mol
    Norbs = mol.nao_nr()

    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C
    ao2occ = mf.mo_coeff[ :, mf.mo_occ > 0.5 ]
    pmol   = mol.copy()
    #Build a new basis on pmol using 'minao' (Minimal Atomic Orbitals)
    if isinstance(pmol.atom, str) and pmol.atom.strip().endswith('.xyz'):
        with open(pmol.atom.strip()) as f:
            lines = f.readlines()
            # Skip the first two lines and extract coordinate data
            atom_data = ''.join(lines[2:]) 
            pmol.atom = atom_data 
    #print(pmol.atom)
    BASIS = build_mol_with_mixed_basis(pmol.atom)
    pmol.build( False, False, basis = BASIS )
    imp_inds_comp = pmol.search_ao_label(['Ce.*'])
    print("This is imp_inds_comp", imp_inds_comp)

    S21    = gto.mole.intor_cross( 'cint1e_ovlp_sph', pmol, mol )
    S1     = mol.intor('cint1e_ovlp_sph')
    S2     = pmol.intor('cint1e_ovlp_sph')
    X      = np.linalg.solve( S2, np.dot( S21, ao2occ ) )
    P12    = np.linalg.solve( S1, S21.T )
    Cp     = np.dot( P12, X )
    Cp     = orthogonalize_iao( Cp, S1 )
    DM1    = np.dot( ao2occ, ao2occ.T )
    DM2    = np.dot( Cp, Cp.T )
    A      = 2 * np.dot( DM1, np.dot( S1, np.dot( DM2, S21.T ) ) ) + P12 - np.dot( DM1 + DM2, S21.T )
    ao2iao = orthogonalize_iao( A, S1 )
    return ( ao2iao , S1, pmol )

#and PAO are constructed from this def
def localize_iao( mol, mf , lo2ao):

    Norbs = mol.nao_nr()
    ao2iao, S1, pmol = construct_iao( mol, mf )
    print("========== Check the orth and norm of IAO ==========")
    results = check_full_orthonormal_metric(ao2iao, S1)
    lo2iao = lo2ao @ ao2iao
    num_iao = ao2iao.shape[1]


    C = mf.mo_coeff
    nelec = mol.nelectron
    nsingleelec = mol.spin
    print("Number of electrons: ", nelec)
    n_mo_closed = int((nelec - nsingleelec) / 2)
    n_mo = int(n_mo_closed + nsingleelec)
    print("Number of closed orbitals: ", n_mo_closed)
    print("Number of open orbitals: ", nsingleelec)
    C_mo = C[:, :n_mo]
    lo2mo = lo2ao @ C_mo

    ok, s, rA, rB = subset_by_principal_angles(lo2mo, lo2iao)
    print("========== Check the most important property of IAO: lo2mo in lo2iao ==========")
    print("subset?", ok, " | cosines:", s[:rA], " | rA,rB:", rA, rB)


    # Determine the complement basis of the IAO space ao2com
    DM_iao     = np.dot( ao2iao, ao2iao.T )
    mx         = np.dot( S1, np.dot( DM_iao, S1 ) )
    eigs, vecs = scipy.linalg.eigh( a=mx, b=S1 ) # Small to large in scipy
    ao2com     = vecs[ :, : Norbs - num_iao ]
    
    # Redo the IAO contruction for the complement space
    p_list = construct_p_list( mol, pmol ) # return array of length Norbs; 1 if similar bf in pmol; 0 otherwise
    S31    = S1[  p_list == 0 , : ]
    S3     = S31[ : , p_list == 0 ]
    X      = np.linalg.solve( S3, np.dot( S31, ao2com ) )
    P13    = np.linalg.solve( S1, S31.T )
    Cp     = np.dot( P13, X )
    Cp     = orthogonalize_iao( Cp, S1 )
    DM1    = np.dot( ao2com, ao2com.T )
    DM3    = np.dot( Cp, Cp.T )
    A      = 2 * np.dot( DM1, np.dot( S1, np.dot( DM3, S31.T ) ) ) + P13 - np.dot( DM1 + DM3, S31.T )
    ao2com = orthogonalize_iao( A, S1 )
    ao2loc = np.hstack( ( ao2iao, ao2com ) )
    ao2loc = resort_orbitals( mol, ao2loc )
    ao2loc = orthogonalize_iao( ao2loc, S1 )
    
    # Quick check
    should_be_1 = np.dot( np.dot( ao2loc.T, S1 ), ao2loc )
    print ("QC-DMET :: iao_helper :: num_orb pmol =", pmol.nao_nr())
    print ("QC-DMET :: iao_helper :: num_orb mol  =", mol.nao_nr())
    print ("QC-DMET :: iao_helper :: norm( I - C_full.T * S * C_full ) =", np.linalg.norm( should_be_1 - np.eye( should_be_1.shape[0] ) ))
    
    '''
    cubes = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    for cube in cubes:
        cube_name = str(cube)
        coeff = ao2loc[:,cube]
        orbital(mol, 'orbital'+cube_name+'.cube', coeff)
    '''
    
    return ao2loc
    
    