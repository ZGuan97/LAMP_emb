#Molecule Struct ure
import os
import numpy as np
import scipy as sp
import scipy.linalg
import h5py
from pyscf import gto, ao2mo, scf, df
from pyscf import lib
from pyscf.lib import chkfile
from pyscf.lo import orth
from pyscf.lo.orth import lowdin
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from embed_sim import rdiis
import ic_helper
from functools import reduce
import basis_set_exchange as bse
import matplotlib.pyplot as plt


def make_es_dm(neo, open_shell, lo2eo, cloao, dm):
        if open_shell:
            es_dm = np.zeros((2, neo, neo))
            dma, dmb = dm
            ldma = reduce(lib.dot, (cloao, dma, cloao.conj().T))
            ldmb = reduce(lib.dot, (cloao, dmb, cloao.conj().T))
            es_dm[0] = reduce(lib.dot, (lo2eo.conj().T, ldma, lo2eo))
            es_dm[1] = reduce(lib.dot, (lo2eo.conj().T, ldmb, lo2eo))
        else:
            es_dm = np.zeros((neo, neo))
            ldm = reduce(lib.dot, (cloao, dm, cloao.conj().T))
            es_dm = reduce(lib.dot, (lo2eo.conj().T, ldm, lo2eo))
        return es_dm




def same_col_space(A, B, atol=1e-9):
    """
    判断 A 与 B 的列空间是否相同（欧氏度量）。
    返回 (is_same, sing_vals)，sing_vals 为 Q_A^T Q_B 的奇异值。
    """
    # 正交基（列满秩部分会被保留）
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)

    # 取有效列（防止数值秩导致的多余零列）
    rA = np.linalg.matrix_rank(A, tol=atol)
    rB = np.linalg.matrix_rank(B, tol=atol)
    QA = QA[:, :rA]
    QB = QB[:, :rB]

    # 若维数不同，列空间不可能相同
    if rA != rB:
        return False, np.array([])

    # 主夹角：奇异值应全 ~ 1
    s = np.linalg.svd(QA.T.conj() @ QB, compute_uv=False)
    is_same = np.allclose(s, np.ones_like(s), atol=10*atol)
    return is_same, s

def verify_matrices_equivalence(A, B, tolerance=1e-12):
    """
    验证矩阵 A 和 B 是否在容忍误差范围内相等。
    
    参数:
    A -- 第一个矩阵
    B -- 第二个矩阵
    tolerance -- 容忍误差，默认为 1e-12
    
    返回:
    True 如果 A 和 B 在容忍误差内相等，否则返回 False
    """
    # 计算矩阵 A 和 B 的差异
    difference = A - B
    frobenius_norm = np.linalg.norm(difference, 'fro')

    # 打印差异的 Frobenius 范数
    print("矩阵 A 和 B 的差异的 Frobenius 范数:", frobenius_norm)

    # 检查是否接近零
    if np.allclose(A, B, atol=tolerance):
        print("矩阵 A 和 B 是相同的矩阵（在容忍误差内）。")
        return True
    else:
        print("矩阵 A 和 B 不是完全相同的矩阵。")
        return False


def check_full_orthonormal(A, B, atol=1e-10):
    """
    检查两组矩阵的列向量是否正交归一（包括各自内部与交叉部分）。

    参数:
        A, B : ndarray
            两个矩阵（列向量集合）
        atol : float
            数值误差容忍阈值，默认 1e-10

    返回:
        results : dict
            包含每个检查结果的布尔值与偏差范数
    """

    results = {}

    # ========= 1️⃣ 检查 A 自身的归一性与正交性 =========
    O_A = A.T.conj() @ A
    diag_A = np.diag(O_A)
    off_A = O_A - np.diag(diag_A)
    is_A_normalized = np.allclose(diag_A, 1, atol=atol)
    is_A_orthogonal = np.allclose(off_A, 0, atol=atol)
    results["A_normalized"] = is_A_normalized
    results["A_orthogonal"] = is_A_orthogonal

    # ========= 2️⃣ 检查 B 自身的归一性与正交性 =========
    O_B = B.T.conj() @ B
    diag_B = np.diag(O_B)
    off_B = O_B - np.diag(diag_B)
    is_B_normalized = np.allclose(diag_B, 1, atol=atol)
    is_B_orthogonal = np.allclose(off_B, 0, atol=atol)
    results["B_normalized"] = is_B_normalized
    results["B_orthogonal"] = is_B_orthogonal

    # ========= 3️⃣ 检查 A 与 B 是否互相正交 =========
    O_AB = A.T.conj() @ B
    is_AB_orthogonal = np.allclose(O_AB, 0, atol=atol)
    results["A_B_orthogonal"] = is_AB_orthogonal

    # ========= 打印结果 =========
    print("===== A 自身检查 =====")
    print("归一性：", is_A_normalized)
    print("正交性：", is_A_orthogonal)
    #print("A 的重叠矩阵：\n", O_A)

    print("\n===== B 自身检查 =====")
    print("归一性：", is_B_normalized)
    print("正交性：", is_B_orthogonal)
    #print("B 的重叠矩阵：\n", O_B)

    print("\n===== A 与 B 的交叉检查 =====")
    print("是否互相正交：", is_AB_orthogonal)
    #print("交叉重叠矩阵 O_AB = A.T.conj() @ B ：\n", O_AB)

    # ========= 计算偏差指标 =========
    results["A_dev"] = np.linalg.norm(O_A - np.eye(O_A.shape[0]), 'fro')
    results["B_dev"] = np.linalg.norm(O_B - np.eye(O_B.shape[0]), 'fro')
    results["AB_dev"] = np.linalg.norm(O_AB, 'fro')
    print(f"\nA 偏离单位矩阵: {results['A_dev']:.3e}")
    print(f"B 偏离单位矩阵: {results['B_dev']:.3e}")
    print(f"A-B 交叉偏离: {results['AB_dev']:.3e}")

    return results




mol_root = gto.M(atom='CeNP2O2_root.xyz',basis={'Ce': gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='Ce',fmt='nwchem')),
                                               'O1' : gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='O',fmt='nwchem')),
                                               'O2' : gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='O',fmt='nwchem')),
                                               'O3' : gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='O',fmt='nwchem'))
                                               },symmetry=0,spin=1,charge=-9,verbose=0)


mol_buf1 = gto.M(atom='CeNP2O2_buf1.xyz',basis={'O1' : gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='O',fmt='nwchem'))},symmetry=0,spin=0,charge=-4,verbose=0)

mol_buf2 = gto.M(atom='CeNP2O2_buf2.xyz',basis={'O2' : gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='O',fmt='nwchem'))},symmetry=0,spin=0,charge=-4,verbose=0)

mol_buf3 = gto.M(atom='CeNP2O2_buf3.xyz',basis={'O3' : gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='O',fmt='nwchem'))},symmetry=0,spin=0,charge=-4,verbose=0)


mol_leaf1 = gto.M(atom='CeNP2O2_leaf1.xyz',basis={
                                               'N': gto.basis.parse(bse.get_basis('ANO-RCC-VDZP',elements='N',fmt='nwchem')),
                                               'C':'ANO-R0',
                                               'H' : 'ANO-R0',
                                               'P': gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='P',fmt='nwchem'))},symmetry=0,spin=0,charge=+3,verbose=0)

mol_leaf2 = gto.M(atom='CeNP2O2_leaf2.xyz',basis={
                                               'N': gto.basis.parse(bse.get_basis('ANO-RCC-VDZP',elements='N',fmt='nwchem')),
                                               'C':'ANO-R0',
                                               'H' : 'ANO-R0',
                                               'P': gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='P',fmt='nwchem'))},symmetry=0,spin=0,charge=+3,verbose=0)

mol_leaf3 = gto.M(atom='CeNP2O2_leaf3.xyz',basis={
                                               'N': gto.basis.parse(bse.get_basis('ANO-RCC-VDZP',elements='N',fmt='nwchem')),
                                               'C':'ANO-R0',
                                               'H' : 'ANO-R0',
                                               'P': gto.basis.parse(bse.get_basis('ANO-RCC-VTZP',elements='P',fmt='nwchem'))},symmetry=0,spin=0,charge=+3,verbose=0)
                                    
mol_bufs = [mol_buf1,mol_buf2,mol_buf3]
mol_leaves = [mol_leaf1,mol_leaf2,mol_leaf3]

core_coeffs = []
bath_coeffs = []
bath_buffs = []
bath_leaves = []
core_buffs = []
core_leaves = []

thres=1e-12

for i in range(len(mol_leaves)):
    # Perform ROHF for leaf_i + buff_i
    mol_tot_i = mol_bufs[i]+mol_leaves[i]
    #print('nao',mol_tot_i.nao_nr())
    mf_i = scf.rhf.RHF(mol_tot_i).x2c().density_fit()
    
    mf_i.chkfile = f'CeNP2O2_rohf_{i}.chk'
    mf_i.init_guess = 'chk'
    scfdat = chkfile.load(mf_i.chkfile,'scf')
    mf_i.e_tot = scfdat['e_tot']
    mf_i.mo_coeff = scfdat['mo_coeff']
    mf_i.mo_occ = scfdat['mo_occ']
    mf_i.mo_energy = scfdat['mo_energy']
    mf_i.level_shift = .2
    mf_i.max_cycle = 500
    mf_i.verbose = 4
    #mf_i.kernel()
    S_ovlp = scf.hf.get_ovlp(mol_tot_i)
    from embed_sim import ssdmet
    OA_i = [mol_bufs[i].atom_symbol(a) + '.*' 
      for a in range(mol_bufs[i].natm)]
    print(OA_i)
    imp_indices = mol_tot_i.search_ao_label(OA_i)
    env_indices = np.array([i for i in range(S_ovlp.shape[0]) if i not in imp_indices])
    mydmet = ssdmet.SSDMET(mf_i, title=f'frag_{i}', imp_idx=imp_indices, threshold=thres, es_natorb=False).density_fit()
    conv_tol = 1e-7
    mydmet.build(conv_tol, restore_imp = True)


    core_coeff_i = mydmet.fo_orb
    ao2eo_i = mydmet.es_orb
    all_indices_i = np.arange(ao2eo_i.shape[1])
    bath_indices = np.setdiff1d(all_indices_i, imp_indices)
    bath_coeff_i = ao2eo_i[:, bath_indices]

    bath_buf_i = bath_coeff_i[imp_indices, :]
    bath_buffs.append(bath_buf_i)
    bath_leave_i = bath_coeff_i[env_indices, :]
    bath_leaves.append(bath_leave_i)

    core_buf_i = core_coeff_i[imp_indices, :]
    core_buffs.append(core_buf_i)
    core_leave_i = core_coeff_i[env_indices, :]
    core_leaves.append(core_leave_i)
    #print('This is shape of bath_leave_i:')
    #print(bath_leave_i.shape)
    core_coeffs.append(core_coeff_i)
    bath_coeffs.append(bath_coeff_i)


def write_xyz(mol, filename="mol.xyz"):
    coords = mol.atom_coords(unit='Angstrom')
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    with open(filename, "w") as f:
        f.write(f"{mol.natm}\nGenerated from PySCF\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")


title = 'CeNP2O2'
mol_tot = np.sum([mol_root]+mol_leaves)
mol_tot.charge = 0
mol_tot.spin = 1
write_xyz(mol_tot, "mol_tot.xyz")
mol_tot.build()
mf = scf.rohf.ROHF(mol_tot).x2c().density_fit()
#print("This is shape of ao2core:",ao2core.shape)
#Build an initial guess for DMET SCF
#mf.chkfile='CeNP2O2_rohf.chk'
#mf.init_guess = 'chk'
#mf.verbose = 4
#mf.max_cycle = 2000
#scfdat = chkfile.load(mf.chkfile,'scf')
#mf.e_tot = scfdat['e_tot']
#mf.mo_coeff = scfdat['mo_coeff']
#mf.mo_occ = scfdat['mo_occ']
#mf.mo_energy = scfdat['mo_energy']
#mf.level_shift = 0.2
#mf.kernel()




all_labels = mol_tot.ao_labels()
nbasis = len(all_labels)
print("Number of AO:", nbasis)

imp_bufs = []
imp_bufs_indices = []
bath_buffs = bath_buffs
for i in range(len(mol_bufs)):
    imp_bufs.append([mol_bufs[i].atom_symbol(a) + '.*'
                for a in range(mol_bufs[i].natm)])
    #print(imp_bufs[i])
    imp_bufs_indices.append(mol_tot.search_ao_label(imp_bufs[i]))
    #print(imp_bufs_indices[i])

imp_A = ['Ce.*','O1.*','O2.*','O3.*']
imp_A_indices = mol_tot.search_ao_label(imp_A)


#=====================================拼接ao2eo=====================================
#1.确定列数
nA = len(imp_A_indices) # A 区行数 = A 区列数
nbath = sum(B.shape[1] for B in bath_buffs)
ncol_total = nA + nbath
ao2eo = np.zeros((nbasis, ncol_total))

#2.左上角块
S = scf.hf.get_ovlp(mol_tot)
ao2lo_old, lo2ao_old = lowdin(S), lowdin(S) @ S
#ao2lo_old, lo2ao_old = lowdin(S), lowdin(S) @ S
#B_indices = np.array([i for i in range(S.shape[0]) if i not in imp_A_indices])
#lo2ao_A = lo2ao_old[:, imp_A_indices]
S_AA = S[np.ix_(imp_A_indices, imp_A_indices)]
X_AA = lowdin(S_AA)
#X_AA_minus_1 = S_AA_1_2 which is X_AA in AO-DMET paper.
ao2eo[np.ix_(imp_A_indices, imp_A_indices)] = X_AA
#ao2imp = ao2eo[:, imp_A_indices]
#LO2imp = lo2ao_old @ ao2imp
#print("Check LO2imp")
#Results = check_full_orthonormal(LO2imp, LO2imp, atol=1e-10)

#3.右上角块
col = len(imp_A_indices)  # 从 A 区列后开始
for i, idx in enumerate(imp_bufs_indices):
    B = bath_buffs[i]   # shape = (len(idx), ncol)
    assert B.shape[0] == len(idx), (i, B.shape, len(idx))

    ncol = B.shape[1]
    ao2eo[np.ix_(idx, np.arange(col, col + ncol))] = B
    col += ncol

#4.右下角块
# 1) 拼右下角：3_i 块对角
C3 = scipy.linalg.block_diag(*bath_leaves)   # shape: (sum nrow_i, sum ncol_i)
# 2) 计算右下角在 ao2eo 里的连续起点                         
nbuf = sum(len(idx) for idx in imp_bufs_indices)  # buffer 总行数（你右上角4_i所在的那些行）
row0 = nA                                         # leaf 区从这里开始（按你图的分块顺序）
col0 = nA                                         # bath 列从这里开始（你右上角就是从 A 后面开始填的）
# 3) 塞进去
nr, nc = C3.shape
ao2eo[row0:row0+nr, col0:col0+nc] = C3

#all_indices = np.arange(ao2eo.shape[1])
#bath_indices = np.setdiff1d(all_indices, imp_A_indices)
#ao2bath = ao2eo[:, bath_indices]
#LO2bath = lo2ao_old @ ao2bath
#print("Check LO2bath")
#Results = check_full_orthonormal(LO2bath, LO2bath, atol=1e-10)
#np.savetxt("ao2eo2.txt", ao2eo, fmt="%.16e")

#ERROR_EO = ao2eo - AO2EO
Lo2eo = lo2ao_old @ ao2eo
G = Lo2eo.T.conj() @ Lo2eo  
eo2eo_new = lowdin(G)
ao2eo_new = ao2eo @ eo2eo_new
ao2eo = ao2eo_new

plt.imshow(np.abs(ao2eo) > 1e-12, cmap='gray', origin='upper')
plt.xlabel("Column index")
plt.ylabel("Row index")
plt.title("Nonzero pattern")
plt.show()

#=====================================拼接ao2core=====================================
#1.确定列数
nA = len(imp_A_indices)
ncore = sum(B.shape[1] for B in core_buffs)
ncol_total = nA + ncore
ao2core_uncut = np.zeros((nbasis, ncol_total))
#print("This is shape of ao2core before cut:",ao2core.shape)
#print("This is shape of X_AA_minus_1",X_AA_minus_1.shape)

#2.左上角块
ao2core_uncut[np.ix_(imp_A_indices, imp_A_indices)] = X_AA

#3.右上角块
col = len(imp_A_indices)  # 从 A 区列后开始
for i, idx in enumerate(imp_bufs_indices):
    B = core_buffs[i]   # shape = (len(idx), ncol)
    assert B.shape[0] == len(idx), (i, B.shape, len(idx))

    ncol = B.shape[1]
    ao2core_uncut[np.ix_(idx, np.arange(col, col + ncol))] = B
    col += ncol

#4.右下角块
# 1) 拼右下角：3_i 块对角
C4 = scipy.linalg.block_diag(*core_leaves)   # shape: (sum nrow_i, sum ncol_i)
# 2) 计算右下角在 ao2eo 里的连续起点                         
nbuf = sum(len(idx) for idx in imp_bufs_indices)  # buffer 总行数（你右上角4_i所在的那些行）
row0 = nA                                         # leaf 区从这里开始（按你图的分块顺序）
col0 = nA                                         # bath 列从这里开始（你右上角就是从 A 后面开始填的）
# 3) 塞进去
nr, nc = C4.shape
ao2core_uncut[row0:row0+nr, col0:col0+nc] = C4
ao2core = ao2core_uncut[:, nA:] #cut
#np.savetxt("ao2core2.txt", ao2eo, fmt="%.16e")

#ERROR_CORE = ao2core - AO2CORE
plt.imshow(np.abs(ao2core) > 1e-12, cmap='gray', origin='upper')
plt.xlabel("Column index")
plt.ylabel("Row index")
plt.title("Nonzero pattern")
plt.show()


#LO2imp = lo2ao_old @ ao2imp
#LO2bath = lo2ao_old @ ao2bath
#LO2core = lo2ao_old @ ao2core

#print("Check LO2imp LO2bath")
#Results = check_full_orthonormal(LO2imp, LO2bath, atol=1e-10)
#print("Check LO2bath LO2core")
#Results = check_full_orthonormal(LO2bath, LO2core, atol=1e-10)
#print("Check LO2imp LO2core")
#Results = check_full_orthonormal(LO2imp, LO2core, atol=1e-10)



#5.check HF in HF DMET exact condition
nelec = mol_tot.nelectron
np.savetxt("ao2eo.txt", ao2eo, fmt="%.16e")
print("This is shape of ao2eo:",ao2eo.shape)

hcore = mf.get_hcore()
dm_core = 2 * ao2core @ ao2core.T.conj()
#print("This is shape of dm_core:",dm_core.shape)
vj, vk = mf.get_jk(dm=dm_core)
eo_hcore_tilde = reduce(lib.dot,(ao2eo.conj().T, hcore + vj - 0.5 * vk, ao2eo))
eo_S = reduce(lib.dot,(ao2eo.conj().T, S, ao2eo))



es_mol = gto.M()
es_mol.verbose = 4
es_mol.spin = mol_tot.spin
es_mol.incore_anyway = True
es_mol.nelectron = int(nelec - ao2core.shape[1]*2)
es_mol.build()

nao = hcore.shape[0]
print('nao',nao)

es_mf = scf.rohf.ROHF(es_mol).x2c().density_fit()
es_mf.mo_energy = np.zeros((nao))
es_mf.verbose = 4
es_mf.max_cycle = 5000
es_mf.diis_space = 8
es_mf.level_shift = .2
es_mf.DIIS = scf.ADIIS
es_mf.conv_tol = 1e-7
es_mf.chkfile = 'imp_rohf.chk'
es_mf.init_guess = 'chk'


def make_es_cderi(title, es_orb, with_df):
    erifile = title+'_es_cderi.h5'
    dataname = 'j3c'
    feri = df.outcore._create_h5file(erifile, dataname)
    ijmosym, nij_pair, moij, ijslice = _conc_mos(es_orb, es_orb, True)
    naux = with_df.get_naoaux()
    neo = es_orb.shape[-1]
    nao_pair = neo*(neo+1)//2
    label = '%s/%d'%(dataname, 0)
    feri[label] = np.zeros((naux,nao_pair),dtype=np.float64)
    nij = 0
    for eri1 in with_df.loop():
        Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym)
        nrow = Lij.shape[0]
        feri[label][nij:nij+nrow] = Lij
        nij += nrow
    return erifile


es_mf.get_hcore = lambda *args: eo_hcore_tilde
es_mf.get_ovlp = lambda *args: eo_S
es_mf.with_df._cderi = make_es_cderi(title, ao2eo, mf.with_df)
es_mf.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=mol_tot.search_ao_label(['Ce.*']),power=0.2)

#dma, dmb = mf.make_rdm1()
#dm = np.stack((dma, dmb), axis=0)
#lo_dm = lo2ao_old @ dm @ lo2ao_old.T.conj()
#lo2eo = lo2ao_old @ ao2eo
#neo = lo2eo.shape[1]
#es_dm = make_es_dm(neo, True, lo2eo, lo2ao_old, dm)

es_mf.kernel()
#err = es_mf.e_tot + mf.energy_tot(dm=dm_core) - mf.e_tot


nes = ao2eo.shape[1]
nfo = ao2core.shape[1]
nfv = nbasis - nes - nfo
ao2fv = np.zeros((nbasis, nfv))



from pyscf.tools import molden
with open(title+' imp_rohf_orbs.molden', 'w')as f1:
    molden.header(mol_tot, f1)
    molden.orbital_coeff(mol_tot, f1, ao2eo @ es_mf.mo_coeff, ene=es_mf.mo_energy, occ=es_mf.mo_occ)


ncas, nelec = 12, 1
#ncas, nelec, es_mo = mydmet.avas(['Ce 4f','Ce 5d'], minao=CLUS_MOL._basis['Ce'], threshold=0.5, openshell_option=3)
with open("CeNP2O2_cas_info") as f:
    lines = f.readlines()
    mo_indices = list(map(int, lines[1].split()))

print("This is mo_indices")
print(mo_indices)



from embed_sim import sacasscf_mixer, siso
es_cas = sacasscf_mixer.sacasscf_mixer(es_mf, ncas, nelec, statelis=[0,12])
es_mo = es_cas.sort_mo(mo_indices)
es_cas.verbose = 4
es_cas.kernel(es_mo)
Ha2cm = 219474.63
np.savetxt(title+'_cas_NO_SOC.txt',(es_cas.fcisolver.e_states-np.min(es_cas.fcisolver.e_states))*Ha2cm,fmt='%.6f')

#===================PT2========================
es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas, method='SC')
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
#===================PT2========================
Ha2cm = 219474.63
np.savetxt(title+'_opt.txt',(es_cas.fcisolver.e_states-np.min(es_cas.fcisolver.e_states))*Ha2cm,fmt='%.6f')

def cas_tot(es_cas):
        from embed_sim import sacasscf_mixer
        mf_no_kernel = scf.rohf.ROHF(mol_tot).x2c().density_fit()
        #total_cas = sacasscf_mixer.sacasscf_mixer(mf, es_cas.ncas, es_cas.nelecas, statelis=sacasscf_mixer.read_statelis(es_cas), weights=es_cas.weights)
        total_cas = sacasscf_mixer.sacasscf_mixer(mf_no_kernel, es_cas.ncas, es_cas.nelecas, statelis=sacasscf_mixer.read_statelis(es_cas), weights=es_cas.weights)
        total_cas.fcisolver = es_cas.fcisolver
        total_cas.ci = es_cas.ci
        total_cas.mo_coeff = np.hstack((ao2core, ao2eo @ es_cas.mo_coeff, ao2fv))
        return total_cas

total_cas = cas_tot(es_cas)

mysiso = siso.SISO(title, total_cas, verbose=6)
mysiso.kernel()





