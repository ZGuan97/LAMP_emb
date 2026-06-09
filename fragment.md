# Fragment-Based Bath Orbital Construction

本文档用于记录在当前 DMET 框架下实现 fragment-based bath orbital construction 的物理动机、公式推导、算法设计和程序实现细节。

## 背景与动机

标准 DMET 中，bath orbital 的构造通常依赖一个全局低层波函数，例如全体系 Hartree-Fock (HF) 或 CASSCF 波函数。给定全局一体密度矩阵，先将轨道空间划分为 impurity 和 environment，再通过 impurity-environment 之间的纠缠结构构造 bath orbitals。这样得到的 embedded cluster space 可以保留 impurity 与环境之间最重要的量子纠缠。

对于过渡金属和镧系配合物，特别是单离子磁体相关体系，全局 HF 计算本身可能已经非常昂贵，甚至在大配体、大基组或低对称性结构下成为瓶颈。因此希望避免对完整体系进行全局 SCF，而是利用体系的化学分片结构近似构造 bath orbitals。

核心想法是：

1. 将配体按照化学上可分离的分子片段划分为多个 ligand fragments。
2. 对中心过渡金属或镧系离子和每一个 ligand fragment 构造一个局部子体系。
3. 在每个局部子体系上进行低层 SCF 计算，得到局部密度矩阵。
4. 从每个局部密度矩阵中构造该 fragment 对 impurity 的 bath orbitals。
5. 将所有 fragment 产生的 bath orbitals 与 impurity orbitals 合并，形成最终 DMET embedded cluster space。

这种做法希望将全局 SCF 的代价替换为一系列更小的 fragment SCF 计算，同时仍然保留不同配体片段对中心金属 impurity 的主要纠缠贡献。

## 对称性与 CAHF

直接截取中心离子和一部分配体会带来一个重要问题：局部子体系不再保留完整配合物的晶体场环境。对于过渡金属或镧系离子，这可能破坏原本的近简并 $d$ 或 $f$ 轨道结构和自旋/多重态结构，导致普通 HF/ROHF/UHF 计算难以收敛，或者收敛到不具有目标物理意义的局部基态。

为缓解这个问题，可以使用已经发展的 configuration-averaged Hartree-Fock (CAHF) 方法。CAHF 的目标不是选择某一个具体自旋态或占据构型作为单一平均场基态，而是对相关自旋态/构型进行平均，构造更平滑、更对称的平均场势。

在 fragment bath construction 中使用 CAHF 的预期优势包括：

1. 对不同自旋态进行平均，减少局部晶体场截断造成的人为偏置。
2. 避免普通 SCF 在近简并 $d/f$ 壳层中选择不稳定或非物理占据。
3. 提高 fragment SCF 的收敛性。
4. 得到更适合作为 bath orbital construction reference 的一体密度矩阵。

因此，每个 center-ligand fragment 的参考波函数可以优先使用 CAHF，而不是普通 HF。

## 轨道划分

设全体系 AO 轨道空间为

$$
\mathcal{A} = \mathcal{I} \oplus \mathcal{E},
$$

其中 $\mathcal{I}$ 是 DMET impurity 空间，通常包含中心离子的局域轨道，例如过渡金属 $3d/4d/5d$ 轨道、镧系 $4f/5d/6s$ 相关轨道，以及必要的双壳层或价层轨道；$\mathcal{E}$ 是环境轨道空间。

将环境进一步按配体分片：

$$
\mathcal{E} = \bigoplus_f \mathcal{L}_f,
$$

其中 $\mathcal{L}_f$ 表示第 $f$ 个 ligand fragment 的 AO 或局域轨道子空间。

对每个 fragment 定义局部计算空间：

$$
\mathcal{A}_f = \mathcal{I} \oplus \mathcal{L}_f.
$$

在 $\mathcal{A}_f$ 上进行 fragment SCF/CAHF 计算，并得到局部一体密度矩阵

$$
\rho^{(f)}.
$$

如果使用非正交 AO 基，需要同时保留局部 overlap matrix

$$
S^{(f)}.
$$

## 固定 impurity 的正交化

传统 DMET 通常会先对全局 AO 空间做 Lowdin symmetric orthogonalization，然后在全局正交基中定义 impurity、environment 和 bath orbitals。对于 fragment-based construction，这一步会引入一个技术问题：每个局部子体系的 overlap matrix 不同，因此在 $\mathcal{I} \oplus \mathcal{L}_1$ 中正交化得到的 impurity 空间，和在 $\mathcal{I} \oplus \mathcal{L}_2$ 中正交化得到的 impurity 空间并不严格相同。

也就是说，形式上都记作同一个 impurity $A$，但 fragment-wise Lowdin 之后实际得到的是

$$
\tilde{A}^{(1)} \ne \tilde{A}^{(2)}.
$$

这会破坏后续合并 bath orbitals 时对同一个 impurity block 的一致定义。为避免这个问题，FDMET 必须在正交化过程中保持 impurity orbitals 固定，只对 ligand/environment 子空间进行与 impurity 正交的处理。这样所有 fragment 都共享同一个 impurity 定义：

$$
A^{(1)} = A^{(2)} = \cdots = A.
$$

固定 impurity 的处理流程可以概括为：

1. 在全局 AO metric 下定义并固定 impurity coefficient $C_I$。
2. 对每个 fragment 的 ligand 轨道 $C_{L_f}$，投影掉其在 impurity 空间上的分量。
3. 在剩余 ligand 子空间内做正交化和线性相关性筛选。
4. 用固定的 $C_I$ 与处理后的 $C_{L_f}^{\perp I}$ 构造 fragment reference 和 bath orbitals。
5. 合并所有 bath orbitals 后，在保持 $C_I$ 不变的前提下，对 bath block 做最终正交化。

对应的 metric projection 可以写为

$$
P_I = C_I (C_I^\dagger S C_I)^{-1} C_I^\dagger S,
$$

$$
C_{L_f}^{\perp I} = (1 - P_I) C_{L_f}.
$$

随后只在 $C_{L_f}^{\perp I}$ 张成的子空间内做正交化。这样可以避免不同 fragment 对 impurity 的重新混合，同时仍然得到数值稳定的局部正交环境空间。

如果使用非正交 AO 基，需要同时保留局部 overlap matrix

$$
S^{(f)}.
$$

为了简化 bath orbital construction，可以在固定 impurity 后的局部轨道空间中正交化 ligand/environment block。若 $X_f$ 表示保持 impurity 不变、只正交化 ligand block 的变换矩阵，则局部密度可变换为：

$$
\tilde{\rho}^{(f)} = X_f^{-1} \rho^{(f)} X_f^{-\dagger}.
$$

后续 bath construction 可以在共享 impurity 定义的正交轨道表示中进行。

## Bath Orbital Construction

在第 $f$ 个 fragment 的正交局部轨道空间中，将密度矩阵按 impurity 和 ligand fragment 分块：

$$
\tilde{\rho}^{(f)} =
\begin{pmatrix}
\rho^{(f)}_{II} & \rho^{(f)}_{IL} \\
\rho^{(f)}_{LI} & \rho^{(f)}_{LL}
\end{pmatrix}.
$$

Fragment bath orbitals 通过对角化环境部分密度矩阵得到。具体地，对 ligand fragment block 做自然轨道分解：

$$
\rho^{(f)}_{LL} U_f = U_f n_f,
$$

其中 $U_f$ 的列向量位于 ligand fragment 空间 $\mathcal{L}_f$，$n_f$ 是对应的环境自然占据数。占据数既不是接近 0 的 frozen virtual，也不是接近 2 的 frozen occupied 时，对应轨道作为该 fragment 贡献的 bath orbital：

$$
B_f = U_f[:, n_{\mathrm{th}} \le n_f \le 2-n_{\mathrm{th}}].
$$

最终 embedded cluster space 由 impurity 与所有 fragment bath orbitals 合并得到：

$$
\mathcal{C}_{emb} =
\mathcal{I} \oplus
\operatorname{span}\{B_1, B_2, \ldots, B_N\}.
$$

实际实现中需要注意：

1. 每个 fragment 最多贡献 $\dim(\mathcal{I})$ 个 bath orbitals。
2. 不同 fragment 产生的 bath orbitals 在全局空间中可能不严格正交，需要在合并后重新正交化。
3. 若 fragment spaces 之间没有重叠，bath orbitals 的跨 fragment 重叠主要来自全局 AO overlap；需要在同一个全局正交基或全局 AO metric 下处理。
4. 需要设定 density occupation threshold，将接近 0 的环境自然轨道归为 frozen virtual，将接近 2 的环境自然轨道归为 frozen occupied，其余轨道作为 bath orbitals。

### Frozen occupied 轨道与整数电子数问题

标准 DMET 中，FO、bath 和 FV 来自同一个全局 reference density 的同一次正交轨道分解。因此 FO 轨道天然正交归一，FO 上的电子数可以写成

$$
N_{\mathrm{FO}} = 2 n_{\mathrm{FO}},
$$

其中 $n_{\mathrm{FO}}$ 是 frozen occupied orbital 的个数。embedded cluster 的电子数随之为

$$
N_{\mathrm{emb}} = N_{\mathrm{tot}} - N_{\mathrm{FO}},
$$

也是整数，可以直接用于后续 ROHF、CASCI 或 CASSCF 计算。

在 fragment-based construction 中，每个 fragment 都会给出一组局部 FO 候选轨道：

$$
C_{\mathrm{FO}}^{(1)}, C_{\mathrm{FO}}^{(2)}, \ldots
$$

这些轨道来自不同的局部 reference density，并且每个局部 reference 都包含同一个 impurity。将它们嵌回母体 AO basis 后，通常既不彼此正交，也不一定与合并后的 impurity+bath 空间严格正交。因此不能简单拼接这些 FO 轨道。若只做线性正交化，得到的正交轨道虽然数值上可用，但它们不再对应原始 fragment density 中占据数严格为 2 的自然轨道；FO 上电子数也不能再简单解释为 $2n_{\mathrm{FO}}$。这会导致 embedded space 的整数电子数无法严格指定。

目前有三类可能方案。当前实现选择方案 1 作为默认路线，因为它不需要从 fragment-local FO 轨道推断全局整数电子数，算法语义最干净，也最容易调试。

1. 不构造 FO。
   - 这是当前实现采用的默认方案。
   - 设置 `fo_orb = []`，并令 `fv_orb` 为 fixed impurity + bath 的全局正交补空间。
   - embedded cluster 的电子数直接取全体系电子数：

$$
N_{\mathrm{emb}} = N_{\mathrm{tot}}.
$$

   - 因为 $N_{\mathrm{FO}}=0$，电子数定义保持整数且没有额外解释。
   - 优点是不会重复计数 fragment-local occupied electron，也不会把非正交 FO 候选轨道强行解释为整数电子数。
   - 缺点是 embedded space 的电子数不再通过 FO 压缩；`impurity + bath` 轨道数必须足够容纳全体系电子数，否则 embedded ROHF/CAS 无法定义。
   - 若出现 $N_\alpha > N_{\mathrm{emb\ orb}}$，代码应直接报错，而不是尝试隐式构造 FO 或改变电子数。

2. 从全局近似 density 重新定义 FO。
   - 不直接合并 fragment-local FO 轨道，而是先把 fragment-local density 映回母体 AO basis。
   - 为避免 impurity 被重复计数，可以优先只拼装 ligand/environment block 的 density。
   - 在 fixed impurity + merged bath 的全局正交补空间中对该近似 environment density 做自然轨道分解。
   - 自然占据数接近 2 的全局正交轨道定义为 FO，自然占据数接近 0 的轨道定义为 FV。
   - 这样 FO 轨道来自一次全局 metric 下的自然轨道分解，可以恢复整数规则 $N_{\mathrm{FO}}=2n_{\mathrm{FO}}$。
   - 这是后续最接近标准 SSDMET 逻辑的推荐方向。

3. 只冻结明确的全局 core 轨道。
   - 不试图恢复所有 fragment-local FO。
   - 只选择远离 impurity+bath、自然占据稳定接近 2、且可以根据 AO label、壳层、能量或 impurity 成分判断为深层 core 的轨道。
   - 该方案物理上更保守，能保持整数电子数，但压缩效率较低。

因此当前 fragment-DMET 的 bath construction 明确定义为只构造 `impurity + bath`，不尝试从 fragment reference 中恢复全局 frozen occupied space。方案 2 和方案 3 暂时作为后续可选扩展，而不是当前默认算法的一部分。

## 与标准 DMET 的关系

标准 DMET 的 bath orbitals 来自全局 reference density：

$$
\rho^{global}.
$$

fragment-based 方法用一组局部 reference densities 近似替代全局 density：

$$
\rho^{global}
\quad \Longrightarrow \quad
\{\rho^{(f)}\}_f.
$$

因此该方法不是严格等价于标准 DMET，而是一个基于化学分片和局部平均场的近似。它的合理性依赖于以下物理假设：

1. 中心离子与每个配体片段的主要纠缠可以通过相应 center-ligand 局部子体系捕捉。
2. 配体之间通过中心离子间接耦合的效应，可以在最终 embedded cluster Hamiltonian 中部分恢复。
3. CAHF reference 能够提供比普通 HF 更稳定、更对称的局部平均场。

## 初步算法

输入：

1. 全体系 molecule object。
2. impurity orbital indices。
3. ligand fragment orbital index sets。
4. CAHF/SCF 参数。
5. bath occupation threshold。

流程：

1. 构造 impurity orbital set $\mathcal{I}$。
2. 对每个 ligand fragment $f$：
   - 构造局部子体系 $\mathcal{A}_f = \mathcal{I} \oplus \mathcal{L}_f$。
   - 生成局部 one-electron integrals、two-electron integrals 和 overlap matrix。
   - 运行 CAHF 或 SCF。
   - 提取局部一体密度矩阵 $\rho^{(f)}$。
   - 保持 impurity block 不变，并在正交局部轨道中构造 bath orbitals $B_f$。
   - 将 $B_f$ 映射回全局轨道表示。
3. 合并所有 $B_f$。
4. 在全局 metric 下对 bath orbitals 做正交化和线性相关性筛选。
5. 构造最终 embedded cluster basis：

$$
C_{emb} = [C_I, C_{B_1}, C_{B_2}, \ldots, C_{B_N}].
$$

6. 使用 $C_{emb}$ 投影全局 Hamiltonian，进入后续 DMET impurity solver。

## 程序实现记录

### 待确定接口

需要在代码中明确以下对象和接口：

1. impurity orbitals 的来源：AO index、localized orbital index，还是 active orbital coefficient。
2. ligand fragment 的定义方式：atom list、AO index list，还是 localized orbital index list。
3. fragment SCF/CAHF 是否直接在截断 molecule 上运行，还是在全局 AO integral 投影后的子空间中运行。
4. bath orbital 输出格式：全局 AO coefficient、orthogonal AO coefficient，还是 localized orbital coefficient。
5. 固定 impurity 的正交化应作为 FDMET 的固定步骤实现，而不是用户可选接口。
6. 与现有 DMET embedding builder 的对接方式。

### 建议模块划分

可能的实现位置：

1. `embed_sim/fragment.py`：fragment bath construction 的主逻辑。
2. `embed_sim/bath.py`：若已有 bath construction，可在其中加入 fragment-based 分支。
3. `embed_sim/cahf.py`：CAHF reference 的接口或 wrapper。
4. `examples/`：加入最小 metal-ligand fragment 示例。

### 当前程序框架状态

已新增 `embed_sim/fragment.py`，其中 `FDMET` 目前作为 `SSDMET` 的子类存在。当前实现重点是先固定对外接口、分片输入和继承关系，而不是立即完成 fragment-local bath 的全部数值流程：

1. `FDMET(mol, ...)` 接收 PySCF `Mole` 对象，并且只保存 molecule 与 AO-index 分片信息，不自动构造或消费全局 SCF reference。
2. `build()` 进入 fragment bath 路径。该路径目前显式抛出 `NotImplementedError`，避免错误地根据全局 SCF density 构造 `FDMET` bath orbital。
3. `FDMET` 的 impurity 和 fragments 用 PySCF AO label/pattern 定义；`imp_idx` 的 property 和 setter 直接继承 `SSDMET` 的实现，fragment labels 也通过同一个 PySCF AO label 解析接口转换为 AO index list。
4. `imp_charge` 和 `fragment_charges` 作为显式输入保存；第 $f$ 个 fragment-local molecule 的总电荷取为 `imp_charge + fragment_charges[f]`。
5. `fragment_scf`、`threshold` 等参数已经在类中保留，供后续 fragment-local bath builder 对接。
6. `examples/fragment_dmet.py` 中已经为 `Co(SH)_4` 定义分片：impurity 为 Co，`L1` 到 `L4` 分别是四个 SH ligand fragment，并显式给出 Co 与每个 SH ligand 的电荷。

这一阶段的基空间约定如下：真正的 fragment-local 实现应从每个 $\mathcal{I}\oplus\mathcal{L}_f$ 的局部 reference density 构造 bath，而不是使用 whole-system SCF density。后续 fragment bath builder 的最终输出仍应为全局 AO coefficient `es_orb`、`fo_orb` 和 `fv_orb`，以便继续复用现有 Hamiltonian projection、AVAS、`total_cas` 和 SISO 接口。

### 当前新增：fragment-local CAHF 第一步

`embed_sim/fragment.py` 现在实现了 fragment reference 的第一步：

1. 对每个 ligand fragment $f$，取全局 AO index 集合 $\mathcal{I}\cup\mathcal{L}_f$，并找到这些 AO 所在的原子。
2. 用这些原子从母体 `Mole` 复制出一个截断的 `fragment_mol`。坐标使用 Bohr 单位，basis 和 ECP 复用母体 molecule 的解析后数据。
3. 在 `fragment_mol` 上运行 fragment-local reference。默认路径为 CAHF；也保留 `rohf` 作为调试路径。
4. 返回 `FragmentReference`，其中保存 `fragment_mol`、局部 `mf`、涉及的全局 atom/AO index，以及全局 AO 到局部 AO 的 impurity/ligand 分块映射。

当前 CAHF 的默认 active-space 参数为 `ncas=5, nelecas=7`，对应当前 Co d7 示例的最小默认值。更一般体系应通过 `fragment_scf_options` 显式传入。`FDMET` 根据 `imp_charge + fragment_charges[f]` 计算第 $f$ 个 fragment 子体系的总 charge，并把该总 charge 传给底层 `run_fragment_scf()`；如果直接调用 `run_fragment_scf()`，也必须显式传入总 `charge`。没有显式 charge 时直接报错。

局部 fragment SCF 的输出等级由 `FDMET.build(fragment_scf_verbose=3)` 控制，与外层 `FDMET.verbose` 分开。该参数只属于本次 build 行为，不保存在 `FDMET` 构造器中。

当前 fragment-local CAHF 的单电子 Hamiltonian 只来自截断后的 `impurity + ligand_f` 局部 molecule，即 PySCF 在该 `fragment_mol` 上计算的 `hcore`。其他 ligand fragment 不在该局部 molecule 中，因此它们的核吸引势、电子库仑势、净负电荷静电场以及由此导致的轨道极化目前都没有进入第 `f` 个 fragment CAHF。换言之，`fragment_charges` 当前只决定对应局部 molecule 的电子数，还没有作为 point charge、embedding potential 或全局投影势作用到其他 fragment 的局部参考计算中。

### 全局 AO index 与局部 fragment AO index

`FDMET` 的输入分片定义在母体 molecule 的全局 AO basis 中。设母体 molecule 的 AO index 为

$$
\mu = 0,1,\ldots,N_{\mathrm{AO}}-1.
$$

用户输入的 `imp_idx` 解析为全局 impurity AO 集合

$$
I = \{\mu_I\},
$$

第 $f$ 个 ligand fragment 解析为全局 ligand AO 集合

$$
L_f = \{\mu_{L_f}\}.
$$

构造第 $f$ 个局部 fragment reference 时，代码先形成

$$
G_f = I \cup L_f,
$$

并从这些全局 AO index 找到它们所属的母体 atom index。随后用这些 atom 重新构造一个局部 `fragment_mol`。因此局部 molecule 有自己的 atom 编号和 AO 编号：

$$
p = 0,1,\ldots,N_{\mathrm{AO}}^{(f)}-1.
$$

这两个编号系统不是同一个编号系统。全局 AO index $\mu$ 指的是母体 molecule 中的 AO；局部 AO index $p$ 指的是截断后的 `fragment_mol` 中的 AO。

当前 `FragmentReference` 保存以下映射信息：

1. `atom_ids`：局部 `fragment_mol` 中每个 atom 对应的母体 molecule atom index。若 `atom_ids = [0, 1, 5]`，表示局部 atom 0/1/2 分别来自母体 atom 0/1/5。
2. `global_idx`：参与该 fragment reference 的母体 AO index，目前按 `list(impurity_idx) + list(fragment_idx)` 保存。
3. `local_imp_idx`：在局部 `fragment_mol` AO basis 中属于 impurity 的 AO index。
4. `local_fragment_idx`：在局部 `fragment_mol` AO basis 中属于 ligand fragment 的 AO index。

映射的构造方式是：对每个被选中的母体 atom，按 PySCF 在该 atom 内生成 AO 的顺序建立对应关系。也就是说，局部 molecule 中某个 atom 的第 $k$ 个 AO，对应母体 molecule 中同一个 atom 的第 $k$ 个 AO。这样避免依赖 AO label 字符串做反查，因为某些壳层或简并 AO 的 label 可能重复或不够唯一。

后续如果要把局部 bath orbital 映射回母体 AO basis，需要使用这个局部到全局 AO 对应关系。设局部 bath orbital 在 `fragment_mol` AO basis 中的系数为

$$
b_p,
$$

则嵌回母体 AO basis 时，应构造一个长度为 $N_{\mathrm{AO}}$ 的系数向量 $B_\mu$：

$$
B_\mu =
\begin{cases}
b_p, & \text{若局部 AO }p\text{ 对应母体 AO }\mu,\\
0, & \text{若母体 AO }\mu\notin G_f.
\end{cases}
$$

这一点很重要：局部 density、局部 MO 和局部 bath orbital 都首先生活在 `fragment_mol` 的局部 AO basis 中，不能直接拿它们的 index 当作母体 molecule 的全局 AO index 使用。

`fragment_scf_options` 现在也支持 fragment-local RDIIS：

```python
fragment_scf_options={
    "diis": "rdiis",
    "rdiis_prop": "dS",
    "rdiis_imp_idx": ["Co.*d"],
    "rdiis_power": 0.2,
}
```

其中 `rdiis_imp_idx` 在每个局部 `fragment_mol` 中解析，因此可以直接使用局部 AO label/pattern。
若没有显式设置 `rdiis_mute`，RDIIS 的输出跟随 `fragment_scf_verbose`：`fragment_scf_verbose < logger.INFO` 时静音，因此默认 `fragment_scf_verbose=3` 不输出 RDIIS entropy；`fragment_scf_verbose=4` 时输出。

### 关键函数

```python
def run_fragment_scf(mol, impurity_idx, fragment_idx, charge, fragment_scf="cahf"):
    """Run HF/CAHF on one metal-ligand fragment and return density information."""
```

```python
def build_fragment(mol, impurity_idx, fragment_idx, charge, fragment_scf="cahf", **kwargs):
    """Build one impurity-ligand fragment Mole and run its local reference."""
```

```python
def fragment_bath_from_fragment_density(dm, impurity_idx, fragment_idx, overlap=None, threshold=1e-13):
    """Construct bath orbitals from one fragment density matrix."""
```

### 当前新增：fragment-local bath orbital construction

`embed_sim/fragment.py` 现在已经开始实现 bath orbital construction。当前实现遵循“尽量复用 SSDMET 路径”的原则：

1. 对每个 impurity-ligand fragment 先运行 `run_fragment_scf()` 得到局部 reference。
2. 用 `ssdmet.mf_or_cas_make_rdm1s()` 提取 fragment-local AO density。
3. 在 `fragment_bath_from_fragment_density()` 中调用模块级 `ssdmet.lowdin_orth(..., preserve_imp=True)`，复用现有固定 impurity 的 Lowdin 正交化实现。
4. 将局部 density 限制到 `local_imp_idx + local_fragment_idx` 的正交轨道子空间。
5. 只对 fragment/environment density block 做自然轨道分解，按标准 SSDMET 的环境自然占据数阈值规则选出 bath orbitals。
6. 利用 `FragmentReference.local_to_global_idx` 将局部 AO coefficient 嵌回母体 molecule 的全局 AO basis。
7. 合并所有 fragment bath 后，在母体 AO overlap metric 下投影掉全局固定 impurity 分量，并对 bath block 做 metric orthonormalization。
8. `FDMET.build()` 现在会设置：
   - `es_orb = [fixed impurity orbitals, merged fragment bath orbitals]`
   - `fo_orb = []`
   - `fv_orb =` 固定 impurity 和 bath 的全局正交补空间
   - `es_int1e` 和 `es_int2e`，用于后续嵌入计算
   - `es_mf = FDMET.CAHF(...)`，嵌入空间 mean-field 默认使用 CAHF 而不是 ROHF

当前 `fo_orb` 明确为空。这不是临时占位，而是当前默认算法选择：fragment-local occupied orbitals 来自多个重叠的局部子体系，不能直接合并为全局 frozen occupied space，否则容易重复计数电子。当前实现令 `nfo = 0`，embedded electron number 等于全体系电子数。若 `impurity + bath` 轨道数不足以容纳该电子数，代码应直接报错。

默认 `es_occ` 使用 embedded space 内的 Aufbau-like 初猜，占据数只用于构造 embedded CAHF 的初始密度。它不是由全局 reference density 对角化得到的自然占据数。fragment bath 的自然占据数目前只作为诊断信息保留。

`FDMET.CAHF()` 仿照 `SSDMET.ROHF()` 构造嵌入空间 mean-field 对象。它接收和 `fragment_scf_options` 一致的常用 CAHF/SCF 参数，例如 `ncas`、`nelecas`、`cahf_spin`、`max_cycle`、`level_shift` 和 `diis`。由于嵌入空间没有 PySCF AO labels，若 embedded CAHF 的 RDIIS 参数中提供的是 AO label/pattern 形式的 `rdiis_imp_idx`，当前实现会改用嵌入空间中的 impurity block index。

### Fragment-density embedded CAHF initial guess

`FDMET` 现在支持可选的 fragment-density embedded CAHF 初猜：

```python
mydmet = fragment.FDMET(
    mol,
    imp_idx="Co.*",
    embedded_init_guess="fragment_density",
    embedded_active_aolabels="Co 3d",
    ...
)
```

该初猜只改变 embedded CAHF 的初始 `mo_coeff/mo_occ/dm`，不改变 CAHF 方程。算法为：

1. 从每个 fragment-local CAHF reference 提取 AO density。
2. 将各 fragment density 嵌回母体 AO basis；ligand-fragment 内部 block 和 impurity-ligand cross block 来自对应 fragment，重复的 impurity block 在所有 fragment 之间平均，不同 ligand fragment 之间的 cross block 默认为 0。
3. 将拼接 density 投影到 embedded basis，并将 trace rescale 到 embedded electron number。
4. 用 `embedded_active_aolabels` 在固定 impurity block 中选定 active orbitals。CoSH4 当前使用 `Co 3d`，正好对应 5 个 active AO；不要用 `Co.*d` 作为 active label，因为 def2-TZVP 中这会选到多组 d 函数。
5. 对剩余环境 density block 对角化，按 occupation-like eigenvalue 从大到小排序；前 `ncore = (nelectron - nelecas) // 2` 个作为 core，其余作为 virtual。
6. 设置 embedded CAHF 初猜：

```text
mo_coeff = [environment core natural orbitals, fixed active AO, environment virtual natural orbitals]
mo_occ   = [2, ..., 2, nelecas/ncas, ..., nelecas/ncas, 0, ..., 0]
```

CoSH4 测试中，`embedded_active_aolabels="Co 3d"` 选择 embedded indices `[18, 19, 20, 21, 22]`。fragment-projected density 的 embedded trace 从 `95.0346402911` rescale 到 `97`，环境 core/virtual 边界为 `core min = 1.12448784038`、`first virtual = 0.379031865686`。embedded CAHF 从该初猜开始的 `init E = -3661.55677969186`，第 73 cycle 收敛到 `E = -3669.30768115344`。相比旧的 `minao` 初猜，该路径避免了从很差的一体初猜开始的大幅能量跳变。

## 开放问题

1. CAHF 平均哪些自旋态和构型？是否需要对不同配体 fragment 使用同一套 averaging scheme？
2. fragment 子体系的电子数和电荷如何定义，尤其是共价配体或带电配体？
3. 截断配体时是否需要 link atom 或 embedding potential 来稳定局部计算？
4. bath orbitals 合并后是否需要限制总 bath 数量，例如通过 occupation ranking？
5. fragment reference density 是否需要与 DMET correlation potential 自洽更新？
6. 固定 impurity 的 fragment bath 与全局 Lowdin bath 的差异如何定量比较？
7. 该方法与现有全局 HF bath 的误差应如何 benchmark？

## 测试计划

1. 小模型体系：比较标准全局 HF bath 与 fragment-based HF bath。
2. 同一小模型：比较 fragment-based HF bath 与 fragment-based CAHF bath。
3. 过渡金属和镧系配合物：检查 fragment CAHF 是否比普通 HF 更容易收敛。
4. 检查 bath orbitals 的正交性、线性相关性和环境自然占据数谱。
5. 比较 embedded cluster size、能量、局部自旋和 $d/f$ orbital occupation。

运行测试前需要激活 conda 环境：

```bash
conda activate pyscf
```
