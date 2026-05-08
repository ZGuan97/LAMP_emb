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
4. `fragment_scf`、`threshold` 等参数已经在类中保留，供后续 fragment-local bath builder 对接。
5. `examples/fragment_dmet.py` 中已经为 `Co(SH)_4` 定义分片：impurity 为 Co，`L1` 到 `L4` 分别是四个 SH ligand fragment。

这一阶段的基空间约定如下：真正的 fragment-local 实现应从每个 $\mathcal{I}\oplus\mathcal{L}_f$ 的局部 reference density 构造 bath，而不是使用 whole-system SCF density。后续 fragment bath builder 的最终输出仍应为全局 AO coefficient `es_orb`、`fo_orb` 和 `fv_orb`，以便继续复用现有 Hamiltonian projection、AVAS、`total_cas` 和 SISO 接口。

### 关键函数草案

```python
def build_fragment_baths(mol, impurity_idx, fragment_idx, fragment_scf="cahf", threshold=1e-13):
    """Build DMET bath orbitals from fragment-local reference densities."""
```

```python
def run_fragment_scf(mol, impurity_idx, fragment_idx, fragment_scf="cahf"):
    """Run HF/CAHF on one metal-ligand fragment and return density information."""
```

```python
def bath_from_fragment_density(dm, impurity_idx, fragment_idx, overlap=None, threshold=1e-13):
    """Construct bath orbitals from one fragment density matrix."""
```

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
