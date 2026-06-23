---
name: submit-fragment-job
description: Submit a fragment DMET calculation job to the Tianjin supercomputer. Creates a timestamped directory under ~/dmet/CoSPh4/fragment_dmet/, writes fragment_dmet.py and run.sh, and submits via sbatch. Use when the user wants to run a CoSPh4 fragment DMET job on 天津超算.
---

# Submit Fragment DMET Job on Tianjin

Submit a CoSPh4 fragment DMET calculation to `tj1.chinahpc.com`. Connection parameters are in the `df-tj-remote` skill.

## Workflow

### 1. Get Timestamp

```bash
ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb df_iopcas_gzb@tj1.chinahpc.com 'date +%Y%m%d_%H%M'
```

### 2. Create Directory and Copy Data Files

```bash
ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb df_iopcas_gzb@tj1.chinahpc.com \
  'mkdir -p ~/dmet/CoSPh4/fragment_dmet/<TIMESTAMP> && \
   cp ~/dmet/CoSPh4/fragment_dmet/20260616_1422/{CoSPh4.xyz,CoSPh4_mag.txt,myHmat} \
      ~/dmet/CoSPh4/fragment_dmet/<TIMESTAMP>/'
```

Data files (`CoSPh4.xyz`, `CoSPh4_mag.txt`, `myHmat`) are copied from the reference run `20260616_1422`.

### 3. Write fragment_dmet.py

Key FDMET parameters for CoSPh4 (molecule: Co + 4 PhS ligands, spin=3, charge=-2):

| Parameter | Value |
|-----------|-------|
| `bas` | `{'default':'def2tzvp','s':'6-31G*','H':'6-31G*'}` |
| `imp_atoms` | `'Co'` |
| `imp_charge` | `2` |
| `ligand_atoms` | 4 ligands × 12 atoms each (indices 2-49, 0-based) |
| `ligand_charges` | `[-1, -1, -1, -1]` |
| `embedded_active_aolabels` | `'Co 3d'` |
| `keep_fv_orbitals` | `False` |
| `embedded_init_guess` | `'fragment_density'` |

#### fragment_scf modes

**`cahf` (CAHF + RDIIS):**
```python
fragment_scf='cahf',
fragment_scf_options={
    'ncas': 5, 'nelecas': 7, 'cahf_spin': 1,
    'diis': 'rdiis', 'rdiis_prop': 'dS', 'rdiis_imp_idx': ['Co.*d'],
    'rdiis_power': 0.2, 'max_cycle': 500, 'level_shift': 4.0,
},
```

**`cahf-soscf` (CAHF + SOSCF):**
```python
fragment_scf='cahf-soscf',
fragment_scf_options={
    'ncas': 5, 'nelecas': 7, 'cahf_spin': 1,
    'init_guess': 'atom', 'pre_scf_max_cycle': 0,
    'avas_aolabels': ['Co 3d'], 'avas_threshold': 0.5,
    'max_cycle': 200, 'conv_tol': 1e-9, 'level_shift': 4.0,
    'diis': 'rdiis', 'rdiis_prop': 'dS',
    'rdiis_imp_idx': ['Co.*d'], 'rdiis_power': 0.2,
},
```

### 4. Write run.sh

Template:
```bash
#!/bin/bash
#SBATCH -J CoSPh4_fdmet
#SBATCH -p p1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -t 7-00:00:00
#SBATCH -o job-%j.out

export OMP_NUM_THREADS=48
export PYTHONPATH=/data/home/df_iopcas_gzb/dmet/embed_sim:$PYTHONPATH

CONDA=/data/home/df_iopcas_gzb/anaconda3

$CONDA/envs/pyscf/bin/python fragment_dmet.py
```

Adjust `-J` (job name) suffix to reflect the fragment_scf mode, e.g. `CoSPh4_fdmet_soscf`.

### 5. Submit

```bash
ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb df_iopcas_gzb@tj1.chinahpc.com \
  'cd ~/dmet/CoSPh4/fragment_dmet/<TIMESTAMP> && sbatch run.sh'
```

## Check Job Status

```bash
ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb df_iopcas_gzb@tj1.chinahpc.com 'squeue -u df_iopcas_gzb'
```

## Reference Run

`~/dmet/CoSPh4/fragment_dmet/20260616_1422/` — baseline CAHF + RDIIS run (completed).
