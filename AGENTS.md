# Agent Instructions

This file defines development rules for AI agents working on this repository.

## Project Overview

`embed_sim` is a PySCF-based Python package that implements Density Matrix Embedding Theory (DMET) calculations for single-ion magnets (transition metal and lanthanide complexes). There is no `setup.py` or `pyproject.toml`; the package is imported directly as `embed_sim.<module>`.

### Modules

| Module | Role |
|---|---|
| `ssdmet.py` | Core single-shot DMET. `SSDMET` class, Lowdin orthogonalization, bath construction, embedded space integrals, checkpointing. |
| `fragment.py` | Fragment-based DMET. `FDMET(SSDMET)`, fragment bath construction, fragment-density initial guess. **Active development.** |
| `cahf.py` | Configuration-averaged Hartree-Fock. `CAHF(ROHF)` with fractional occupation Roothaan equations. |
| `aodmet.py` | AO-based DMET variant. `AODMET(SSDMET)`, environment-only orthogonalization. |
| `df.py` | Density-fitting extensions. `DFSSDMET`, `DFAODMET`, `DFNEVPT`, `DFSISO`. |
| `siso.py` | Spin-orbit coupling. `SISO` class, SOC integrals, Hamiltonian diagonalization, transition properties. |
| `rdiis.py` | Regularized DIIS convergence accelerator. `RDIIS(lib.diis.DIIS)`. |
| `myavas.py` | AVAS active space construction (modified from PySCF). |
| `sacasscf_mixer.py` | SA-CASSCF solver setup with spin-state list and NEVPT2. |
| `spin_utils.py` | Spin operators, Weyl state counting, ZFS/Zeeman Hamiltonians. |

### Class Hierarchy

- `SSDMET(lib.StreamObject)` → `FDMET`, `DFSSDMET`
- `AODMET(SSDMET)` → `DFAODMET`
- `CAHF(scf.rohf.ROHF)`
- `RDIIS(lib.diis.DIIS)`
- `SISO` (standalone)

### Design Documents

- `fragment.md` contains algorithm derivations for fragment bath construction (in Chinese). Keep it synchronized with `fragment.py` changes.

## General Rules

- Read relevant existing code before making changes.
- Follow the style and structure of existing code as closely as possible.
- Prefer small, focused additions over modifying existing modules.
- Avoid unrelated refactors. Do not change files outside the requested task.
- Do not automatically create commits.
- Do not add new Python dependencies. The existing stack is PySCF, NumPy, SciPy, h5py, sympy, and prettytable.

## Coding Conventions

### Placement & Scope

- Fragment-based bath orbital code goes in `embed_sim/fragment.py`.
- Minimize changes to existing modules. Touch them only when an integration point requires it.
- When adding new code, mimic nearby patterns, naming, and organization.
- Prefer explicit, readable code over clever abstractions.
- Keep temporary or experimental implementation details easy to inspect and debug.

### Formatting

- **Indentation**: 4 spaces. No tabs.
- **Line length**: Target ~120 characters. Older modules may exceed this; do not reformat old lines unless modifying them.
- **Blank lines**: Two blank lines between top-level definitions. One blank line between methods.

### Naming

- **Functions and variables**: `snake_case` (e.g., `build_embeded_subspace`, `fo_orb`, `es_int1e`).
- **Classes**: `PascalCase` (e.g., `SSDMET`, `FDMET`, `CAHF`, `SISO`, `RDIIS`).
- **Domain abbreviations**: Use uppercase for well-known acronyms — SSDMET, FDMET, CAHF, AVAS, RDIIS, DMET, SCF, HF, ROHF, CASSCF, NEVPT2, SISO, ZFS.
- **Short variable names**: Consistent with QC conventions — `dm` (density matrix), `mf` (mean-field object), `mol` (molecule), `es_orb` (embedded space orbitals), `fo_orb`/`fv_orb` (frozen occupied/virtual), `imp_idx` (impurity indices), `ncas`/`nelecas` (active space parameters).

### Imports

- Order: standard library → third-party (numpy, scipy, h5py) → PySCF → local (`embed_sim.*`).
- Standard aliases: `import numpy as np`, `from functools import reduce`.
- PySCF imports: `from pyscf import gto, scf, ao2mo, lib`, `from pyscf.lib import logger`.
- Local imports: `from embed_sim import ssdmet`, `from embed_sim import cahf`. Often deferred inside functions to avoid circular imports.
- Do not reorder or reformat imports in existing files unless adding new ones.

### Comments & Docstrings

- **All code comments in English.** Explain mathematical intent, assumptions, dimensions, and conventions when not obvious.
- **Design documents in Chinese** (fragment.md, etc.) unless requested otherwise.
- **Docstrings**: Use `"""triple double quotes"""`. New code should follow NumPy-style docstrings (see `FDMET` class in `fragment.py` as the reference pattern).
- Existing modules have minimal docstrings — do not add docstrings to old code unless modifying that function or class.
- Add detailed comments for nontrivial numerical, quantum-chemistry, or embedding logic.

### Logging

- Use PySCF `logger` (`from pyscf.lib import logger`) for diagnostic output. Follow `logger.info`, `logger.debug`, `logger.warn` levels.
- Pattern: `log = logger.new_logger(self, verbose)` inside methods, or `logger.info(self, 'message %s', value)`.
- Avoid bare `print()` in new code. Existing `print()` calls in older modules are legacy; do not refactor them unless modifying those functions.

### Type Hints

- `fragment.py` uses limited type hints (e.g., `@dataclass` field types).
- New code is encouraged to use type hints for function signatures, especially for public functions and class methods.
- Do not add type hints to existing functions in older modules unless refactoring them.

## Numerical and Quantum-Chemistry Code

- Be strict during development to make debugging easier.
- Check matrix shapes, orbital counts, electron counts, and index ranges when practical.
- Raise errors directly for inconsistent numerical states, invalid dimensions, unsupported basis conventions, or electron-count mismatches.
- Prefer fail-fast behavior during development over warnings that allow execution to continue with questionable data.
- Make overlap-metric conventions explicit.
- Be careful about AO, orthogonal AO, localized orbital, active orbital, impurity, bath, and environment bases.
- Do not silently discard linearly dependent orbitals or small singular values without making the threshold clear.
- Prefer explicit thresholds such as `svd_tol`, `lindep_tol`, or `orth_tol` over hidden constants.
- Check Hermiticity or symmetry of density matrices, Fock matrices, and projected Hamiltonians when practical.
- Preserve impurity definitions consistently. For fragment-based bath construction, use the existing `preserve_imp` implementation by default.
- Development code may be more defensive than final production code. Strict checks can be removed later to simplify the final human-readable implementation.

## Fragment Bath Development

- Keep `fragment.md` synchronized with implementation changes related to fragment-based bath orbital construction.
- Document algorithmic changes in `fragment.md` with detailed formula derivations when possible.
- In the code or implementation-record section of `fragment.md`, summarize the current code changes at a high level only. Do not make this section overly detailed.
- When implementing fragment bath logic, clearly state which basis each quantity lives in.
- Avoid introducing a second implementation of impurity-preserving orthogonalization. Reuse the existing `preserve_imp` path.
- For CAHF, fragment SCF, density construction, bath construction, and embedded cluster construction, record assumptions and unresolved questions in `fragment.md`.
- Example coverage can be minimal at first. A single example with one fragment is enough unless more examples are explicitly requested.

## Testing

- Activate the `pyscf` conda environment before running tests: `conda activate pyscf`.
- There is no formal test framework. Validation is done through example scripts in `examples/`.
- To syntax-check a module: `python -m py_compile embed_sim/fragment.py`.
- To test fragment DMET: `python examples/fragment_dmet.py`.
- When no project-level test command exists, run the smallest relevant import, example, or numerical sanity check.
- Report which checks were run and whether they passed.
- Testing conventions may change later.

## Git and Workspace Hygiene

- Do not commit changes unless explicitly asked.
- Do not revert user changes.
- Do not modify unrelated files.
- Before editing a file, inspect the relevant surrounding code.
- Avoid broad rewrites or restructuring unless explicitly requested.
- If existing uncommitted changes affect the task, work with them rather than overwriting them.
