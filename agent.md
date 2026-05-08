# Agent Instructions

This file defines development rules for AI agents working on this repository.

## General Scope

- These instructions are intended for AI agents.
- Read the relevant existing code before making changes.
- Follow the style and structure of the existing code as closely as possible.
- Prefer small, focused additions over modifying existing modules.
- Avoid unrelated refactors.
- Do not change files outside the requested task unless the change is required.
- Do not automatically create commits.
- Write a dated coding log only when the user asks for a summary.
- Name coding log files with the format `coding_log_YYYYMMDD_HHMM.md`.
- Write coding logs in Chinese.

## Coding Style

- Minimize changes to existing modules when possible.
- For fragment-based bath orbital construction, create and implement most functionality in `embed_sim/fragment.py`.
- Touch existing modules only when an integration point is required.
- When adding new code, mimic nearby code patterns, naming conventions, and organization.
- Prefer explicit, readable code over clever abstractions.
- Add detailed comments for nontrivial numerical, quantum-chemistry, or embedding logic.
- Comments should explain mathematical intent, assumptions, dimensions, and conventions when they are not obvious.
- Keep temporary or experimental implementation details easy to inspect and debug.
- Write code and code comments in English.
- Write design documents and formula derivations in Chinese unless requested otherwise.
- Do not add new Python dependencies. Use the existing stack, especially PySCF, NumPy, and SciPy.

## Numerical And Quantum-Chemistry Code

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
- For logging and diagnostic output, mimic PySCF's logging style rather than using ad hoc print statements.

## Fragment Bath Development

- Keep `fragment.md` synchronized with implementation changes related to fragment-based bath orbital construction.
- Document algorithmic changes in `fragment.md` with detailed formula derivations when possible.
- In the code or implementation-record section of `fragment.md`, summarize the current code changes at a high level only. Do not make this section overly detailed.
- When the user asks for a summary of fragment-related work, include a `coding_log_YYYYMMDD_HHMM.md` entry with what changed, why it changed, and any important assumptions or remaining issues.
- When implementing fragment bath logic, clearly state which basis each quantity lives in.
- Avoid introducing a second implementation of impurity-preserving orthogonalization. Reuse the existing `preserve_imp` path.
- For CAHF, fragment SCF, density construction, bath construction, and embedded cluster construction, record assumptions and unresolved questions in `fragment.md`.
- Example coverage can be minimal at first. A single example with one fragment is enough unless more examples are explicitly requested.

## Testing

- Before running tests, activate the `pyscf` conda environment.
- Use `conda activate pyscf` before invoking any test command.
- Testing conventions are not fixed yet and may change later.
- When no project-level test command exists, run the smallest relevant import, example, or numerical sanity check.
- Report which checks were run and whether they passed.

## Git And Workspace Hygiene

- Do not commit changes unless explicitly asked.
- Do not revert user changes.
- Do not modify unrelated files.
- Before editing a file, inspect the relevant surrounding code.
- Avoid broad rewrites or restructuring unless explicitly requested.
- If existing uncommitted changes affect the task, work with them rather than overwriting them.
