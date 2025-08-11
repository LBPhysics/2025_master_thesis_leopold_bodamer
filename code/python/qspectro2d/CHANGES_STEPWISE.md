# Stepwise Refactor Change Log

This file tracks incremental changes performed during the current refactor / simplification session (branch: t_det_max_independent_of_T_wait). Each entry is atomic so you can craft a clean commit history or squash later.

Date: 2025-08-09

## Completed Changes

1. Simulation module split (earlier session)
   - Added `core/simulation/` package with `config.py` and `builders.py`.
   - Moved `SimulationConfig`, `SimulationModuleOQS`, and `H_int_` out of `simulation_class.py`.
   - Introduced backward-compatible imports in legacy `simulation_class.py`.

2. Removed SolverType enum (simplification)
   - Deleted enum usage; solver now validated by simple membership in `SUPPORTED_SOLVERS`.
   - Updated decay-channel selection logic in `builders.py` to string comparisons.

3. Added caching to `AtomicSystem`
   - `eigenstates`, `sm_op`, `dip_op` converted to `@cached_property` with invalidation in `update_freqs_cm`.
   - Ensured safe copy of eigenvalues when shifting under RWA in `H0_diagonalized`.

4. Created paper & Redfield separation
   - Added `simulation/liouvillian_paper.py` (moved `matrix_ODE_paper` + helpers).
   - Added `simulation/redfield.py` (moved `R_paper` + helpers).
   - Removed original implementations from `simulation_class.py`.
   - Updated `simulation/__init__.py` to export `matrix_ODE_paper` and `R_paper`.

5. Legacy module cleanup
   - Previously slimmed `simulation_class.py` (compat layer).
   - (NEW) Removed `simulation_class.py` entirely; users must import from `qspectro2d.core.simulation`.

6. Interaction & decay channel simplification
   - `H_int_` retained (still minimal) with clear RWA vs non-RWA branches (no extra abstraction layers).
   - Decay channel logic restricted to ME / BR else empty (paper solvers manage their own structures).

7. Duplicate unit conversion removed
   - Deleted local `convert_cm_to_fs` definition inside `atomic_system/system_class.py`.
   - Now importing canonical converter from `utils/units_and_rwa.py`.

8. Added cache reset + corrected dimer mixing angle (current step)
   - Implemented `AtomicSystem.reset_cache()` centralizing cached property invalidation.
   - Updated `update_freqs_cm` to call `reset_cache()`.
   - Replaced non-standard `theta` formula with standard definition: θ = 0.5 * arctan2(2J, Δ).
   - Added detailed docstring explaining symbols and usage.

## Pending / Proposed Next Steps (awaiting user go)
A. (DONE) Add `AtomicSystem.reset_cache()` & improve `theta` docstring (correct formula clarity).
B. (DONE) Vectorize pulse field functions (`pulse_envelope`, `E_pulse`, `Epsilon_pulse`) for ndarray input without Python loops.
C. (DONE) Refactor summary methods to return strings (no side-effect printing) and removed debug print in `me_decay_channels`.
D. (DONE) Added one-time DeprecationWarning emission in legacy `simulation_class.py`.
E. (DONE) Fully removed internal usage of `simulation_class.py`; replaced file content with raising stub forcing import migration.
F. (DONE) Pulse invariant caching: precomputed _t_start/_t_end, _sigma, _boundary_val in `LaserPulse`; updated `pulse_envelope` to use them.
   - Adjusted gaussian pulse cached window to use `active_time_range` (≈1% cutoff) while keeping cos2 at strict ±FWHM.
G. (DONE) Introduced `_single_pulse_envelope` helper.
   - Removed temporary dynamic sequence hack in `E_pulse` / `Epsilon_pulse`.
   - Centralized single-pulse logic; reduced repeated attribute lookups & clarified flow.
H. (DONE) Documentation update: Added high-level Simulation Pipeline section to root `README.md` (remaining: docstring uniformity & solver registry docs – future steps).
I. (DONE) Added `test_laser_fcts.py` covering single-pulse envelope (cos2/gaussian), combined envelopes, field functions, vectorization parity, and error handling.
J. (DONE) Added visualization helper `visualize_pulse_envelopes_and_fields()` to `test_laser_fcts.py` (interactive optional plots comparing cos2 vs Gaussian; lightweight sanity asserts preserved).
K. (DONE) Gaussian envelope semantics update: baseline now set at extended active window edge (±n_fwhm*FWHM) retaining tails between ±FWHM and edge; updated docstrings & tests.
L. (DONE) Top-level package __init__ refactored to lazy-import core symbols via __getattr__ to eliminate circular import warnings involving AtomicSystem.
M. (DONE) Extended lazy import: replaced remaining eager imports (baths, spectroscopy, visualization, data, config) with unified `_LAZY_SYMBOLS` map for on-demand loading; suppresses initialization-time circular warnings.

## Circular Import Assessment (re: convert_cm_to_fs)
- `atomic_system/system_class.py` now imports `convert_cm_to_fs` from `utils/units_and_rwa`.
- `utils/units_and_rwa.py` does NOT import `AtomicSystem` or any module that imports `AtomicSystem` transitively.
- Therefore no new circular import is introduced by this change.

## Notes
- All edits kept minimal; no behavioral changes expected except improved maintainability.
- No additional dependencies introduced.
- Legacy public API path removed; attempting to import `qspectro2d.core.simulation_class` will now fail (intentional simplification).

(End of log – append further steps below as they are approved and implemented.)
 
Date: 2025-08-10

## Completed Changes (new)

9. Structured configuration dataclasses (experimental layer)
   - Added `config/models.py` with grouped dataclasses: AtomicConfig, LaserConfig, BathConfig, SignalProcessingConfig, SolverConfig, SimulationWindowConfig, MasterConfig.
   - Added `config/loader.py` exposing `load_config()` returning MasterConfig from legacy defaults.
   - Updated `config/__init__.py` to export new symbols (backwards compatible; flat constants untouched).
   - Validation pattern: per-section validate() plus existing `validate_defaults()` call for physics-specific warnings.
   - No behavioral change for existing code paths; purely additive API to enable gradual migration.

10. Removed flat constant exports (breaking change by request)
   - Deleted re-export of legacy simulation parameter constants from `config.__init__`.
   - Added single global `CONFIG` instance; all code now accesses structured fields.
   - Updated usages in `spectroscopy/calculations.py`, `utils/simulation_utils.py` to use `CONFIG`.
   - Redirected `HBAR` imports to `qspectro2d.constants` in builders & system_laser_class.
   - NOTE: Any external user code must migrate to structured API.

## Pending / Proposed Next Steps (awaiting user go)
N. YAML/TOML override support in `load_config(path)` with deep merge.
O. Environment variable override layer (`QSPEC_*`).
P. Aggregated validation report returning structured warnings list.
Q. Remove directory-creation side effects from `paths.py` (introduce explicit `ensure_dirs()`).
R. Migrate any remaining internal modules (if discovered) & documentation update for new API.

(End of log – append further steps below as they are approved and implemented.)

11. Added layered configuration loader + pure paths
   - Implemented YAML/TOML/JSON + environment + runtime override support in `config/loader.py` (`load_config(...)`).
   - Precedence: defaults < project file < user file < env < overrides.
   - Added deep merge utility and environment parsing (int/float/bool/JSON heuristics).
   - Introduced optional project root file discovery (`qspectro2d.config.{yml,toml,json}`).
   - Refactored `config/paths.py` to remove import-time directory creation; added `ensure_dirs()`.
   - Exported `ensure_dirs` via `config.__init__` and documented usage in README.
   - Added PyYAML dependency to `requirements.txt` for YAML support.
