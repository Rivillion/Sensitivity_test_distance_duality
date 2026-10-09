# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reproduction code for the paper *Sensitivity of calibration-free DESI DR2 BAO–supernova constraints to a distance-duality tilt* (Fay, Artola, Perivolaropoulos). `README.md` is the paper landing page (Figure 3 PNG, title, abstract, citation); full user-facing instructions live in `REPRODUCING.md`, and release checks are logged in `VALIDATION.md`.

## Environment and commands

Python 3.14 virtualenv at `.venv`; run everything as modules from the repository root (scripts import `src.*`, `figures.*`, `results.*`). There is no linter or build step.

```bash
python data/download_data.py              # fetch the 8 upstream inputs (hash-checked); --check for offline verification
python -m unittest discover -s tests -v   # full test suite (fast, ~4 s)
python -m unittest tests.test_manuscript_tables.ManuscriptTableTests.test_table_vii_cutoffs   # single test
python -m results.verify_release          # input hashes, all 54 archived runs, likelihood re-evaluations, saved minima
python -m results.verify_release --checksums-only
python -m results.check_calibration --compare-runs
python -m results.make_tables             # all tables -> results/tables/
python -m figures.make_figures            # all five figures; or python -m figures.figure3, etc.
```

Table and figure generation only read archived posteriors and minima; they never sample or minimize. Sampling (`results.sample_baseline|sample_calibration|sample_robustness`) and minimization (`results.minimize {controls,tracers,cuts,non-sn}`) are expensive and should always target a separate root under `results/reruns/` (git-ignored), e.g. `--output-root results/reruns/baseline`.

## Architecture

Pipeline: **likelihoods → sampler/minimizer → archived products → tables/figures**.

- `src/background.py`: flat/curved expansion history and distances (w0wa CPL dark energy, Omega_k).
- `src/likelihoods.py`: one class per dataset (`PantheonPlus`, `DES_Dovekie`, `UnionBinned`, `DESI_BAO`, `CosmicChronometer`). The distance-duality tilt `eta(z) = alpha (1+z)^epsilon` enters the SN distances; in the calibration-free analysis the BAO amplitude (`H0rd`) and SN offset (`nu`) are sampled explicitly and later marginalized.
- `src/sampler.py`: Nautilus nested-sampling engine shared by all drivers (`CombinedLikelihood`, priors in `build_prior`, run directories, resume signatures, GetDist constraints). Each run writes `posterior.npz` + `summary.json` (+ a git-ignored `sampler.hdf5` checkpoint).
- Drivers in `results/`: `sample_baseline.py` (inventory `jobs_baseline.json`, 38 runs), `sample_calibration.py` → `src/sample_section4b.py` (3 calibrated Sec. IV.B runs with Gaussian r_d / M_B priors from `src/calibration.py`), `sample_robustness.py` → `src/robustness.py` (inventory `jobs_robustness.json`, 13 runs; `group` is `tracer` or `cut`). `submit.sh`/`sampling.slurm` wrap these as SLURM arrays.
- `src/profile_fits.py` + `src/minima.py` (CLI `results/minimize.py`): profiled multistart minimization for Tables V–VII, the Figure 3 curvature points, and the non-SN values. Products are in `results/minima/*.json` with `PROVENANCE.json`. Profiling is only valid for minimum chi-square, never for posterior intervals; intervals always come from the weighted draws.
- `src/posterior_io.py` loads a run (`load_run`) for tables/figures; `src/release_io.py` handles hashing, input verification, run validation and the `repo_relative`/`resolve` path helpers.
- `results/make_tables.py` writes every table in manuscript numbering (`table_01`, `table_03`–`table_10`; Table II is a convention and is not generated) plus `section4a.json` and, via `calibration_table.py`, `section4b.*`. `posterior_tables.py` builds the REVTeX `table*` environments for Tables VIII and X.
- `figures/figureN.py` follow manuscript numbering; `figures/common.py` holds shared paths, colours, style and the compilation axis helper and must be imported before matplotlib (it sets `MPLCONFIGDIR`). Each figure also writes `figureN_values.json` with plotted values and provenance; `figure3.py` additionally writes the README PNG; `figure4.py` uses `calibrated_residuals.py` and writes `figure4_density.npz`.

Environment overrides for pointing outputs at new runs: `SN_BAO_POSTERIOR_ROOT` (baseline posteriors read by figures), `SN_BAO_FIGURE_DIR`, `SN_BAO_MINIMA_DIR`, `SN_BAO_RUNS_DIR` (sampler output); `make_tables` takes `--baseline-root`, `--calibration-root`, `--robustness-root`, `--minima-dir`, `--output-dir`.

## Release invariants

- `results/runs/` holds the archived posteriors; `results/ARCHIVED_RUNS.json` pins their exact bytes and `verify_release` checks it. Never rewrite archived run files. Adding or removing a run means updating the job inventory, the manifest, the driver task ranges (`choices=range(...)`, `submit.sh` array), the counts in `verify_release.py`, and `REPRODUCING.md`/`VALIDATION.md`.
- Archived `summary.json` files contain historical absolute paths and source hashes; these are preserved deliberately and never read. New provenance must be written with `repo_relative` (no absolute paths).
- Committed tables and figures must be reproducible byte-for-byte (figures pixel-identical) from a clean checkout by `make_tables` / `make_figures`. Regenerate and commit outputs together with any generator change. `tests/test_manuscript_tables.py` pins the manuscript values.
- Known intentional differences: for DES-Dovekie at zmin = 0.100 the generated Table VII reports the boundary minima (0.95 / 0.82) while the manuscript quotes the interior minima; leave the code as is. Table VII's first row per sample is the baseline run (the zmin = 0.025 Dovekie cut equals the released selection, so it has no separate run).
- Figure sizes target APS REVTeX (column width 3.375 in); keep font sizes fixed when changing layouts.
- Downloaded data files are git-ignored and hash-verified (`data/SHA256SUMS`); the two cosmic-chronometer files are bundled and must stay unchanged. Data licensing is documented in `data/SOURCES.md`.
