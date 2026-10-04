# Sensitivity test of distance duality

Reproduction code for **Sensitivity of calibration-free DESI DR2 BAO-supernova constraints to a distance-duality tilt**, by Stéphane Fay, Mikel Artola and Leandros Perivolaropoulos.

Repository: [Rivillion/Sensitivity_test_distance_duality](https://github.com/Rivillion/Sensitivity_test_distance_duality).

This release contains the current explicit-calibration likelihoods, all **57 completed 3,000-live-point posterior datasets**, the four manuscript figures, and generators for Tables III-X and the calibrated Section IV.B quantities. Section V is outside the scripted reproduction scope. Tables I-II specify the models and priors rather than sampled results; the parameter inventory is in `src/sampler.py`, with the Gaussian calibration values in `src/calibration.py`.

## Install

The pinned environment was tested with Python **3.14**. Run commands from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python data/download_data.py
python -m pip install -r requirements.txt
python -m results.verify_release --checksums-only
```

The dependencies cover sampling, posterior statistics, plotting and vector PDF assembly. No manuscript PDF, LaTeX installation, cluster account, old workspace or absolute source-directory path is required.

The downloader uses only the Python standard library. It fetches the eight DESI, Pantheon+, DES-Dovekie and Union inputs from pinned repository commits and verifies their exact SHA-256 hashes before installing them. Valid local copies are reused; changed copies require `--force` to replace. Downloads are excluded from Git. The two author-prepared cosmic chronometer files are bundled unchanged and checked, never downloaded or overwritten. Internet access is needed only when fetching missing inputs or installing dependencies. Run `python data/download_data.py --check` for an offline input check without scientific dependencies.

## Regenerate the published outputs

These commands read the saved weighted posterior draws and minima; they perform no nested sampling:

```bash
python -m results.make_tables
python -m figures.make_figures
```

The four `figures/figureN.py` scripts can also be run individually, for example `python -m figures.figure3`. Tables are saved as `.tex` and machine-readable `.json` under `results/tables/`. `tables.md` presents the comparison tables; `calibration.md` contains the separate-fit H0 ratio, magnitude offset and Gaussian separation. `standalone.json` contains the BAO, CC and BAO+CC posterior constraints from Section IV.A.

| Manuscript output | Generator | Saved output |
| --- | --- | --- |
| Figure 1: compilation comparison | `figures/figure1.py` | `figures/figure1.pdf` |
| Figure 2: Pantheon+ CPL contours | `figures/figure2.py` | `figures/figure2.pdf` |
| Figure 3: calibrated alpha/epsilon | `figures/figure3.py` | `figures/figure3.pdf` |
| Figure 4: four compilation triangles | `figures/figure4.py` | `figures/figure4.pdf` |
| Tables III-X | `results/make_tables.py` | `results/tables/table_03.tex` through `table_10.tex` |
| Section IV.B calibration quantities | `results/calibration_table.py` | `results/tables/calibration.{tex,json,md}` |

The LaTeX for Tables IV and X uses REVTeX `ruledtabular`. The other `.tex` files are tabular fragments that can be inserted into the manuscript's table environments. Each figure has a JSON provenance record; Figure 3 also saves its density grid.

## Rerun the inference

There are three disjoint job inventories, with some intentionally repeated baselines:

- **38 baseline fits:** four models, four SN compilations, with and without CC, plus LCDM/CPL fits to BAO alone, CC alone, and BAO+CC.
- **3 calibrated fits:** BAO, Pantheon+, and their joint fit, sampling independent Gaussian sound-horizon and absolute-magnitude calibrations.
- **16 robustness fits:** LCDM+epsilon with Pantheon+ BAO tracer deletions and Pantheon+/DES-Dovekie redshift cuts.

Use separate output directories to compare new draws with the archived results:

```bash
python -m results.sample_baseline --cores 4 --output-root results/reruns/baseline
python -m results.sample_calibration --cores 4 --output-root results/reruns/calibration
python -m results.sample_robustness --cores 4 --output-root results/reruns/robustness
```

Baseline and calibrated sampling default to 3,000 live points; the robustness driver fixes 3,000. All default to an effective sample target of 10,000 and discard exploration draws. Seeds and selections are recorded in `results/jobs_baseline.json`, `results/jobs_robustness.json`, and the calibrated sampler. Sampling is stochastic; parallel reruns need not reproduce draws byte for byte. The released posterior files give the exact figure and table inputs.

`--task-id 0` selects one baseline or robustness job; `--list` prints either inventory without sampling. `--only joint` selects one calibrated fit. See `python -m results.sample_baseline --help` and the corresponding drivers for options. Running without an output override validates and skips the released completed runs. Baseline and robustness `--fresh` explicitly replace selected outputs, so use a separate root for new production runs. The calibrated driver rejects incompatible existing runs and requires a separate root.

New inference saves `sampler.hdf5`, `posterior.npz`, and `summary.json`. Compatible interrupted runs can resume from their new checkpoints. The release includes posterior files and summaries, not the original HDF5 checkpoints; the archived runs support reporting but cannot resume the original Nautilus state. Checkpoints, caches and logs are excluded by `.gitignore`.

To plot a new baseline ensemble:

```bash
export SN_BAO_POSTERIOR_ROOT="$PWD/results/reruns/baseline"
export SN_BAO_FIGURE_DIR="$PWD/results/reruns/figures"
python -m figures.figure1
python -m figures.figure2
python -m figures.figure4
python -m figures.figure3 --posterior results/reruns/calibration/joint/lcdm-epsilon/posterior.npz
```

To export new tables, supply the matching roots:

```bash
python -m results.make_tables \
  --baseline-root results/reruns/baseline \
  --calibration-root results/reruns/calibration \
  --robustness-root results/reruns/robustness \
  --output-dir results/reruns/tables
```

This uses the released control/tracer/cut minima unless `--minima-dir` is also supplied. Those minima are independent of sampling; the robustness posterior inventory alone cannot produce the CPL improvement columns.

For a SLURM cluster with the same environment on shared storage, use `bash results/submit.sh baseline`, `bash results/submit.sh calibration`, or `bash results/submit.sh robustness`. Append site-specific `sbatch` options such as `--partition=...`. The launchers write to `results/reruns/` and require persistent shared checkpoint storage. Adapt CPU, memory and wall-time directives to your cluster. They have been checked locally for syntax; no cluster jobs were submitted.

## Regenerate likelihood minima

`src/profile_fits.py` profiles conditional nuisance amplitudes to accelerate multistart minimization and explicitly searches matter-density boundaries. This is valid for minimum chi-square when the reconstructed calibrations lie inside their stated bounds. It does **not** define posterior intervals. Posterior medians and errors always come from the explicit weighted draws.

Tables VI-IX use compact minimum products validated against the full explicit likelihood. Their input provenance is recorded in `results/minima/PROVENANCE.json`. To recompute them from the observational inputs:

```bash
python -m results.minimize controls --output-dir results/reruns/minima
python -m results.minimize tracers --output-dir results/reruns/minima
python -m results.minimize cuts --output-dir results/reruns/minima
python -m results.make_tables --minima-dir results/reruns/minima --output-dir results/reruns/tables
```

These are substantial optimizations, especially the tracer inventory. Optional `ladder`, `cc-ladder`, `non-sn`, and `information` commands provide independent diagnostics; `information` requires a preceding `ladder` in the same output directory. Tables III-V and X use the current explicitly optimized run minima directly.

The exported tables incorporate the verified numerical corrections: Table V DES-Dovekie delta chi-square **-3.36**; Table IX DES-Dovekie zmin=0.100 improvements **0.95 and 0.82**; Table VIII Union3/LRG1 reduction **54%**. The boundary searches are necessary to recover the cutoff correction. There is no dependence on frozen manuscript table values.

## Validate

```bash
python -m unittest discover -s tests -v
python -m results.verify_release
python -m results.check_calibration --compare-runs
```

The verification command checks input hashes, all 57 run archives and configurations, representative full-likelihood evaluations from each posterior, and every saved optimized minimum. The calibration check independently integrates the separate H0 posteriors, including the Jacobian induced by the uniform H0rd prior, and checks sampled Gaussian calibrations and derived coordinates.

Archived summaries retain their original metadata, including historical execution paths and source hashes, to preserve their bytes. Current scripts locate files relative to this repository, verify input hashes, and do not read those recorded paths. `results/ARCHIVED_RUNS.json` pins the exact original posterior and summary bytes. It remains fixed if you generate new runs; use separate roots and update the archive manifest deliberately for a future release.

## Layout and citation

```text
figures/          four plotting scripts, PDFs and provenance records
src/              shared backgrounds, likelihoods, sampling and minimization
results/          reproduction drivers and job inventories
  runs/           baseline/, calibration/, robustness/ weighted posteriors
  minima/         verified control, tracer and cutoff minima
  tables/         generated LaTeX tables and numerical JSON outputs
data/             downloader, bundled CC inputs, source notes and checksums
tests/            scientific and release-integrity checks
```

The code is licensed under MIT; observational data retain their upstream terms and attribution requirements. The [source notes](data/SOURCES.md) document all inputs, their citations, verified upstream sources and published licenses. This repository distributes pinned download instructions for the eight upstream inputs and bundles the two author-prepared CC files, derived from the manuscript references, unchanged. Union's upstream MIT notice is retained beside its download destination. `CITATION.cff` supplies the software citation metadata. For the manuscript citation, tag the submitted version and add its archival DOI to `CITATION.cff` when available.
