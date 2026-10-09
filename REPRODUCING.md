# Reproducing the results

These instructions explain how to install the environment, regenerate every figure and table of **Sensitivity of calibration-free DESI DR2 BAO-supernova constraints to a distance-duality tilt** (Stéphane Fay, Mikel Artola and Leandros Perivolaropoulos) from the archived posteriors, rerun the inference, and validate the release. See the [README](README.md) for the paper abstract and citation.

This release contains the current explicit-calibration likelihoods, all **54 completed 3,000-live-point posterior datasets**, the five manuscript figures, and generators for Tables I and III-X and the Section IV.A and IV.B quantities. Table II (Jeffreys' scale for the AIC) is a convention and is not generated. Table IX is exported from the priors in `src/sampler.py` and the Gaussian calibrations in `src/calibration.py`. Section V is outside the scripted reproduction scope. Script and output numbers follow the manuscript.

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

The five `figures/figureN.py` scripts can also be run individually, for example `python -m figures.figure3`. Tables are saved under `results/tables/` as `.tex` and machine-readable `.json` files. `tables.md` collects all of them for reading.

| Manuscript output | Generator | Saved output |
| --- | --- | --- |
| Figure 1: epsilon in LambdaCDM + epsilon and w0waCDM + epsilon | `figures/figure1.py` | `figures/figure1.pdf` |
| Figure 2: Pantheon+ (w0, wa) contours | `figures/figure2.py` | `figures/figure2.pdf` |
| Figure 3: AIC improvement with free epsilon or curvature | `figures/figure3.py` | `figures/figure3.pdf` (and the README preview `figure3.png`) |
| Figure 4: calibrated alpha/epsilon and distance residuals | `figures/figure4.py`, `figures/calibrated_residuals.py` | `figures/figure4.pdf` |
| Figure 5: four compilation triangles | `figures/figure5.py` | `figures/figure5.pdf` |
| Table I: supernova samples | `results/make_tables.py` | `results/tables/table_01.tex` |
| Tables III-VII: model comparisons and sensitivity tests | `results/make_tables.py` | `results/tables/table_03.tex` to `table_07.tex` |
| Table VIII: BAO+SN+CC posteriors | `results/make_tables.py`, `results/posterior_tables.py` | `results/tables/table_08.tex` |
| Table IX: priors | `results/make_tables.py` | `results/tables/table_09.tex` |
| Table X: BAO+SN posteriors | `results/make_tables.py`, `results/posterior_tables.py` | `results/tables/table_10.tex` |
| Section IV.A: BAO/CC constraints and BAO+SN+CC significances | `results/make_tables.py` | `results/tables/section4a.json` |
| Section IV.B: calibrated H0, alpha and Eq. (21) | `results/calibration_table.py` | `results/tables/section4b.{tex,json,md}` |

Tables VIII and X are complete REVTeX `table*` environments. The other `.tex` files are tabular fragments that can be inserted into the manuscript's table environments. Each figure writes a `figureN_values.json` record with its plotted values and provenance. Figure 4 also saves its density grid. Shared colours, typography and paths live in `figures/common.py`.

Figure 3 compares the dark-energy AIC improvement in three settings: no extension, free epsilon in flat geometry, and free curvature with epsilon fixed to zero. Each AIC difference is the corresponding chi-square difference minus four. The curvature points use the Table V controls saved in `results/minima/controls.json`. Set `SN_BAO_MINIMA_DIR` to use a different directory of control minima, together with the matching `SN_BAO_POSTERIOR_ROOT` for baseline runs.

Figure 4 has two panels. The left shows the full explicit-calibration alpha/epsilon posterior. The right compares calibrated Pantheon+ and transverse DESI BAO distance-modulus residuals against a fixed BAO-only flat LambdaCDM reference, using the mean calibrations from `src/calibration.py`. The residual panel refits only the one-dimensional reference and the joint LambdaCDM + epsilon geometry; it performs no posterior sampling. The 18 supernova bins are simultaneous generalized-least-squares estimates using the full covariance, and the joint best-fit line uses the same binning operator. Error bars include the common absolute-magnitude or sound-horizon calibration uncertainty. The reference parameters remain fixed and the plotted estimates are correlated. `figure4_values.json` records the fits, bin edges, plotted values, full displayed covariance matrices, input hashes and caption.

Table VII quotes the baseline LambdaCDM + epsilon posterior in the first row of each sample, as its caption states; the other rows use the robustness runs. The DES-Dovekie zmin=0.025 cut equals its released selection, so it has no separate robustness run. Table VI's JSON also records the tilt under each tracer deletion, which the text discusses without tabulating: the Pantheon+ posteriors and the best-fit tilts for all samples.

## Rerun the inference

There are three disjoint job inventories, with some intentionally repeated baselines:

- **38 baseline fits:** four models, four SN compilations, with and without CC, plus LCDM/CPL fits to BAO alone, CC alone, and BAO+CC.
- **3 calibrated fits:** BAO, Pantheon+, and their joint fit, sampling independent Gaussian sound-horizon and absolute-magnitude calibrations.
- **13 robustness fits:** LCDM+epsilon with Pantheon+ single BAO tracer deletions and Pantheon+/DES-Dovekie redshift cuts.

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
python -m figures.figure3
python -m figures.figure5
python -m figures.figure4 --posterior results/reruns/calibration/joint/lcdm-epsilon/posterior.npz
```

To export new tables, supply the matching roots:

```bash
python -m results.make_tables \
  --baseline-root results/reruns/baseline \
  --calibration-root results/reruns/calibration \
  --robustness-root results/reruns/robustness \
  --output-dir results/reruns/tables
```

This uses the released control, tracer, cut and non-SN minima unless `--minima-dir` is also supplied. Those minima are independent of sampling; the robustness posterior inventory alone cannot produce the CPL improvement columns.

For a SLURM cluster with the same environment on shared storage, use `bash results/submit.sh baseline`, `bash results/submit.sh calibration`, or `bash results/submit.sh robustness`. Append site-specific `sbatch` options such as `--partition=...`. The launchers write to `results/reruns/` and require persistent shared checkpoint storage. Adapt CPU, memory and wall-time directives to your cluster. They have been checked locally for syntax; no cluster jobs were submitted.

## Regenerate likelihood minima

`src/profile_fits.py` profiles conditional nuisance amplitudes to accelerate multistart minimization and explicitly searches matter-density boundaries. This is valid for minimum chi-square when the reconstructed calibrations lie inside their stated bounds. It does **not** define posterior intervals. Posterior medians and errors always come from the explicit weighted draws.

Table V (and the curvature points of Figure 3), Table VI and Table VII use compact minimum products validated against the full explicit likelihood. The same holds for the BAO-only row of Table III and the Section IV.A values without supernovae. Their input provenance is recorded in `results/minima/PROVENANCE.json`. To recompute them from the observational inputs:

```bash
python -m results.minimize controls --output-dir results/reruns/minima
python -m results.minimize tracers --output-dir results/reruns/minima
python -m results.minimize cuts --output-dir results/reruns/minima
python -m results.minimize non-sn --output-dir results/reruns/minima
python -m results.make_tables --minima-dir results/reruns/minima --output-dir results/reruns/tables
```

These are substantial optimizations, especially the tracer inventory. Tables III-V, VIII and X otherwise use the explicitly optimized run minima directly.

The exported tables incorporate the verified numerical corrections: Table IV DES-Dovekie delta chi-square **3.36**; Table VI Union3/LRG1 reduction **54%**. The minimizer explicitly searches the matter-density prior boundaries. For DES-Dovekie at zmin=0.100, the w0waCDM minima lie on the Omega_m = 0.1 boundary, so the generated Table VII reports **0.95 and 0.82**. The manuscript quotes the interior minima there. There is no dependence on frozen manuscript table values.

## Validate

```bash
python -m unittest discover -s tests -v
python -m results.verify_release
python -m results.check_calibration --compare-runs
```

The verification command checks input hashes, all 54 run archives and configurations, representative full-likelihood evaluations from each posterior, and every saved optimized minimum. The calibration check independently integrates the separate H0 posteriors, including the Jacobian induced by the uniform H0rd prior, and checks sampled Gaussian calibrations and derived coordinates.

Archived summaries retain their original metadata, including historical execution paths and source hashes, to preserve their bytes. Current scripts locate files relative to this repository, verify input hashes, and do not read those recorded paths. `results/ARCHIVED_RUNS.json` pins the exact original posterior and summary bytes. It remains fixed if you generate new runs; use separate roots and update the archive manifest deliberately for a future release.

## Layout and licensing

```text
figures/          five plotting scripts, shared style, PDFs and provenance records
src/              shared backgrounds, likelihoods, sampling and minimization
results/          reproduction drivers and job inventories
  runs/           baseline/, calibration/, robustness/ weighted posteriors
  minima/         verified control, tracer and cutoff minima
  tables/         generated LaTeX tables and numerical JSON outputs
data/             downloader, bundled CC inputs, source notes and checksums
tests/            scientific and release-integrity checks
```

The code is licensed under MIT; observational data retain their upstream terms and attribution requirements. The [source notes](data/SOURCES.md) document all inputs, their citations, verified upstream sources and published licenses. This repository distributes pinned download instructions for the eight upstream inputs and bundles the two author-prepared CC files, derived from the manuscript references, unchanged. Union's upstream MIT notice is retained beside its download destination. `CITATION.cff` supplies the software citation metadata; the paper citation is given in the [README](README.md#citing-the-paper).
