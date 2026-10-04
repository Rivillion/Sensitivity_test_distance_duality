# Observational inputs

These are the exact input bytes used for the released posteriors. The public code repository bundles only the two author-prepared CC files. Run `python data/download_data.py` from the repository root to obtain the other eight files directly from pinned upstream repositories. Every file is verified against its recorded SHA-256 hash; valid existing copies are reused. Run `python data/download_data.py --check` for an offline check requiring only the Python standard library, or `python -m results.verify_release --checksums-only` to also validate the archived results after installing the scientific dependencies. Downloaded upstream files are excluded from Git by explicit `.gitignore` entries.

| Input files | Role |
| --- | --- |
| `desi_gaussian_bao_ALL_GCcomb_mean.txt`, `desi_gaussian_bao_ALL_GCcomb_cov.txt` | Thirteen DESI DR2 BAO observables and full covariance |
| `Pantheon+SH0ES.dat`, `Pantheon+SH0ES_STAT+SYS.cov` | Corrected Pantheon+ apparent magnitudes and full covariance |
| `DES-Dovekie_HD.csv`, `STAT+SYS.npz` | DES-Dovekie Hubble diagram and inverse covariance |
| `raw/sn/union3/mu_mat_union3_cosmo=2_mu.fits` | Released 22-node Union3 model-2 distance likelihood |
| `raw/sn/union3/mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits` | Released 22-node Union3.1 model-2 distance likelihood |
| `Hz_full_2022_mod.txt`, `covmat_Hz_full_2022.txt` | Supplied 33-point CC table and its full covariance |

## Provenance and attribution

Checked against public upstream sources on **2026-10-04**. All eight DESI, Pantheon+, DES-Dovekie and Union input files are **byte-identical** to the pinned upstream files below. [UPSTREAM_PROVENANCE.json](UPSTREAM_PROVENANCE.json) records their immutable download URLs, upstream paths, commits, sizes and independently compared SHA-256 hashes. These matches establish reproducible sources; they do not reconstruct the original download dates. The manuscript authors confirmed that the CC inputs are self-produced files based on the references cited below; they have no upstream download URL and are preserved unchanged in this release.

| Dataset | Verified source commit | Redistribution status |
| --- | --- | --- |
| DESI DR2 BAO | [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data/tree/bb0c1c9009dc76d1391300e169e8df38fd1096db/desi_bao_dr2), `bb0c1c9009dc76d1391300e169e8df38fd1096db` | DESI publishes a CC BY 4.0 data policy; retain attribution, citations and required acknowledgments |
| Pantheon+ | [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease/tree/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR), `c447f0fea703fcd0fff57de5000947b5ca81286b` | No explicit redistribution license located in the checked repository or release documentation |
| DES-Dovekie | [des-science/DES-SN5YR](https://github.com/des-science/DES-SN5YR/tree/c9a4fcafc4cbd19bd750dee47fc76194a45c181f/4_DISTANCES_COVMAT), `c9a4fcafc4cbd19bd750dee47fc76194a45c181f` | No explicit redistribution license located for these updated files; older DES-SN5YR archive terms do not establish their status |
| Union3 / Union3.1 | [rubind/union3_release](https://github.com/rubind/union3_release/tree/f5c387349b68f3535fdabf0fbc4f45cd407f9084), `f5c387349b68f3535fdabf0fbc4f45cd407f9084` | Upstream MIT license; David Rubin's notice retained in [raw/sn/union3/LICENSE](raw/sn/union3/LICENSE) |
| Cosmic chronometers (CC) | Author-prepared 33-point table and covariance based on the manuscript references; exact hashes in `SHA256SUMS` | Bundled unchanged at the manuscript authors' instruction; no public upstream repository |

The repository's root MIT license applies to its code. It does not replace upstream data licenses or establish permission for inputs whose terms remain unconfirmed. Public availability and scientific citation alone do not establish a redistribution license.

### DESI DR2 BAO

The mean and covariance are unmodified files from the `desi_bao_dr2` directory above. DESI's [official DR2 BAO product documentation](https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html) explicitly links to this Cobaya data directory. The [DESI data license and acknowledgments policy](https://data.desi.lbl.gov/doc/acknowledgments/) states CC BY 4.0 and requires release citations, disclosure of changes when redistributing, and its specified acknowledgments in publications or derived works. The [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) remains applicable to these data. The copies here are unchanged; redshift/tracer selections are performed by the analysis scripts.

Cite DESI DR2 Results II: *Measurements of Baryon Acoustic Oscillations and Cosmological Constraints*, [arXiv:2503.14738](https://arxiv.org/abs/2503.14738), and DESI DR2 Results I: *Baryon Acoustic Oscillations from the Lyman Alpha Forest*, [arXiv:2503.14739](https://arxiv.org/abs/2503.14739), for the included galaxy/quasar and Ly-alpha measurements. Use the acknowledgment text supplied on the official policy page in the manuscript and any applicable derived publication; the source link here is not a substitute for that text.

### Pantheon+

The two files match the upstream `Pantheon+_Data/4_DISTANCES_AND_COVAR` directory. Cite Scolnic et al., *The Pantheon+ Analysis: The Full Data Set and Light-curve Release*, [arXiv:2112.03863](https://arxiv.org/abs/2112.03863), and Brout et al., *The Pantheon+ Analysis: Cosmological Constraints*, [arXiv:2202.04077](https://arxiv.org/abs/2202.04077). The [upstream distance README](https://github.com/PantheonPlusSH0ES/DataRelease/blob/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/README) also identifies Riess et al., [arXiv:2112.04510](https://arxiv.org/abs/2112.04510), for SH0ES calibration; our likelihood excludes the Cepheid calibrators.

The repository API reports no license, its checked tree has no license/copyright file, and the checked root and distance READMEs provide no explicit redistribution grant. This is a finding about the checked sources, not a claim that no permission could exist elsewhere. This release supplies pinned download instructions and fetches the files directly from upstream; it does not bundle them in Git.

### DES-Dovekie

Cite Popovic et al., *The Dark Energy Survey Supernova Program: A Reanalysis of Cosmology Results and Evidence for Evolving Dark Energy with an Updated Type Ia Supernova Calibration*, [arXiv:2511.07517](https://arxiv.org/abs/2511.07517). The [upstream README](https://github.com/des-science/DES-SN5YR/blob/c9a4fcafc4cbd19bd750dee47fc76194a45c181f/README.md) also requests the underlying Vincenzi et al. cosmological-analysis/systematics and Sanchez et al. light-curve-release citations. Follow those references when describing the underlying DES observations and pipeline.

The checked current repository has no license/copyright file and its root and distance READMEs provide no explicit redistribution grant. [Zenodo record 12720778](https://zenodo.org/records/12720778) does declare CC BY 4.0, but contains `DES-SN5YR-1.2.zip`, dated 2024-07-11. The upstream README explicitly distinguishes this original DES-SN5YR archive from its superseding DES-Dovekie results. The Zenodo latest-version endpoint still resolves to that older archive at the audit date. Its license therefore does **not establish permission for the two Dovekie files used here**. This release supplies pinned download instructions and fetches the updated files directly from upstream; it does not bundle them in Git.

### Union3 and Union3.1

Both FITS files are unchanged copies from the pinned upstream repository, which carries the [MIT license](https://github.com/rubind/union3_release/blob/f5c387349b68f3535fdabf0fbc4f45cd407f9084/LICENSE), copyright (c) 2024 David Rubin. A byte-identical copy of that notice is included beside the FITS files and must accompany redistributed copies. Their original acquisition records and FITS layout are documented in [raw/sn/union3/README.md](raw/sn/union3/README.md).

Cite Rubin et al., *Union Through UNITY: Cosmology with 2,000 SNe Using a Unified Bayesian Framework*, [arXiv:2311.12098](https://arxiv.org/abs/2311.12098), for Union3. For Union3.1/UNITY1.8 cite Rubin et al., *Banana Split: Improved Cosmological Constraints with Two Light-Curve-Shape and Color Populations Using Union3.1+UNITY1.8*, [arXiv:2601.19854](https://arxiv.org/abs/2601.19854), and Hoyt et al., *Union3.1: Reducing Systematics in Supernova Cosmology with Self-consistent Measurements of Host Galaxy Properties for 2000 Type Ia Supernovae*, [arXiv:2601.19424](https://arxiv.org/abs/2601.19424).

### Cosmic chronometers

The manuscript authors prepared `Hz_full_2022_mod.txt` and `covmat_Hz_full_2022.txt` from the cited references and requested that both files be distributed unchanged. They are not downloaded from a public repository. The manuscript cites Moresco et al., *Unveiling the Universe with Emerging Cosmological Probes*, [arXiv:2201.07241](https://arxiv.org/abs/2201.07241), and Moresco et al., *Setting the Stage for Cosmic Chronometers II: Impact of Stellar Population Synthesis Models Systematics and Full Covariance Matrix*, [arXiv:2003.07362](https://arxiv.org/abs/2003.07362). Cite these references when using the bundled CC inputs. The source record reflects the authors' confirmation; this release does not claim that these files are an official upstream data product or independently regenerate their preparation.

No alternative CC compilation or reconstructed covariance has been substituted. The downloader verifies these files and never overwrites them, even with `--force`.

## Distribution policy

Only the two author-prepared CC numerical inputs are included in the public code distribution. DESI, Pantheon+, DES-Dovekie and Union numerical inputs are fetched directly from upstream using fixed commits and hashes. Their source notes, citations, checksums and Union's MIT notice remain versioned. Local downloaded copies may remain present for analysis but are explicitly ignored by Git. When preparing an archive independently of Git, exclude the same eight files listed in `.gitignore`.

## Selection and covariance treatment

Pantheon+ uses zHD >= 0.01, excludes Cepheid calibrators, and retains 1,580 SN with the principal covariance submatrix. DES-Dovekie retains 1,820 SN at its default selection; the stored `.npz` contains inverse covariance, which is converted to covariance before redshift cuts. Union3 and Union3.1 each use their released 22-node compressed likelihoods. Each compilation is fitted separately because the products overlap and no cross-covariance is supplied.

The CC table columns are redshift, H(z) in km/s/Mpc, and quoted uncertainty in the same units. All 33 rows, spanning z=0.070 to 1.965, are retained in file order. The covariance is used directly in the same row order; the uncertainty column is not added to it. No diagonal inflation, jitter or stellar-population covariance reconstruction is applied. The hashes identify these supplied bytes without asserting an upstream release or commit. All CC likelihoods in this release use this dataset.

For BAO tracer deletions and SN cuts, retain the principal submatrix of the covariance and factor it again. Do not slice a precision matrix or an existing Cholesky factor. The shared likelihood and robustness implementations enforce this distinction.
