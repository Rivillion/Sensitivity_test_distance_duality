"""Generate the optimized minima used by manuscript Tables III, V-VII and Sec. IV.A.

Usage: python -m results.minimize {controls,tracers,cuts,non-sn}
controls: Table V alternative tilts and curvature (also Fig. 3);
tracers: Table VI DESI DR2 tracer deletions; cuts: Table VII SN redshift cuts;
non-sn: BAO, CC and BAO+CC alone (Table III and Sec. IV.A).
Results are written as JSON under results/minima. All reported minima are
fresh optimizations of the released likelihoods.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from src import profile_fits as pf
from src.likelihoods import CosmicChronometer
from src.likelihoods import DES_Dovekie, PantheonPlus, UnionBinned

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "minima"
SAMPLES = ("pantheon", "dovekie", "union3", "union31")
TRACERS = tuple(pf.BLOCKS.items())


def make_sn(name: str, zmin: float | None = None):
    data = ROOT / "data"
    if name == "pantheon":
        return PantheonPlus(
            data / "Pantheon+SH0ES.dat",
            data / "Pantheon+SH0ES_STAT+SYS.cov",
            **({"zmin": zmin} if zmin is not None else {}),
        )
    if name == "dovekie":
        return DES_Dovekie(
            data / "DES-Dovekie_HD.csv", data / "STAT+SYS.npz",
            **({"zmin": zmin} if zmin is not None else {}),
        )
    union = {
        "union3": "mu_mat_union3_cosmo=2_mu.fits",
        "union31": "mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits",
    }
    return UnionBinned(data / "raw" / "sn" / "union3" / union[name])


def fit_record(fit: pf.Fit) -> dict:
    return {
        "chi2": fit.chi2,
        "geometry": fit.geometry.tolist(),
        "tilt": fit.epsilon,
        "at_omega_m_boundary": any(
            abs(fit.geometry[0] - edge) < 1e-7
            for edge in pf.PRIOR_BOUNDS["Omega_m"]
        ),
    }


def fit_ladder(sn, bao) -> dict:
    """Minimize all four models and record both dark-energy improvements."""
    fits = {
        model: pf.fit(sn, bao, cpl=cpl, free_epsilon=epsilon)
        for model, cpl, epsilon in (
            ("lcdm", False, False), ("w0wa", True, False),
            ("lcdm-epsilon", False, True), ("w0wa-epsilon", True, True),
        )
    }
    chi = {key: value.chi2 for key, value in fits.items()}
    result = {
        "n_sn": len(sn.z_cmb), "n_bao": len(bao.z_obs),
        "fits": {key: fit_record(value) for key, value in fits.items()},
        "delta_de": chi["lcdm"] - chi["w0wa"],
        "delta_de_given_tilt": chi["lcdm-epsilon"] - chi["w0wa-epsilon"],
    }
    return result


def controls() -> dict:
    result = {}
    for sample in SAMPLES:
        sn = make_sn(sample)
        bao = pf.subset_bao(())
        variants = {}
        for tilt in ("linear", "saturating"):
            pf.configure_tilt(sn, tilt)
            null = pf.fit(sn, bao, cpl=False, free_epsilon=True,
                          tilt_bounds=(-10.0, 10.0))
            alternative = pf.fit(sn, bao, cpl=True, free_epsilon=True,
                                 tilt_bounds=(-10.0, 10.0))
            variants[tilt] = {
                "lcdm": fit_record(null), "w0wa": fit_record(alternative),
                "delta_de": null.chi2 - alternative.chi2,
            }
        pf.configure_tilt(sn, "log")
        # Free spatial curvature in both backgrounds, with epsilon fixed to zero.
        null = pf.fit(sn, bao, cpl=False, free_epsilon=False, curvature_on=True)
        alternative = pf.fit(sn, bao, cpl=True, free_epsilon=False, curvature_on=True)
        variants["curvature"] = {
            "lcdm": fit_record(null), "w0wa": fit_record(alternative),
            "delta_de": null.chi2 - alternative.chi2,
        }
        result[sample] = variants
    return result


def tracers() -> dict:
    result = {}
    for sample in SAMPLES:
        sn = make_sn(sample)
        result[sample] = {
            block: fit_ladder(sn, pf.subset_bao(dropped))
            for block, dropped in TRACERS
        }
    return result


def cuts() -> dict:
    result = {}
    for sample, cutoffs in (
        ("pantheon", (0.010, 0.023, 0.050, 0.100)),
        ("dovekie", (0.025, 0.050, 0.100)),
    ):
        result[sample] = {
            f"{cutoff:.3f}": fit_ladder(make_sn(sample, cutoff), pf.subset_bao(()))
            for cutoff in cutoffs
        }
    return result


def non_sn() -> dict:
    from scipy.optimize import differential_evolution, minimize_scalar
    from src.background import Background
    from src.likelihoods import DESI_BAO

    data = ROOT / "data"
    bao = DESI_BAO(data / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
                   data / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
                   amplitude_mode="profile")
    cc = CosmicChronometer(data / "Hz_full_2022_mod.txt",
                           data / "covmat_Hz_full_2022.txt",
                           h0_mode="profile")
    result = {}
    for label, terms in (("bao", (bao,)), ("cc", (cc,)), ("bao_cc", (bao, cc))):
        def objective(values, cpl):
            try:
                bg = Background(Omega_m=float(values[0]),
                                w0=float(values[1]) if cpl else -1.0,
                                wa=float(values[2]) if cpl else 0.0)
                return sum(term.profile_chi2(bg) for term in terms)
            except (ValueError, FloatingPointError, OverflowError):
                return math.inf
        null = minimize_scalar(lambda x: objective((x,), False),
                               bounds=(0.1, 0.5), method="bounded",
                               options={"xatol": 1e-12})
        alt = differential_evolution(
            lambda x: objective(x, True),
            ((0.1, 0.5), (-3.0, 0.5), (-4.0, 3.0)),
            seed=240924, popsize=20, tol=1e-10, polish=True,
        )
        result[label] = {
            "lcdm": {"chi2": float(null.fun), "geometry": [float(null.x)]},
            "w0wa": {"chi2": float(alt.fun), "geometry": alt.x.tolist()},
            "delta_de": float(null.fun - alt.fun),
        }
    return result


COMMANDS = {"controls": controls, "tracers": tracers, "cuts": cuts, "non-sn": non_sn}


