"""Command-line Nautilus sampler for the SN, BAO, and CC likelihoods."""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

# Likelihood evaluation is parallelized with processes.  Keep each worker's
# small BLAS operations single-threaded unless the caller explicitly requests
# another setting in the environment.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np  # type: ignore
from scipy.optimize import minimize  # type: ignore

try:
    from .background import DEFAULT_N_GRID, Background, _BatchedBackground, c_kms
except ImportError:
    from background import DEFAULT_N_GRID, Background, _BatchedBackground, c_kms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = Path(os.environ.get("SN_BAO_RUNS_DIR", str(PROJECT_ROOT / "results/runs/baseline")))

MODEL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "lcdm": ("Omega_m",),
    "w0wa": ("Omega_m", "w0", "wa"),
    "lcdm-epsilon": ("Omega_m", "epsilon"),
    "w0wa-epsilon": ("Omega_m", "w0", "wa", "epsilon"),
}

PRIOR_BOUNDS: dict[str, tuple[float, float]] = {
    "Omega_m": (0.1, 0.5),
    "Omega_k": (-0.5, 0.5),
    "w0": (-3.0, 0.5),
    "wa": (-4.0, 3.0),
    "epsilon": (-0.4, 0.4),
    "H0": (20.0, 120.0),
    "b": (10.0, 100.0),  # Legacy profile/calibrated diagnostics only.
    "H0rd": (c_kms / 100.0, c_kms / 10.0),
    "r_d": (50.0, 300.0),
    "nu": (10.0, 60.0),
}

PARAMETER_LABELS: dict[str, str] = {
    "Omega_m": r"\Omega_{\rm m}",
    "Omega_k": r"\Omega_k",
    "w0": r"w_0",
    "wa": r"w_a",
    "epsilon": r"\epsilon",
    "H0": r"H_0",
    "b": r"b",
    "H0rd": r"H_0 r_{\rm d}",
    "nu": r"\nu",
    "r_d": r"r_{\rm d}",
}

LIKELIHOOD_ORDER = (
    "pantheon",
    "dovekie",
    "union3",
    "union31",
    "desi",
    "cc",
)
SN_LIKELIHOODS = frozenset(
    ("pantheon", "dovekie", "union3", "union31")
)
PRIVATE_LCDM_PARAMETER = "_nautilus_auxiliary"
CHECKPOINT_FILENAME = "sampler.hdf5"
POSTERIOR_FILENAME = "posterior.npz"
SUMMARY_FILENAME = "summary.json"

N_EFF = 10_000
CREDIBLE_LEVEL = 0.68
NUMERICS_VERSION = 6
EXPLICIT_SAMPLING_VERSION = 1
CC_DATASET = "full-2022"
CC_LIKELIHOOD_VERSION = 1
CC_DATA_FILENAME = "Hz_full_2022_mod.txt"
CC_COVARIANCE_FILENAME = "covmat_Hz_full_2022.txt"
BAO_SN_LIKELIHOOD_VERSION = 2
TARGET_BATCH_PER_WORKER = 64
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 512


def positive_integer(value: str) -> int:
    """Argparse converter requiring a strictly positive integer."""
    try:
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return integer


def nonnegative_integer(value: str) -> int:
    """Argparse converter requiring a non-negative integer."""
    try:
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if integer < 0:
        raise argparse.ArgumentTypeError(
            "must be greater than or equal to zero"
        )
    return integer


def nonnegative_float(value: str) -> float:
    """Argparse converter requiring a finite, non-negative float."""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "must be finite and greater than or equal to zero"
        )
    return number


def derive_sound_horizon(b: Any, h0: Any) -> np.ndarray | float:
    """Transform the sampled BAO amplitude and Hubble scale to Mpc."""
    amplitude = np.asarray(b, dtype=np.float64)
    hubble = np.asarray(h0, dtype=np.float64)
    if (np.any(~np.isfinite(amplitude)) or np.any(~np.isfinite(hubble))
            or np.any(amplitude <= 0.0) or np.any(hubble <= 0.0)):
        raise ValueError("b and H0 must be finite and strictly positive.")
    result = c_kms / (amplitude * hubble)
    if result.ndim == 0:
        return float(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Nautilus nested sampling for SN, DESI BAO, and the "
            "33-point full-2022 cosmic-chronometer likelihood."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_PARAMETERS),
        help="Cosmological model to sample.",
    )
    parser.add_argument(
        "--curvature-on",
        action="store_true",
        help=(
            "Sample Omega_k with a uniform (-0.5, 0.5) prior "
            "(default: spatially flat with Omega_k=0)."
        ),
    )
    parser.add_argument(
        "--likelihoods",
        required=True,
        nargs="+",
        choices=LIKELIHOOD_ORDER,
        help=(
            "One or more likelihoods. Select at most one supernova sample "
            "because cross-covariances are not available."
        ),
    )
    parser.add_argument(
        "--zmin",
        type=nonnegative_float,
        help=(
            "Retain Pantheon+ or DES-Dovekie supernovae with zHD >= ZMIN. "
            "When omitted, each likelihood keeps its historical cutoff."
        ),
    )
    parser.add_argument(
        "--cc-h0-mode",
        choices=("explicit", "profile", "marginalized"),
        default="explicit",
        help=(
            "CC H0 treatment: sampling requires explicit mode. Analytic "
            "profile and marginalized modes are reserved for reference calculations."
        ),
    )
    parser.add_argument(
        "--cc-data-path",
        type=Path,
        help=(
            "Override the three-column CC table. The default is "
            "data/Hz_full_2022_mod.txt."
        ),
    )
    parser.add_argument(
        "--cc-covariance-path",
        type=Path,
        help=(
            "Override the supplied full CC covariance (default: "
            "data/covmat_Hz_full_2022.txt)."
        ),
    )
    parser.add_argument(
        "--cores",
        required=True,
        type=positive_integer,
        help="Number of CPU processes to use.",
    )
    parser.add_argument(
        "--nlive",
        type=positive_integer,
        default=3000,
        help="Number of Nautilus live points (default: 3000).",
    )
    parser.add_argument(
        "--n-batch",
        type=positive_integer,
        help=(
            "Nautilus likelihood batch size. By default it is selected "
            "from --cores."
        ),
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_integer,
        help="Optional Nautilus random seed for reproducible runs.",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save/resume the checkpoint and posterior products; use "
            "--no-save for an in-memory run (default: save)."
        ),
    )
    exploration_group = parser.add_mutually_exclusive_group()
    exploration_group.add_argument(
        "--discard-exploration",
        dest="discard_exploration",
        action="store_true",
        help="Discard Nautilus exploration samples (default).",
    )
    exploration_group.add_argument(
        "--keep-exploration",
        dest="discard_exploration",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep Nautilus exploration samples.",
    )
    parser.set_defaults(discard_exploration=True)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Replace this dataset/model run instead of resuming it.",
    )
    return parser


def canonicalize_likelihoods(names: Iterable[str]) -> tuple[str, ...]:
    """Return likelihood names in a stable directory/configuration order."""
    supplied = tuple(names)
    if len(set(supplied)) != len(supplied):
        raise ValueError("Each likelihood may be selected only once.")
    selected_sn = SN_LIKELIHOODS.intersection(supplied)
    if len(selected_sn) > 1:
        raise ValueError(
            "Supernova samples cannot be combined without supplied "
            "cross-covariances."
        )
    return tuple(name for name in LIKELIHOOD_ORDER if name in supplied)


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Validate argument combinations and normalize dataset ordering."""
    available_cores = os.cpu_count()
    if available_cores is not None and args.cores > available_cores:
        parser.error(
            f"--cores cannot exceed the {available_cores} CPUs available "
            "on this machine."
        )
    if args.fresh and not args.save:
        parser.error("--fresh cannot be used together with --no-save.")
    try:
        args.likelihoods = canonicalize_likelihoods(args.likelihoods)
    except ValueError as exc:
        parser.error(str(exc))
    if args.zmin is not None and not (
        {"pantheon", "dovekie"}.intersection(args.likelihoods)
    ):
        parser.error(
            "--zmin requires either the pantheon or dovekie likelihood."
        )
    if args.cc_h0_mode != "explicit":
        parser.error(
            "Sampling requires --cc-h0-mode explicit; analytic modes are diagnostic only."
        )
    return args


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return parser, validate_arguments(parser, args)


def require_runtime_dependencies(*, save: bool) -> None:
    """Fail before a run with an actionable missing-dependency message."""
    modules = {
        "nautilus": "nautilus-sampler",
        "getdist": "getdist",
        "pandas": "pandas",
    }
    if save:
        modules["h5py"] = "h5py"

    missing = [
        package
        for module, package in modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Missing required Python package(s): "
            f"{', '.join(missing)}. Install them in the active environment "
            f"with `python -m pip install {packages}`."
        )


def sampled_parameters(
    model: str,
    likelihoods: Sequence[str] = (),
    cc_h0_mode: str = "explicit",
    *,
    curvature_on: bool = False,
) -> tuple[str, ...]:
    """Return cosmological and shared calibration parameters for one run."""
    if cc_h0_mode != "explicit":
        raise ValueError("Sampling requires explicit CC calibration.")
    selected = set(likelihoods)
    has_sn = bool(SN_LIKELIHOODS.intersection(selected))
    parameters = [name for name in MODEL_PARAMETERS[model]
                  if name != "epsilon" or has_sn]
    if curvature_on:
        parameters.insert(1, "Omega_k")
    if "cc" in selected:
        parameters.append("H0")
        if "desi" in selected:
            parameters.append("r_d")
    elif "desi" in selected:
        parameters.append("H0rd")
    if has_sn:
        parameters.append("nu")
    return tuple(parameters)


def build_prior(
    model: str,
    prior_class: Any | None = None,
    *,
    likelihoods: Sequence[str] = (),
    cc_h0_mode: str = "explicit",
    curvature_on: bool = False,
) -> Any:
    """Build a Nautilus prior with parameters selected by the model."""
    if prior_class is None:
        from nautilus import Prior  # type: ignore

        prior_class = Prior

    prior = prior_class()
    parameters = sampled_parameters(
        model,
        likelihoods,
        cc_h0_mode,
        curvature_on=curvature_on,
    )
    for parameter in parameters:
        prior.add_parameter(parameter, dist=PRIOR_BOUNDS[parameter])

    # Nautilus 1.0.6 refuses one-dimensional sampling problems. Integrating
    # this normalized, likelihood-independent coordinate leaves Z unchanged.
    if len(parameters) == 1:
        prior.add_parameter(PRIVATE_LCDM_PARAMETER, dist=(0.0, 1.0))

    return prior


class CombinedLikelihood:
    """Pickle-friendly sum of selected likelihood objects."""

    def __init__(self, likelihoods: Sequence[Any], required_parameters=()) -> None:
        self.likelihoods = tuple(likelihoods)
        self.required_parameters = tuple(required_parameters)

    @staticmethod
    def _batch_parameter(
        parameters: dict[str, Any],
        name: str,
        default: float,
        batch_size: int,
    ) -> np.ndarray:
        value = np.asarray(
            parameters.get(name, default),
            dtype=np.float64,
        )
        if value.ndim == 0:
            return np.full(batch_size, float(value), dtype=np.float64)
        if value.shape != (batch_size,):
            raise ValueError(
                f"Batched parameter {name} must have shape "
                f"({batch_size},)."
            )
        return value

    def _call_scalar(self, parameters: dict[str, Any]) -> float:
        try:
            background = Background(
                Omega_m=parameters["Omega_m"],
                Omega_k=parameters.get("Omega_k", 0.0),
                w0=parameters.get("w0", -1.0),
                wa=parameters.get("wa", 0.0),
                epsilon=parameters.get("epsilon", 0.0),
                alpha=1.0,
                H0=parameters.get("H0"),
            )
            background.r_d = parameters.get("r_d")
            background.H0rd = parameters.get("H0rd")
            background.nu = parameters.get("nu")
            total = sum(
                getattr(likelihood, "explicit_log_likelihood", likelihood.log_likelihood)(background)
                for likelihood in self.likelihoods
            )
        except (
            FloatingPointError,
            OverflowError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            return -np.inf

        if not np.isfinite(total):
            return -np.inf
        return float(total)

    def _call_batch(
        self,
        parameters: dict[str, Any],
        omega_m: np.ndarray,
    ) -> np.ndarray:
        batch_size = omega_m.size
        try:
            calibration_valid = np.ones(batch_size, dtype=bool)
            h0_values = None
            if "H0" in parameters:
                raw_h0 = self._batch_parameter(
                    parameters,
                    "H0",
                    np.nan,
                    batch_size,
                )
                h0_valid = np.isfinite(raw_h0) & (raw_h0 > 0.0)
                calibration_valid &= h0_valid
                h0_values = np.where(h0_valid, raw_h0, 70.0)

            calibrations = {}
            for name in ("r_d", "H0rd", "nu"):
                if name in parameters:
                    raw = self._batch_parameter(parameters, name, np.nan, batch_size)
                    valid = np.isfinite(raw)
                    if name != "nu":
                        valid &= raw > 0
                    calibration_valid &= valid
                    calibrations[name] = np.where(valid, raw, 40.0 if name == "nu" else 150.0)

            background = _BatchedBackground(
                Omega_m=omega_m,
                Omega_k=self._batch_parameter(
                    parameters,
                    "Omega_k",
                    0.0,
                    batch_size,
                ),
                w0=self._batch_parameter(
                    parameters,
                    "w0",
                    -1.0,
                    batch_size,
                ),
                wa=self._batch_parameter(
                    parameters,
                    "wa",
                    0.0,
                    batch_size,
                ),
                epsilon=self._batch_parameter(
                    parameters,
                    "epsilon",
                    0.0,
                    batch_size,
                ),
                alpha=np.ones(batch_size, dtype=np.float64),
            )
            background.H0 = h0_values
            background.r_d = calibrations.get("r_d")
            background.H0rd = calibrations.get("H0rd")
            background.nu = calibrations.get("nu")
            total = np.zeros(batch_size, dtype=np.float64)
            for likelihood in self.likelihoods:
                total += np.asarray(
                    getattr(likelihood, "explicit_log_likelihood", likelihood.log_likelihood)(background),
                    dtype=np.float64,
                )
        except (
            FloatingPointError,
            OverflowError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            return np.full(batch_size, -np.inf, dtype=np.float64)

        valid = background.valid & calibration_valid & np.isfinite(total)
        return np.where(valid, total, -np.inf)

    def __call__(
        self,
        parameters: dict[str, Any],
    ) -> float | np.ndarray:
        missing = set(self.required_parameters).difference(parameters)
        if missing:
            raise ValueError(f"Missing required calibration parameters: {sorted(missing)}")
        omega_m = np.asarray(parameters["Omega_m"], dtype=np.float64)
        if omega_m.ndim == 0:
            return self._call_scalar(parameters)
        if omega_m.ndim != 1:
            raise ValueError(
                "Omega_m must be scalar or a one-dimensional batch."
            )
        return self._call_batch(parameters, omega_m)


def initialize_likelihoods(
    names: Sequence[str],
    *,
    zmin: float | None = None,
    cc_h0_mode: str = "explicit",
    cc_data_path: Path | None = None,
    cc_covariance_path: Path | None = None,
) -> CombinedLikelihood:
    """Load only the requested likelihood data."""
    try:
        from .likelihoods import (
            DESI_BAO,
            DES_Dovekie,
            PantheonPlus,
            UnionBinned,
            CosmicChronometer,
        )
    except ImportError:
        from likelihoods import (
            DESI_BAO,
            DES_Dovekie,
            PantheonPlus,
            UnionBinned,
            CosmicChronometer,
        )

    selected_cc_data_path = cc_data_path if cc_data_path is not None else DATA_DIR / CC_DATA_FILENAME
    selected_covariance_path = (
        cc_covariance_path if cc_covariance_path is not None else DATA_DIR / CC_COVARIANCE_FILENAME
    )

    pantheon_arguments: dict[str, float] = {}
    dovekie_arguments: dict[str, float] = {}
    if zmin is not None:
        pantheon_arguments["zmin"] = zmin
        dovekie_arguments["zmin"] = zmin

    factories = {
        "pantheon": lambda: PantheonPlus(
            DATA_DIR / "Pantheon+SH0ES.dat",
            DATA_DIR / "Pantheon+SH0ES_STAT+SYS.cov",
            **pantheon_arguments,
        ),
        "dovekie": lambda: DES_Dovekie(
            DATA_DIR / "DES-Dovekie_HD.csv",
            DATA_DIR / "STAT+SYS.npz",
            **dovekie_arguments,
        ),
        "union3": lambda: UnionBinned(
            DATA_DIR
            / "raw"
            / "sn"
            / "union3"
            / "mu_mat_union3_cosmo=2_mu.fits",
        ),
        "union31": lambda: UnionBinned(
            DATA_DIR
            / "raw"
            / "sn"
            / "union3"
            / "mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits",
        ),
        "desi": lambda: DESI_BAO(
            DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
            DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
            amplitude_mode=(
                "profile" if "cc" in names else "marginalized"
            ),
        ),
        "cc": lambda: CosmicChronometer(
            selected_cc_data_path,
            selected_covariance_path,
            h0_mode=cc_h0_mode,
        ),
    }
    if cc_h0_mode != "explicit":
        raise ValueError("Sampling requires explicit CC calibration.")
    required = sampled_parameters("lcdm", names)[1:]
    terms = [factories[name]() for name in names]
    for term in terms:
        term.calibration_mode = "explicit"
    return CombinedLikelihood(terms, required)


def fixed_parameters(
    model: str,
    likelihoods: Sequence[str] = (),
    cc_h0_mode: str = "explicit",
    *,
    curvature_on: bool = False,
) -> dict[str, float]:
    """Return cosmological values fixed for a model."""
    free = set(
        sampled_parameters(
            model,
            likelihoods,
            cc_h0_mode,
            curvature_on=curvature_on,
        )
    )
    fixed = {"alpha": 1.0}
    if not curvature_on:
        fixed["Omega_k"] = 0.0
    for name, value in (("w0", -1.0), ("wa", 0.0), ("epsilon", 0.0)):
        if name not in free:
            fixed[name] = value
    return fixed


def active_priors(
    model: str,
    likelihoods: Sequence[str] = (),
    cc_h0_mode: str = "explicit",
    *,
    curvature_on: bool = False,
) -> dict[str, list[float]]:
    """Return JSON-serializable prior bounds for a model."""
    return {
        parameter: list(PRIOR_BOUNDS[parameter])
        for parameter in sampled_parameters(
            model,
            likelihoods,
            cc_h0_mode,
            curvature_on=curvature_on,
        )
    }


def format_zmin(zmin: float) -> str:
    """Format a cutoff for a stable folder name with three decimals minimum."""
    value = float(zmin)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("zmin must be finite and non-negative.")
    text = f"{value:.15f}".rstrip("0")
    integer, _, fraction = text.partition(".")
    return f"{integer}.{fraction.ljust(3, '0')}"


def run_directory(
    model: str,
    likelihoods: Sequence[str],
    *,
    zmin: float | None = None,
    cc_h0_mode: str = "explicit",
    curvature_on: bool = False,
) -> Path:
    """Return the canonical output directory for a run."""
    dataset_slug = "+".join(canonicalize_likelihoods(likelihoods))
    if "cc" in likelihoods:
        dataset_slug = (
            f"{dataset_slug}" if cc_h0_mode == "explicit" else f"{dataset_slug}-{cc_h0_mode}"
        )
    if zmin is not None:
        cutoff_slug = f"{dataset_slug} cutoff"
        if curvature_on:
            cutoff_slug += ", curvature free"
        return (
            RUNS_DIR
            / "explicit_sampling"
            / cutoff_slug
            / f"zmin {format_zmin(zmin)}"
            / model
        )
    geometry_slug = (
        "Curvature free, zmin 0.01"
        if curvature_on
        else "Spatially flat, zmin 0.01"
    )
    return RUNS_DIR / dataset_slug / model if not curvature_on else RUNS_DIR / "curved" / dataset_slug / model


def cc_input_metadata(data_path=None, covariance_path=None):
    """Pin the actual CC input bytes for checkpoint compatibility."""
    table = Path(data_path) if data_path is not None else DATA_DIR / CC_DATA_FILENAME
    covariance = Path(covariance_path) if covariance_path is not None else DATA_DIR / CC_COVARIANCE_FILENAME
    return {
        "dataset": CC_DATASET, "likelihood_version": CC_LIKELIHOOD_VERSION,
        "data_path": str(table.resolve()), "covariance_path": str(covariance.resolve()),
        "data_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "covariance_sha256": hashlib.sha256(covariance.read_bytes()).hexdigest(),
    }


def resume_signature(
    model: str,
    likelihoods: Sequence[str],
    nlive: int,
    discard_exploration: bool,
    *,
    zmin: float | None = None,
    cc_h0_mode: str = "explicit",
    cc_data_path: str | None = None,
    cc_covariance_path: str | None = None,
    curvature_on: bool = False,
) -> dict[str, Any]:
    """Return fields that must agree before a checkpoint can be resumed."""
    signature = {
        "explicit_sampling_version": EXPLICIT_SAMPLING_VERSION,
        "model": model,
        "likelihoods": list(likelihoods),
        "parameters": list(
            sampled_parameters(
                model,
                likelihoods,
                cc_h0_mode,
                curvature_on=curvature_on,
            )
        ),
        "priors": active_priors(
            model,
            likelihoods,
            cc_h0_mode,
            curvature_on=curvature_on,
        ),
        "fixed_parameters": fixed_parameters(
            model,
            likelihoods,
            cc_h0_mode,
            curvature_on=curvature_on,
        ),
        "curvature_on": bool(curvature_on),
        "zmin": zmin,
        "cc": {
            **cc_input_metadata(cc_data_path, cc_covariance_path),
            "h0_mode": cc_h0_mode,
        }
        if "cc" in likelihoods
        else None,
        "nlive": nlive,
        "discard_exploration": discard_exploration,
        "n_eff": N_EFF,
        "numerics": {
            "version": NUMERICS_VERSION,
            "integration": "simpson",
            "n_grid": DEFAULT_N_GRID,
            "vectorized": True,
        },
    }
    selected = set(likelihoods)
    if SN_LIKELIHOODS.intersection(selected) or (
        "desi" in selected and "cc" not in selected
    ):
        signature["bao_sn_likelihood_version"] = BAO_SN_LIKELIHOOD_VERSION
    return signature


def adaptive_batch_size(cores: int) -> int:
    """Return a bounded batch size that keeps each worker productive."""
    return max(
        MIN_BATCH_SIZE,
        min(TARGET_BATCH_PER_WORKER * cores, MAX_BATCH_SIZE),
    )


def requested_batch_size(args: argparse.Namespace) -> int:
    """Return the explicit batch size or the core-dependent default."""
    configured = getattr(args, "n_batch", None)
    if configured is not None:
        return int(configured)
    return adaptive_batch_size(args.cores)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def load_summary(path: Path) -> dict[str, Any]:
    """Load summary metadata, reporting malformed files clearly."""
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read existing run metadata at {path}. "
            "Use --fresh to replace this run."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid existing run metadata at {path}. "
            "Use --fresh to replace this run."
        )
    return payload


def prepare_saved_run(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Prepare output paths and validate or clear an existing run."""
    cc_h0_mode = getattr(args, "cc_h0_mode", "explicit")
    cc_data_path_value = getattr(args, "cc_data_path", None)
    cc_covariance_path_value = getattr(args, "cc_covariance_path", None)
    curvature_on = bool(getattr(args, "curvature_on", False))
    zmin = getattr(args, "zmin", None)
    cc_data_path = (
        str(cc_data_path_value) if cc_data_path_value is not None else None
    )
    cc_covariance_path = (
        str(cc_covariance_path_value)
        if cc_covariance_path_value is not None
        else None
    )
    directory = run_directory(
        args.model,
        args.likelihoods,
        zmin=zmin,
        cc_h0_mode=cc_h0_mode,
        curvature_on=curvature_on,
    )
    checkpoint = directory / CHECKPOINT_FILENAME
    posterior = directory / POSTERIOR_FILENAME
    summary_path = directory / SUMMARY_FILENAME
    artifacts = (checkpoint, posterior, summary_path)

    directory.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
    elif any(artifact.exists() for artifact in artifacts):
        if not checkpoint.exists() or not summary_path.exists():
            raise RuntimeError(
                f"Incomplete saved run found in {directory}. "
                "Use --fresh to replace it."
            )
        saved_summary = load_summary(summary_path)
        expected = resume_signature(
            args.model,
            args.likelihoods,
            args.nlive,
            args.discard_exploration,
            zmin=zmin,
            cc_h0_mode=cc_h0_mode,
            cc_data_path=cc_data_path,
            cc_covariance_path=cc_covariance_path,
            curvature_on=curvature_on,
        )
        if saved_summary.get("resume_signature") != expected:
            raise RuntimeError(
                f"Saved run in {directory} is incompatible with the "
                "requested configuration. Use --fresh to replace it."
            )

    previous_summary = (
        load_summary(summary_path) if summary_path.exists() else {}
    )
    initial_summary = {
        "calibration_treatment": "explicit_sampling",
        "explicit_sampling_version": EXPLICIT_SAMPLING_VERSION,
        "status": "resuming" if checkpoint.exists() else "running",
        "created_at": previous_summary.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "model": args.model,
        "likelihoods": list(args.likelihoods),
        "parameters": list(
            sampled_parameters(
                args.model,
                args.likelihoods,
                cc_h0_mode,
                curvature_on=curvature_on,
            )
        ),
        "priors": active_priors(
            args.model,
            args.likelihoods,
            cc_h0_mode,
            curvature_on=curvature_on,
        ),
        "fixed_parameters": fixed_parameters(
            args.model,
            args.likelihoods,
            cc_h0_mode,
            curvature_on=curvature_on,
        ),
        "curvature_on": curvature_on,
        "zmin": zmin,
        "cc": {
            **cc_input_metadata(cc_data_path, cc_covariance_path),
            "h0_mode": cc_h0_mode,
        }
        if "cc" in args.likelihoods
        else None,
        "cores": args.cores,
        "nlive": args.nlive,
        "discard_exploration": args.discard_exploration,
        "n_batch": requested_batch_size(args),
        "seed": (
            previous_summary.get("seed")
            if checkpoint.exists()
            else getattr(args, "seed", None)
        ),
        "numerics_version": NUMERICS_VERSION,
        "resume_signature": resume_signature(
            args.model,
            args.likelihoods,
            args.nlive,
            args.discard_exploration,
            zmin=zmin,
            cc_h0_mode=cc_h0_mode,
            cc_data_path=cc_data_path,
            cc_covariance_path=cc_covariance_path,
            curvature_on=curvature_on,
        ),
    }
    write_json_atomic(summary_path, initial_summary)
    return checkpoint, posterior, summary_path, initial_summary


def compute_getdist_constraints(
    samples: np.ndarray,
    weights: np.ndarray,
    log_likelihoods: np.ndarray,
    parameter_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute weighted medians and 68% equal-tail limits with GetDist."""
    from getdist import MCSamples, chains  # type: ignore

    previous_load_details = chains.print_load_details
    chains.print_load_details = False
    try:
        getdist_samples = MCSamples(
            samples=samples,
            weights=weights,
            loglikes=-log_likelihoods,
            names=list(parameter_names),
            labels=[
                PARAMETER_LABELS.get(parameter, parameter)
                for parameter in parameter_names
            ],
            settings={"ignore_rows": 0},
        )
    finally:
        chains.print_load_details = previous_load_details

    constraints: dict[str, dict[str, float]] = {}
    for index, parameter in enumerate(parameter_names):
        median = float(getdist_samples.confidence(index, 0.5))
        lower, upper = getdist_samples.twoTailLimits(
            index,
            CREDIBLE_LEVEL,
        )
        lower = float(lower)
        upper = float(upper)
        constraints[parameter] = {
            "median": median,
            "lower_68": lower,
            "upper_68": upper,
            "minus": median - lower,
            "plus": upper - median,
        }
    return constraints


def print_results(
    constraints: dict[str, dict[str, float]],
    maximum_likelihood_parameters: dict[str, float],
    minimum_chi2: float,
    log_z: float,
) -> None:
    """Print the required scientific summary."""
    print("\nResults")
    print("-------")
    print("Posterior constraints (GetDist, 68% equal-tail):")
    for parameter, values in constraints.items():
        print(
            f"  {parameter} = {values['median']:.6g} "
            f"-{values['minus']:.3g} +{values['plus']:.3g}"
        )
    print("Maximum-likelihood parameters:")
    for parameter, value in maximum_likelihood_parameters.items():
        print(f"  {parameter} = {value:.6g}")
    print(f"Minimum chi^2: {minimum_chi2:.6g}")
    print(f"Bayesian evidence (ln Z): {log_z:.6g}")


def maximum_likelihood_result(
    sampler: Any,
    posterior_points: np.ndarray,
    posterior_log_likelihoods: np.ndarray,
    number_of_parameters: int,
) -> tuple[np.ndarray, float]:
    """Return the best physical point across all Nautilus evaluations."""
    try:
        unit_points = np.concatenate(sampler.points)
        all_log_likelihoods = np.concatenate(sampler.log_l)
        finite = np.isfinite(all_log_likelihoods)
        if not np.any(finite):
            raise ValueError("No finite likelihood evaluations.")
        finite_indices = np.flatnonzero(finite)
        best_index = finite_indices[
            np.argmax(all_log_likelihoods[finite_indices])
        ]
        unit_point = unit_points[best_index]
        if callable(sampler.prior):
            physical_point = sampler.prior(unit_point)
        else:
            physical_point = sampler.prior.unit_to_physical(unit_point)
        best_point = np.asarray(physical_point, dtype=np.float64)
        best_log_likelihood = float(all_log_likelihoods[best_index])
    except (AttributeError, TypeError, ValueError):
        best_index = int(np.argmax(posterior_log_likelihoods))
        best_point = np.asarray(
            posterior_points[best_index],
            dtype=np.float64,
        )
        best_log_likelihood = float(
            posterior_log_likelihoods[best_index]
        )

    return best_point[:number_of_parameters], best_log_likelihood


def profiled_maximum_likelihood_result(
    likelihood: Any,
    model: str,
    parameter_names: Sequence[str],
    initial_point: np.ndarray,
    *,
    curvature_on: bool = False,
) -> tuple[np.ndarray, float] | None:
    """Refine a non-CC best fit while profiling all linear nuisances.

    Returns ``None`` for likelihood combinations that do not implement the
    profile interface, leaving their historical maximum-likelihood path
    untouched.
    """
    terms = getattr(likelihood, "likelihoods", None)
    if not terms or not all(
        getattr(term, "supports_profiled_best_fit", False) for term in terms
    ):
        return None

    geometry_names = [
        name for name in MODEL_PARAMETERS[model] if name != "epsilon"
    ]
    if curvature_on:
        geometry_names.insert(1, "Omega_k")
    point_by_name = {
        name: float(initial_point[index])
        for index, name in enumerate(parameter_names)
    }
    initial_geometry = np.asarray(
        [point_by_name[name] for name in geometry_names],
        dtype=np.float64,
    )
    geometry_bounds = [PRIOR_BOUNDS[name] for name in geometry_names]
    profile_epsilon = "epsilon" in MODEL_PARAMETERS[model]

    def evaluate(geometry: np.ndarray, *, return_epsilon: bool = False):
        parameters = dict(zip(geometry_names, geometry, strict=True))
        try:
            background = Background(
                Omega_m=parameters["Omega_m"],
                Omega_k=parameters.get("Omega_k", 0.0),
                w0=parameters.get("w0", -1.0),
                wa=parameters.get("wa", 0.0),
                epsilon=0.0,
                alpha=1.0,
            )
            chi2 = 0.0
            epsilon_hat = 0.0
            for term in terms:
                if getattr(term, "profiles_epsilon", False):
                    term_chi2, term_epsilon = term.profiled_linear_nuisances(
                        background,
                        profile_epsilon=profile_epsilon,
                        epsilon_bounds=(
                            PRIOR_BOUNDS["epsilon"]
                            if profile_epsilon
                            else None
                        ),
                    )
                    chi2 += float(term_chi2)
                    epsilon_hat = float(term_epsilon)
                else:
                    chi2 += float(term.profile_chi2(background))
        except (
            FloatingPointError,
            OverflowError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            return (math.inf, math.nan) if return_epsilon else math.inf

        if not math.isfinite(chi2):
            return (math.inf, math.nan) if return_epsilon else math.inf
        return (chi2, epsilon_hat) if return_epsilon else chi2

    result = minimize(
        evaluate,
        initial_geometry,
        method="Powell",
        bounds=geometry_bounds,
        options={"xtol": 1.0e-7, "ftol": 1.0e-10, "maxiter": 500},
    )
    candidates = [initial_geometry]
    if np.all(np.isfinite(result.x)):
        candidates.append(np.asarray(result.x, dtype=np.float64))
    best_geometry = min(candidates, key=evaluate)
    minimum_chi2, epsilon_hat = evaluate(
        best_geometry,
        return_epsilon=True,
    )
    if not math.isfinite(minimum_chi2):
        raise RuntimeError("Profiled best-fit optimization did not converge.")

    best_point = np.asarray(initial_point, dtype=np.float64).copy()
    index_by_name = {name: index for index, name in enumerate(parameter_names)}
    for name, value in zip(geometry_names, best_geometry, strict=True):
        best_point[index_by_name[name]] = value
    if profile_epsilon:
        best_point[index_by_name["epsilon"]] = epsilon_hat
    return best_point, -0.5 * minimum_chi2


def minimize_explicit(likelihood, parameter_names, initial_point):
    """Refine all free coordinates, scaled by prior width, without profiling."""
    bounds = np.asarray([PRIOR_BOUNDS[name] for name in parameter_names])
    lower, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    initial = np.clip((np.asarray(initial_point) - lower) / width, 0, 1)

    def objective(unit):
        values = lower + np.asarray(unit) * width
        return -2.0 * likelihood(dict(zip(parameter_names, values, strict=True)))

    result = minimize(
        objective, initial, method="L-BFGS-B", jac="3-point",
        bounds=[(0, 1)] * len(initial),
        options={"ftol": 1e-13, "gtol": 1e-6, "maxiter": 1000, "maxls": 40},
    )
    candidates = [initial]
    if np.all(np.isfinite(result.x)):
        candidates.append(result.x)
    best = min(candidates, key=objective)
    chi2 = float(objective(best))
    if not np.isfinite(chi2):
        raise RuntimeError("Explicit best-fit optimization found no finite likelihood.")
    status = {"success": bool(result.success), "message": str(result.message),
              "evaluations": int(result.nfev)}
    return lower + best * width, chi2, status


def explicit_maximum_likelihood_result(likelihood, parameter_names, initial_point):
    """Return refined physical parameters and their full log likelihood."""
    point, chi2, _ = minimize_explicit(likelihood, parameter_names, initial_point)
    return point, -0.5 * chi2


def write_posterior_atomic(
    path: Path,
    *,
    samples: np.ndarray,
    weights: np.ndarray,
    log_weights: np.ndarray,
    log_likelihoods: np.ndarray,
    parameter_names: Sequence[str],
    constraints: dict[str, dict[str, float]],
    minimum_chi2: float,
) -> None:
    """Atomically save portable posterior arrays."""
    medians = np.array(
        [constraints[name]["median"] for name in parameter_names],
        dtype=np.float64,
    )
    lower = np.array(
        [constraints[name]["lower_68"] for name in parameter_names],
        dtype=np.float64,
    )
    upper = np.array(
        [constraints[name]["upper_68"] for name in parameter_names],
        dtype=np.float64,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        np.savez_compressed(
            stream,
            samples=samples,
            weights=weights,
            log_weights=log_weights,
            log_likelihoods=log_likelihoods,
            parameter_names=np.asarray(parameter_names, dtype=np.str_),
            medians=medians,
            lower_68=lower,
            upper_68=upper,
            minimum_chi2=np.asarray(minimum_chi2, dtype=np.float64),
        )
    temporary_path.replace(path)


def completed_summary(
    initial_summary: dict[str, Any],
    *,
    success: bool,
    sampler: Any,
    minimum_chi2: float,
    log_z: float,
    constraints: dict[str, dict[str, float]],
    sample_count: int,
    timings: dict[str, float | int],
) -> dict[str, Any]:
    """Add final sampler and posterior information to run metadata."""
    summary = dict(initial_summary)
    summary.update(
        {
            "status": "completed" if success else "stopped",
            "updated_at": utc_now(),
            "successful": bool(success),
            "posterior_samples": sample_count,
            "likelihood_evaluations": int(sampler.n_like),
            "effective_sample_size": float(sampler.n_eff),
            "minimum_chi2": minimum_chi2,
            "log_z": log_z,
            "constraints_68_equal_tail": constraints,
            **timings,
        }
    )
    return summary


def run_sampler(args: argparse.Namespace) -> int:
    """Initialize, run, summarize, and optionally save Nautilus."""
    from nautilus import Sampler  # type: ignore

    if args.save and not args.fresh:
        directory = run_directory(args.model, args.likelihoods, zmin=args.zmin,
                                  cc_h0_mode=args.cc_h0_mode, curvature_on=args.curvature_on)
        if (directory / "summary.json").is_file():
            saved = load_summary(directory / "summary.json")
            if saved.get("status") == "completed":
                from src.release_io import validate_completed
                expected = resume_signature(args.model, args.likelihoods, args.nlive,
                    args.discard_exploration, zmin=args.zmin, cc_h0_mode=args.cc_h0_mode,
                    cc_data_path=args.cc_data_path, cc_covariance_path=args.cc_covariance_path,
                    curvature_on=args.curvature_on)
                validate_completed(directory, expected)
                print(f"Already completed: {directory}")
                return 0
    total_start = perf_counter()
    cc_h0_mode = getattr(args, "cc_h0_mode", "explicit")
    cc_data_path = getattr(args, "cc_data_path", None)
    cc_covariance_path = getattr(args, "cc_covariance_path", None)
    curvature_on = bool(getattr(args, "curvature_on", False))
    zmin = getattr(args, "zmin", None)
    likelihood = initialize_likelihoods(
        args.likelihoods,
        zmin=zmin,
        cc_h0_mode=cc_h0_mode,
        cc_data_path=cc_data_path,
        cc_covariance_path=cc_covariance_path,
    )
    parameter_names = sampled_parameters(
        args.model,
        args.likelihoods,
        cc_h0_mode,
        curvature_on=curvature_on,
    )
    prior = build_prior(
        args.model,
        likelihoods=args.likelihoods,
        cc_h0_mode=cc_h0_mode,
        curvature_on=curvature_on,
    )

    checkpoint: Path | None = None
    posterior_path: Path | None = None
    summary_path: Path | None = None
    initial_summary: dict[str, Any] | None = None
    if args.save:
        (
            checkpoint,
            posterior_path,
            summary_path,
            initial_summary,
        ) = prepare_saved_run(args)

    batch_size = requested_batch_size(args)
    requested_seed = getattr(args, "seed", None)
    effective_seed = (
        initial_summary.get("seed")
        if initial_summary is not None
        else requested_seed
    )
    sampler = Sampler(
        prior,
        likelihood,
        n_live=args.nlive,
        n_batch=batch_size,
        vectorized=True,
        pool=None if args.cores == 1 else args.cores,
        seed=requested_seed,
        filepath=checkpoint,
        resume=args.save and not args.fresh,
    )
    initialization_seconds = perf_counter() - total_start

    print(f"\nModel: {args.model}")
    print(f"Curvature: {'free' if curvature_on else 'fixed (Omega_k=0)'}")
    print(f"Likelihoods: {', '.join(args.likelihoods)}")
    if zmin is not None:
        print(f"Supernova zmin: {zmin}")
    print(f"Parameters: {', '.join(parameter_names)}")
    if "cc" in args.likelihoods:
        print(f"CC dataset: {CC_DATASET}")
        print(f"CC H0 mode: {cc_h0_mode}")
    print(f"Cores: {args.cores}")
    print(f"Batch size: {batch_size}")
    print(f"Live points: {args.nlive}")
    print(f"Seed: {effective_seed}")
    print(f"Discard exploration: {args.discard_exploration}")
    print(f"Save data: {args.save}")

    exploration_start = perf_counter()
    exploration_n_like_start = int(sampler.n_like)
    try:
        sampler.run(
            n_eff=0,
            discard_exploration=args.discard_exploration,
            verbose=True,
        )
        exploration_seconds = perf_counter() - exploration_start
        exploration_n_like = int(sampler.n_like) - exploration_n_like_start

        sampling_start = perf_counter()
        sampling_n_like_start = int(sampler.n_like)
        success = sampler.run(
            n_eff=N_EFF,
            discard_exploration=args.discard_exploration,
            verbose=True,
        )
        sampling_seconds = perf_counter() - sampling_start
        sampling_n_like = int(sampler.n_like) - sampling_n_like_start
    except KeyboardInterrupt:
        if args.save and summary_path is not None and initial_summary is not None:
            interrupted_summary = dict(initial_summary)
            interrupted_summary.update(
                {
                    "status": "interrupted",
                    "updated_at": utc_now(),
                    "total_seconds": perf_counter() - total_start,
                    "likelihood_evaluations": int(sampler.n_like),
                }
            )
            write_json_atomic(summary_path, interrupted_summary)
            print(f"\nRun interrupted. Checkpoint retained at {checkpoint}.")
        else:
            print("\nRun interrupted. Saving was disabled; no checkpoint exists.")
        return 130

    postprocessing_start = perf_counter()
    points, log_weights, log_likelihoods = sampler.posterior(
        return_as_dict=False
    )
    samples = np.asarray(
        points[:, : len(parameter_names)],
        dtype=np.float64,
    )
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
    weights = np.exp(log_weights)

    maximum_likelihood, maximum_log_likelihood = maximum_likelihood_result(
        sampler,
        np.asarray(points, dtype=np.float64),
        log_likelihoods,
        len(parameter_names),
    )
    maximum_likelihood, minimum_chi2, optimizer_status = minimize_explicit(
        likelihood, parameter_names, maximum_likelihood)
    maximum_log_likelihood = -0.5 * minimum_chi2
    constraints = compute_getdist_constraints(
        samples,
        weights,
        log_likelihoods,
        parameter_names,
    )
    maximum_likelihood_parameters = {
        parameter: float(maximum_likelihood[index])
        for index, parameter in enumerate(parameter_names)
    }
    minimum_chi2 = float(-2.0 * maximum_log_likelihood)
    if sampler.log_z is None or not math.isfinite(float(sampler.log_z)):
        raise RuntimeError("Nautilus did not produce a finite Bayesian evidence.")
    log_z = float(sampler.log_z)

    print_results(
        constraints,
        maximum_likelihood_parameters,
        minimum_chi2,
        log_z,
    )

    if args.save:
        assert posterior_path is not None
        assert summary_path is not None
        assert initial_summary is not None
        write_posterior_atomic(
            posterior_path,
            samples=samples,
            weights=weights,
            log_weights=log_weights,
            log_likelihoods=log_likelihoods,
            parameter_names=parameter_names,
            constraints=constraints,
            minimum_chi2=minimum_chi2,
        )
        postprocessing_seconds = perf_counter() - postprocessing_start
        total_seconds = perf_counter() - total_start
        likelihood_seconds = exploration_seconds + sampling_seconds
        evaluated_in_call = exploration_n_like + sampling_n_like
        evaluation_rate = (
            evaluated_in_call / likelihood_seconds
            if likelihood_seconds > 0.0
            else 0.0
        )
        summary = completed_summary(
            initial_summary,
            success=success,
            sampler=sampler,
            minimum_chi2=minimum_chi2,
            log_z=log_z,
            constraints=constraints,
            sample_count=len(samples),
            timings={
                "initialization_seconds": initialization_seconds,
                "exploration_seconds": exploration_seconds,
                "sampling_seconds": sampling_seconds,
                "postprocessing_seconds": postprocessing_seconds,
                "total_seconds": total_seconds,
                "exploration_likelihood_evaluations": exploration_n_like,
                "sampling_likelihood_evaluations": sampling_n_like,
                "likelihood_evaluations_per_second": evaluation_rate,
            },
        )
        summary["maximum_likelihood_parameters"] = maximum_likelihood_parameters
        summary["maximum_likelihood_optimizer"] = optimizer_status
        write_json_atomic(summary_path, summary)
        print(f"Saved run products to {summary_path.parent}")
    else:
        postprocessing_seconds = perf_counter() - postprocessing_start
        total_seconds = perf_counter() - total_start

    print(
        "Timing (seconds): "
        f"init={initialization_seconds:.3f}, "
        f"exploration={exploration_seconds:.3f}, "
        f"sampling={sampling_seconds:.3f}, "
        f"post={postprocessing_seconds:.3f}, "
        f"total={total_seconds:.3f}"
    )

    return 0 if success else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser, args = parse_arguments(argv)
    try:
        require_runtime_dependencies(save=args.save)
        return run_sampler(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
