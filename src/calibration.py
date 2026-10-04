"""External Gaussian calibrations and physical intercept utilities."""
import math
from pathlib import Path
import numpy as np
from src.background import Background, _BatchedBackground
from src.likelihoods import DESI_BAO, PantheonPlus, _sn_log_shape
ROOT = Path(__file__).resolve().parent.parent
RD_CALIBRATION = (147.09, 0.26)  # Mpc; manuscript Table II
MB_CALIBRATION = (-19.253, 0.027)  # mag; manuscript Table II

def pantheon_and_bao() -> tuple[PantheonPlus, DESI_BAO]:
    data = ROOT / "data"
    sn = PantheonPlus(
        data / "Pantheon+SH0ES.dat",
        data / "Pantheon+SH0ES_STAT+SYS.cov",
    )
    bao = DESI_BAO(
        data / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        data / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
    )
    return sn, bao



def sn_intercept(sn: PantheonPlus, background: Background | _BatchedBackground,
                 precision_ones: np.ndarray) -> np.ndarray | float:
    """Return the physical apparent-magnitude intercept nu of Eq. (11).

    The existing SN likelihood adds ``M_fiducial`` to its shape before
    profiling the offset. Add that value back to recover the physical
    intercept, which contains the physical M_B rather than a fiducial shift.
    """
    shape = _sn_log_shape(
        background, sn.z_cmb, sn.standard_redshift_term,
        sn.tilt_redshift_term,
    )
    residual = sn.mb_obs - shape
    answer = (residual @ precision_ones) / sn.offset_norm
    return float(answer) if np.ndim(answer) == 0 else np.asarray(answer)



def weighted_correlation(x: np.ndarray, y: np.ndarray,
                         weights: np.ndarray) -> float:
    w = weights / np.sum(weights)
    dx = x - np.dot(w, x)
    dy = y - np.dot(w, y)
    return float(np.dot(w, dx * dy) / math.sqrt(
        np.dot(w, dx * dx) * np.dot(w, dy * dy)))


