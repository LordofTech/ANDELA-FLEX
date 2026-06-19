"""Dipole inversion with simple TV regularization."""

from __future__ import annotations

import numpy as np
from scipy.fft import fftn, ifftn


GAMMA_HZ_PER_T = 42.577e6


def _dipole_kernel(shape: tuple[int, ...], b0_axis: int = 2) -> np.ndarray:
    """K-space dipole kernel for B0 along axis (default z)."""
    grids = np.meshgrid(
        *[np.fft.fftfreq(n, d=1.0) for n in shape],
        indexing="ij",
    )
    k2 = sum(g**2 for g in grids)
    k2 = np.maximum(k2, 1e-12)
    kz = grids[b0_axis]
    d = (1.0 / 3.0) - (kz**2 / k2)
    d = np.where(k2 < 1e-12, 0.0, d)
    return d.astype(np.complex128)


def kspace_divide(field: np.ndarray, mask: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """Truncated division inversion in k-space."""
    d = _dipole_kernel(field.shape)
    f_k = fftn(np.where(mask, field, 0.0))
    chi_k = f_k / (d + tau)
    chi = np.real(ifftn(chi_k))
    return np.where(mask, chi, 0.0)


def _gradient(x: np.ndarray) -> list[np.ndarray]:
    out = []
    for ax in range(x.ndim):
        out.append(np.roll(x, -1, axis=ax) - x)
    return out


def _divergence(grads: list[np.ndarray]) -> np.ndarray:
    div = np.zeros_like(grads[0])
    for ax, g in enumerate(grads):
        div += g - np.roll(g, 1, axis=ax)
    return div


def tv_dipole_invert(
    field_hz: np.ndarray,
    mask: np.ndarray,
    lam: float = 0.02,
    steps: int = 80,
    tau: float = 0.05,
    calc_mask: np.ndarray | None = None,
    calc_lam_mult: float = 3.0,
) -> np.ndarray:
    """
    Iterative refinement: start with k-space division, then a few TV steps.
    Higher lambda near calcification rim to tame streaks.
    """
    chi = kspace_divide(field_hz, mask, tau=tau)
    local_lam = np.full_like(chi, lam, dtype=np.float64)
    if calc_mask is not None and np.any(calc_mask):
        local_lam = np.where(calc_mask, lam * calc_lam_mult, lam)

    d = _dipole_kernel(field_hz.shape)
    d_conj = np.conj(d)

    for _ in range(steps):
        residual_k = fftn(np.where(mask, d * chi - field_hz, 0.0))
        grad_data = np.real(ifftn(residual_k * d_conj))

        grads = _gradient(chi)
        div_tv = _divergence([local_lam * g / (np.abs(g) + 1e-8) for g in grads])

        chi = chi - 0.15 * (grad_data + div_tv)
        chi = np.where(mask, chi, 0.0)

    return chi


def hz_to_ppm(chi_hz: np.ndarray, b0_t: float = 7.0) -> np.ndarray:
    return chi_hz / (GAMMA_HZ_PER_T * b0_t) * 1e6


def calcification_masks(chi_ppm: np.ndarray, mask: np.ndarray, percentile: float = 99.5):
    """High-susceptibility blob + dilated rim for streak control."""
    from scipy.ndimage import binary_dilation, generate_binary_structure

    vals = chi_ppm[mask]
    if vals.size == 0:
        return np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool)

    thr = np.percentile(vals, percentile)
    core = (chi_ppm > thr) & mask
    struct = generate_binary_structure(chi_ppm.ndim, 1)
    rim = binary_dilation(core, structure=struct, iterations=2) & ~core & mask
    return core, rim
