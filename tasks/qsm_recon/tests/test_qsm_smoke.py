from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tasks.qsm_recon.inversion import calcification_masks, hz_to_ppm, tv_dipole_invert
from tasks.qsm_recon.phase_fit import discover_echoes, fieldmap_from_multiecho, stack_echoes
from tasks.qsm_recon.reconstruct import pipeline

PHANTOM_DIR = Path(__file__).resolve().parents[3] / "sample_data" / "qsm" / "sub-1" / "anat"


@pytest.fixture(scope="module")
def phantom_dir():
    maker = Path(__file__).resolve().parents[3] / "sample_data" / "qsm" / "make_phantom.py"
    if not PHANTOM_DIR.is_dir():
        import subprocess
        import sys

        subprocess.check_call([sys.executable, str(maker)])
    return PHANTOM_DIR


def test_field_fit(phantom_dir):
    echoes = discover_echoes(phantom_dir)
    mag, phase, te, _, _ = stack_echoes(echoes)
    field, mask = fieldmap_from_multiecho(mag, phase, te)
    assert mask.sum() > 0
    inside = field[mask]
    assert np.nanstd(inside) > 0


def test_ppm_range(phantom_dir):
    img = pipeline(phantom_dir)
    data = np.asarray(img.dataobj)
    assert data.shape == (16, 16, 16)
    assert np.isfinite(data[data != 0]).all()
    # shouldn't be wildly off ppm scale
    assert np.abs(data).max() < 500


def test_calc_masks():
    chi = np.zeros((8, 8, 8))
    chi[3:5, 3:5, 3:5] = 0.5
    mask = np.ones_like(chi, dtype=bool)
    core, rim = calcification_masks(chi, mask, percentile=80)
    assert core.any()
