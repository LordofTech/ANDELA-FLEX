"""Write a tiny synthetic MEGRE dataset for local QSM smoke tests."""

from pathlib import Path

import nibabel as nib
import numpy as np

TE_MS = [4, 12, 20, 28]
OUT = Path(__file__).resolve().parent / "sub-1" / "anat"


def main():
    shape = (16, 16, 16)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    OUT.mkdir(parents=True, exist_ok=True)

    # blob in the middle — pretend susceptibility source
    grid = np.zeros(shape, dtype=np.float64)
    grid[6:10, 6:10, 6:10] = 1.0

    for i, te in enumerate(TE_MS):
        te_s = te / 1000.0
        phase = 2 * np.pi * 40.0 * te_s * grid  # ~40 Hz offset in blob
        mag = 100.0 + 20.0 * grid
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

        stem = f"sub-1_echo-{i + 1}"
        nib.save(nib.Nifti1Image(mag.astype(np.float32), affine), str(OUT / f"{stem}_mag.nii"))
        nib.save(nib.Nifti1Image(phase.astype(np.float32), affine), str(OUT / f"{stem}_phase.nii"))

        import json

        meta = {"EchoTime": te, "MagneticFieldStrength": 7.0}
        (OUT / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")

    print("wrote", OUT)


if __name__ == "__main__":
    main()
