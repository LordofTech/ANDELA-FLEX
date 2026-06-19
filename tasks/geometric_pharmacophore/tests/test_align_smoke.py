import json
from pathlib import Path

import pytest

try:
    from rdkit.Chem import SDMolSupplier
    from tasks.geometric_pharmacophore.align import run
except ImportError:
    pytest.skip("rdkit not available", allow_module_level=True)

SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "pharm" / "targets.json"


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample targets missing")
def test_align_writes_sdf(tmp_path):
    out = tmp_path / "docked_poses.sdf"
    run(SAMPLE, out)
    assert out.is_file()

    mols = list(SDMolSupplier(str(out)))
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert len(mols) == len(raw)
    for mol, key in zip(mols, raw.keys()):
        assert mol is not None
        assert mol.GetNumConformers() == 1
        assert mol.GetProp("_Name") == key
