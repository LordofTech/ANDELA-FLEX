import math

import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    pytest.skip("rdkit not available", allow_module_level=True)

from tasks.geometric_pharmacophore.scoring import (
    has_exclusion_clash,
    pose_score,
    site_distance,
    theoretical_max_score,
)


def test_gaussian_score_at_zero_distance():
    sites = [{"family": "Donor", "x": 0, "y": 0, "z": 0, "weight": 2.0}]
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()

    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 8:
            conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
            break

    s = pose_score(mol, 0, sites)
    assert s == pytest.approx(2.0, rel=0.05)


def test_exclusion_clash():
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()
    c_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6][0]
    conf.SetAtomPosition(c_idx, (0.0, 0.0, 0.0))

    vols = [{"x": 0.0, "y": 0.0, "z": 0.0, "radius": 1.2}]
    assert has_exclusion_clash(mol, 0, vols) is True

    conf.SetAtomPosition(c_idx, (5.0, 0.0, 0.0))
    assert has_exclusion_clash(mol, 0, vols) is False


def test_theoretical_max():
    sites = [{"weight": 1.0}, {"weight": 0.5}]
    assert theoretical_max_score(sites) == 1.5


def test_site_distance_inf_when_no_atoms():
    d = site_distance({"x": 0, "y": 0, "z": 0, "family": "Donor"}, [])
    assert math.isinf(d)
