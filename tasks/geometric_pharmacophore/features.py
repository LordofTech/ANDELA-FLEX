"""Map RDKit feature types to pharmacophore families."""

from __future__ import annotations

import os
from dataclasses import dataclass

from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

FAMILY_TYPES = {
    "Donor": {"Donor", "SingleAtomDonor"},
    "Acceptor": {"Acceptor", "SingleAtomAcceptor"},
    "Hydrophobe": {"Hydrophobe", "LumpedHydrophobe"},
    "Aromatic": {"Aromatic"},
}

_fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_factory = ChemicalFeatures.BuildFeatureFactory(_fdef)


@dataclass
class LigandFeature:
    family: str
    atom_ids: tuple[int, ...]
    position: tuple[float, float, float]


def family_for_type(feat_type: str) -> str | None:
    for family, types in FAMILY_TYPES.items():
        if feat_type in types:
            return family
    return None


def ligand_features(mol, conf_id: int = 0) -> list[LigandFeature]:
    conf = mol.GetConformer(conf_id)
    out: list[LigandFeature] = []

    for feat in _factory.GetFeaturesForMol(mol, confId=conf_id):
        family = family_for_type(feat.GetFamily())
        if family is None:
            continue
        pos = feat.GetPos()
        atom_ids = tuple(feat.GetAtomIds())
        out.append(
            LigandFeature(
                family=family,
                atom_ids=atom_ids,
                position=(pos.x, pos.y, pos.z),
            )
        )

    # fallback: aromatic rings sometimes missed — check ring atoms
    if not any(f.family == "Aromatic" for f in out):
        from rdkit.Chem import GetSSSR

        rings = GetSSSR(mol)
        for ring in rings:
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                xs, ys, zs = [], [], []
                for idx in ring:
                    p = conf.GetAtomPosition(idx)
                    xs.append(p.x)
                    ys.append(p.y)
                    zs.append(p.z)
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                cz = sum(zs) / len(zs)
                out.append(
                    LigandFeature(
                        family="Aromatic",
                        atom_ids=tuple(ring),
                        position=(cx, cy, cz),
                    )
                )
                break

    return out


def atom_positions_by_family(mol, conf_id: int = 0) -> dict[str, list[tuple[int, tuple[float, float, float]]]]:
    """Atom coords grouped by family (an atom can appear under multiple families)."""
    buckets: dict[str, list[tuple[int, tuple[float, float, float]]]] = {
        k: [] for k in FAMILY_TYPES
    }
    conf = mol.GetConformer(conf_id)
    seen: set[tuple[str, int]] = set()

    for lf in ligand_features(mol, conf_id):
        for aid in lf.atom_ids:
            key = (lf.family, aid)
            if key in seen:
                continue
            seen.add(key)
            p = conf.GetAtomPosition(aid)
            buckets[lf.family].append((aid, (p.x, p.y, p.z)))

    return buckets
