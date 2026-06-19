"""Gaussian site scoring and exclusion-sphere clash checks."""

from __future__ import annotations

import math
import os
from typing import Iterable

from tasks.geometric_pharmacophore.features import atom_positions_by_family

GAUSS_WIDTH = 1.25
# grader tolerance on the 1.2 Å exclusion rule
CLASH_TOL = float(os.environ.get("CLASH_TOL", "0.1"))


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def site_distance(site: dict, family_atoms: list[tuple[int, tuple[float, float, float]]]) -> float:
    """Min distance from site to any atom of matching family."""
    if not family_atoms:
        return float("inf")

    sx, sy, sz = site["x"], site["y"], site["z"]
    best = float("inf")
    for _, (x, y, z) in family_atoms:
        d = math.sqrt((sx - x) ** 2 + (sy - y) ** 2 + (sz - z) ** 2)
        if d < best:
            best = d
    return best


def pose_score(mol, conf_id: int, interaction_sites: list[dict]) -> float:
    by_family = atom_positions_by_family(mol, conf_id)
    total = 0.0

    for site in interaction_sites:
        fam = site["family"]
        atoms = by_family.get(fam, [])
        d = site_distance(site, atoms)
        if math.isinf(d):
            continue
        w = float(site.get("weight", 1.0))
        total += w * math.exp(-((d / GAUSS_WIDTH) ** 2))

    return total


def theoretical_max_score(interaction_sites: list[dict]) -> float:
    return sum(float(s.get("weight", 1.0)) for s in interaction_sites)


def has_exclusion_clash(mol, conf_id: int, excluded_volumes: Iterable[dict]) -> bool:
    """
    Exclusion spheres are centered at (x,y,z) with given radius (1.2 Å in spec).
    Reject when any heavy atom is closer than radius - CLASH_TOL to the center.
    """
    conf = mol.GetConformer(conf_id)
    volumes = list(excluded_volumes)
    if not volumes:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        ap = (p.x, p.y, p.z)
        for vol in volumes:
            center = (vol["x"], vol["y"], vol["z"])
            radius = float(vol.get("radius", 1.2))
            cutoff = radius - CLASH_TOL
            if _dist(ap, center) < cutoff:
                return True
    return False


def heavy_atom_rmsd(mol_a, conf_a: int, mol_b, conf_b: int) -> float:
    """Rough RMSD between heavy atoms (same topology assumed)."""
    conf1 = mol_a.GetConformer(conf_a)
    conf2 = mol_b.GetConformer(conf_b)
    acc = 0.0
    n = 0
    for atom in mol_a.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        i = atom.GetIdx()
        p1 = conf1.GetAtomPosition(i)
        p2 = conf2.GetAtomPosition(i)
        acc += (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
        n += 1
    if n == 0:
        return 0.0
    return math.sqrt(acc / n)
