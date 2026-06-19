"""Edge-case helpers: assignment caps and 1-point alignment."""

from __future__ import annotations

import itertools


def capped_product(options_per_group, limit: int):
    """itertools.product with a hard stop — keeps combinatorics sane."""
    n = 0
    for combo in itertools.product(*options_per_group):
        yield combo
        n += 1
        if n >= limit:
            break


def translate_only(ref_pt, probe_pt):
    """Single-point 'alignment': shift probe so one atom hits the site."""
    return (
        ref_pt[0] - probe_pt[0],
        ref_pt[1] - probe_pt[1],
        ref_pt[2] - probe_pt[2],
    )
