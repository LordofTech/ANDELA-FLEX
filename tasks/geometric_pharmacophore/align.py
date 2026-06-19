#!/usr/bin/env python3
"""CLI entrypoint for geometric-pharmacophore-alignment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rdkit.Chem import SDWriter

from tasks.geometric_pharmacophore.align_core import prepare_output_mol, search_best_pose

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT = "/root/data/targets.json"
DEFAULT_OUTPUT = "/root/results/docked_poses.sdf"


def run(input_path: Path, output_path: Path) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("targets.json must be an object")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SDWriter(str(output_path))

    try:
        for target_id, spec in raw.items():
            log.info("docking %s", target_id)
            pose = search_best_pose(spec)
            out_mol = prepare_output_mol(pose, target_id, spec["smiles"])
            writer.write(out_mol)
            log.info("  score=%.4f clash_free=%s", pose.score, pose.clash_free)
    finally:
        writer.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="pharmacophore cross-docking")
    p.add_argument("-i", "--input", default=DEFAULT_INPUT)
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        log.error("missing input: %s", input_path)
        return 1

    run(input_path, output_path)
    log.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
