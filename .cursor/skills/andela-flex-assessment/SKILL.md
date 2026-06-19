---
name: andela-flex-assessment
description: >-
  Andela FLEX pilot tasks — geometric pharmacophore cross-docking and 7T QSM
  reconstruction. Use when working on targets.json docking, docked_poses.sdf,
  MEGRE Chi maps, or qsm-recon-challenge.
---

Andela FLEX assessment notes

Task A: geometric-pharmacophore-alignment

Cross-dock SMILES ligands into a pharmacophore pocket (interaction sites + exclusion spheres, no protein structure).

Input: /root/data/targets.json (5 targets, keep JSON key order)

Each target has smiles, interaction_sites (Donor/Acceptor/Hydrophobe/Aromatic + xyz + weight), excluded_volumes (xyz + 1.2 A radius).

Pipeline: conformers, match features by family, rigid align, clash check, score, pick best pose.

Score: sum of w_i * exp(-(d_i / 1.25)^2) where d_i is min distance from site i to the nearest matching-family ligand atom.

Clash: reject if any heavy atom is closer than 1.2 A to an exclusion center. CLASH_TOL env (default 0.1) adds a bit of slack.

Output: /root/results/docked_poses.sdf, one pose per target, same topology as input SMILES.

Stack: Python 3.10+, RDKit, Docker.


Task B: qsm-recon-challenge

Chi map from multi-echo GRE at 7T. Echo times 4, 12, 20, 28 ms from JSON sidecars.

Input: /app/data/sub-1/anat/ (.nii mag/phase + json)

Output: /app/data/derivatives/qsm/sub-1/anat/sub-1_MEGRE_Chimap.nii in ppm, same matrix size as inputs.

Care about NRMSE, tissue-specific error, calcification streaking (CalcStreak).


Run scripts

  ./scripts/run_pharm.sh
  ./scripts/run_qsm.sh
