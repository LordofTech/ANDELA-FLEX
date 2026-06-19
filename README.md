Andela FLEX pilot

Two batch CLI tasks from the assessment: pharmacophore cross-docking and QSM reconstruction.


Pharmacophore task

Reads targets.json, docks each SMILES onto the interaction sites, writes one pose per target to docked_poses.sdf.

Local (needs rdkit):

  pip install -r requirements-pharm.txt
  python -m tasks.geometric_pharmacophore.align -i sample_data/pharm/targets.json -o output/pharm/docked_poses.sdf

Docker:

  bash scripts/run_pharm.sh

Inside the assessment container the paths are /root/data/targets.json in and /root/results/docked_poses.sdf out.

For a full run on real assessment data:

  docker run --rm -e NUM_CONFS=80 -e MAX_ASSIGNMENTS=800 \
    -v /path/to/data:/root/data:ro \
    -v /path/to/output:/root/results \
    andela-flex-pharm

Put targets.json in the data folder. My finished run is in output/pharm-real/docked_poses.sdf.

Useful env vars:
  NUM_CONFS       conformers per ligand (default 100)
  CLASH_TOL       slack on the 1.2 A exclusion rule (default 0.1)
  MAX_ASSIGNMENTS cap on alignment permutations (default 3000)

Big ligands automatically get fewer conformers and a greedier search so the job doesn't run forever.


QSM task

Rebuilds a susceptibility (Chi) map from multi-echo magnitude/phase NIfTI files.

Optional phantom for local smoke tests:

  python sample_data/qsm/make_phantom.py

  pip install -r requirements-qsm.txt
  python -m tasks.qsm_recon.reconstruct --anat sample_data/qsm/sub-1/anat --out output/qsm/sub-1_MEGRE_Chimap.nii

Or just: bash scripts/run_qsm.sh

Assessment output path: /app/data/derivatives/qsm/sub-1/anat/sub-1_MEGRE_Chimap.nii


Tests

  pip install -r requirements-pharm.txt -r requirements-qsm.txt -r requirements-dev.txt
  pytest


Known limitation

QSM side is a lightweight TV dipole solver, not STI-Suite or MEDI. Works for the file layout and ppm scaling but you'll want to tune lambda and iteration count on the real MEGRE data if calcification streaking is an issue.
