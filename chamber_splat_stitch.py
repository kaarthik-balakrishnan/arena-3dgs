# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chamber-Based 3D Gaussian Splatting with Stitching
#
# Uses the naming convention `Chamber{X}_{NN}_[Chamber{Y}]_[view].jpg` to:
# 1. Run per-chamber COLMAP independently
# 2. Find 3D-3D correspondences between adjacent chambers via **transition images**
# 3. Compute rigid transforms (RANSAC on SIFT descriptor matches from binary COLMAP data)
# 4. Merge into a unified sparse model
# 5. Train a single 3DGS on the unified data (or per-chamber + stitch)
#
# **Naming convention:**
# | Pattern | Meaning |
# |---|---|
# | `Chamber1_04.jpg` | Interior of Chamber 1, no transition |
# | `Chamber2_05_Chamber3.jpg` | Taken FROM Chamber 2, Chamber 3 visible |
# | `Chamber1_17_topview.jpg` | Top-down angled view of Chamber 1 |
# | `Chamber1_20_outside.jpg` | Exterior near Chamber 1 (optional) |
#
# **Spatial layout:** Chamber1 — Chamber2 — ... — Chamber9 (linear chain)

# %% [markdown]
# ## Cell 1: Setup — Mount Drive & Install Dependencies

# %%
import os, sys, json, time, shutil, subprocess, struct, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import urllib.request
import zipfile

BASE = Path.cwd()
SCRIPTS = BASE / 'scripts'
sys.path.insert(0, str(SCRIPTS))

# Detect Colab runtime regardless of Drive mount status
ON_COLAB = False
try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    pass

# Mount Google Drive (Colab only)
if ON_COLAB:
    DRIVE_MOUNT = Path('/content/drive')
    print("Colab detected — mounting Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    WORKSPACE = DRIVE_MOUNT / 'MyDrive' / 'arena_3dgs'
else:
    DRIVE_MOUNT = None
    WORKSPACE = BASE / 'colmap_workspace'

WORKSPACE.mkdir(parents=True, exist_ok=True)
CHECKPOINT = WORKSPACE / 'stitch_session.json'

print(f"Workspace: {WORKSPACE}")
print(f"Python: {sys.version}")

REPO_URL = "https://github.com/kaarthik-balakrishnan/arena-3dgs"
RAW_URL = "https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main"

# %% [markdown]
# ## Cell 1b: Load the chamber_splat module

# %%
# Download the module from GitHub if missing (works on Colab or locally)
if not (SCRIPTS / 'chamber_splat.py').exists():
    print("Downloading chamber_splat.py from GitHub...")
    SCRIPTS.mkdir(exist_ok=True)
    url = ("https://raw.githubusercontent.com/"
           "kaarthik-balakrishnan/arena-3dgs/main/scripts/chamber_splat.py")
    try:
        urllib.request.urlretrieve(url, SCRIPTS / 'chamber_splat.py')
        print("  Done.")
    except Exception as e:
        print(f"  Download failed: {e}")

if not (SCRIPTS / 'chamber_splat.py').exists():
    print("ERROR: scripts/chamber_splat.py not found!")
    print("Open the notebook from GitHub to auto-download:")
    print("  https://github.com/kaarthik-balakrishnan/arena-3dgs")
    print("Or manually place scripts/chamber_splat.py from the repo.")
    raise SystemExit(1)

from scripts.chamber_splat import (
    parse_image_name, categorize_images, print_dataset_summary,
    filter_bad_images,
    run_chamber_colmap, load_colmap_model,
    get_bridge_point_ids, get_bridge_descriptors, get_bridge_positions,
    match_bridge_points, compute_rigid_transform_ransac,
    compute_chamber_transform, format_transform,
    apply_transform_to_model, merge_models_into_unified,
    export_colmap_model_text, export_colmap_model_binary,
    setup_3dgs_input, train_unified_3dgs,
    save_checkpoint, load_checkpoint,
)

# %% [markdown]
# ## Cell 1c: Install COLMAP (Colab)

# %%
# Check if COLMAP is available
colmap_available = False
try:
    subprocess.run(['colmap', 'help'], capture_output=True, check=True)
    colmap_available = True
    print("COLMAP already installed")
except (FileNotFoundError, subprocess.CalledProcessError):
    pass

if not colmap_available and ON_COLAB:
    print("Installing COLMAP via apt-get...")
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'colmap'], check=True)
    # Verify
    result = subprocess.run(['colmap', 'help'], capture_output=True, text=True)
    print(result.stdout[:200])
    colmap_available = True

if not colmap_available:
    print("WARNING: COLMAP not available. Install locally: brew install colmap")
    print("Proceeding with existing COLMAP data if available.")

# %% [markdown]
# ## Cell 1d: Install PyTorch + 3DGS (for GPU training)

# %%
HAS_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        HAS_GPU = True
    elif torch.backends.mps.is_available():
        print("MPS available (Apple Silicon)")
        HAS_GPU = True
    else:
        print("PyTorch available but no GPU detected")
except ImportError:
    print("PyTorch not installed. Training will require Colab T4 GPU.")
    print("For Colab: Runtime → Change runtime type → T4 GPU")

# %% [markdown]
# ## Cell 2: Load Session State (Checkpoint Resume)

# %%
state = load_checkpoint(CHECKPOINT)
print("Session state:")
for k, v in state.items():
    if isinstance(v, dict):
        print(f"  {k}: {len(v)} entries")
    elif isinstance(v, list):
        print(f"  {k}: {len(v)} items")
    else:
        print(f"  {k}: {v}")

# Decision: fresh start or continue
if state.get('completed_steps'):
    print(f"\nCompleted steps: {state.get('completed_steps')}")
    continue_session = input("Continue from where you left off? (Y/n): ").strip().lower()
    if continue_session == 'n':
        state = {}
        print("Starting fresh.")
    else:
        print("Resuming session.")
else:
    print("No prior session — starting fresh.")

# %% [markdown]
# ## Cell 3: Data Categorization & Filtering
# Uses the naming convention to classify every image by chamber and type.

# %%
IMAGE_DIR = BASE / 'splat-files-processed'

# On Colab, download images from GitHub if not present
if len(list(IMAGE_DIR.glob('*.jpg'))) < 50:
    print("Downloading arena images from GitHub...")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    # Fetch actual file list from GitHub API
    api_url = ("https://api.github.com/repos/"
               "kaarthik-balakrishnan/arena-3dgs/contents/splat-files-processed")
    req = urllib.request.Request(api_url, headers={"User-Agent": "arena-3dgs"})
    try:
        with urllib.request.urlopen(req) as resp:
            entries = json.loads(resp.read().decode())
        filenames = [e['name'] for e in entries if e['name'].endswith('.jpg')]
    except Exception as e:
        print(f"  GitHub API error: {e}")
        print("  Falling back to known file list...")
        filenames = [
            "Chamber1_01.jpg", "Chamber1_02.jpg", "Chamber1_03_Chamber2.jpg",
            "Chamber1_04.jpg", "Chamber1_05.jpg", "Chamber1_06.jpg",
            "Chamber1_07_Chamber2.jpg", "Chamber1_08.jpg",
            "Chamber1_09_Chamber2.jpg", "Chamber1_10.jpg",
            "Chamber1_11_Chamber2.jpg", "Chamber1_12_Chamber2.jpg",
            "Chamber1_13.jpg", "Chamber1_14.jpg", "Chamber1_15.jpg",
            "Chamber1_16.jpg", "Chamber1_17_topview.jpg",
            "Chamber1_18_Chamber2_topview.jpg", "Chamber1_19_Chamber2.jpg",
            "Chamber1_20_outside.jpg", "Chamber1_21_outside.jpg",
            "Chamber1_22_Chamber2.jpg", "Chamber1_23_Chamber2.jpg",
            "Chamber1_24.jpg", "Chamber1_25.jpg", "Chamber1_26.jpg",
            "Chamber2_01_Chamber1.jpg", "Chamber2_02_Chamber1_topview.jpg",
            "Chamber2_03_Chamber1_topview.jpg", "Chamber2_04_Chamber3.jpg",
            "Chamber2_05_Chamber3.jpg", "Chamber2_06_Chamber1.jpg",
            "Chamber2_07_Chamber1_topview.jpg", "Chamber2_08_Chamber3_topview.jpg",
            "Chamber2_09_Chamber3_topview.jpg", "Chamber2_10_outside.jpg",
            "Chamber3_01_Chamber4.jpg", "Chamber3_02_Chamber4_topview.jpg",
            "Chamber3_03_Chamber2.jpg", "Chamber3_04_Chamber4_topview.jpg",
            "Chamber3_05_outside.jpg", "Chamber4_01_Chamber3_topview.jpg",
            "Chamber4_02_Chamber3.jpg", "Chamber4_03_topview.jpg",
            "Chamber4_04_Chamber5_topview.jpg", "Chamber4_05_Chamber5_topview.jpg",
            "Chamber4_06_Chamber5.jpg", "Chamber4_07_Chamber3.jpg",
            "Chamber4_08_Chamber5_topview.jpg", "Chamber4_09_outside.jpg",
            "Chamber5_01_Chamber4_topview.jpg", "Chamber5_02_Chamber6_topview.jpg",
            "Chamber5_03_Chamber6_topview.jpg", "Chamber5_04_Chamber4.jpg",
            "Chamber5_05_Chamber6.jpg", "Chamber5_06_Chamber6_topview.jpg",
            "Chamber5_07_outside.jpg", "Chamber6_01_Chamber5_topview.jpg",
            "Chamber6_02_Chamber5_topview.jpg", "Chamber6_03_Chamber5_topview.jpg",
            "Chamber6_04_Chamber7_topview.jpg", "Chamber6_05_Chamber7_topview.jpg",
            "Chamber6_06_Chamber7.jpg", "Chamber6_07_Chamber7.jpg",
            "Chamber6_08_Chamber5.jpg", "Chamber6_09_Chamber7_topview.jpg",
            "Chamber6_10_outside.jpg", "Chamber7_01_Chamber6_topview.jpg",
            "Chamber7_02_Chamber6_topview.jpg", "Chamber7_03_Chamber8.jpg",
            "Chamber7_04_Chamber8_topview.jpg", "Chamber7_05_Chamber8.jpg",
            "Chamber7_06_Chamber6.jpg", "Chamber7_07_Chamber6_topview.jpg",
            "Chamber7_08_outside.jpg", "Chamber8_01_Chamber7_topview.jpg",
            "Chamber8_02_topview.jpg", "Chamber8_03_Chamber9_topview.jpg",
            "Chamber8_04_Chamber7.jpg", "Chamber8_05_Chamber7.jpg",
            "Chamber8_06_Chamber7.jpg", "Chamber8_07_Chamber9_topview.jpg",
            "Chamber8_08_Chamber9.jpg", "Chamber8_09_outside.jpg",
            "Chamber9_01_Chamber8_topview.jpg", "Chamber9_02_Chamber8_topview.jpg",
            "Chamber9_03.jpg", "Chamber9_04.jpg", "Chamber9_05.jpg",
            "Chamber9_06.jpg", "Chamber9_07.jpg",
        ]
    for fname in filenames:
        if (IMAGE_DIR / fname).exists():
            continue
        try:
            urllib.request.urlretrieve(
                f"{RAW_URL}/splat-files-processed/{fname}",
                IMAGE_DIR / fname
            )
        except Exception as e:
            print(f"  Failed to download {fname}: {e}")
    n_imgs = len(list(IMAGE_DIR.glob('*.jpg')))
    print(f"  Downloaded {n_imgs}/91 images")

# Pre-compute quality scores (lightweight)
print("Computing image quality metrics...")
from PIL import Image
quality_scores = {}
for p in sorted(IMAGE_DIR.glob('*.jpg')):
    img = np.array(Image.open(p).convert('L'), dtype=np.float32)
    gx = np.abs(np.diff(img, axis=1)).mean()
    gy = np.abs(np.diff(img, axis=0)).mean()
    sharpness = float((gx + gy) / 2)
    brightness = float(img.mean())
    quality_scores[p.name] = {'sharpness': sharpness, 'brightness': brightness}

# Categorize
by_chamber_raw, transitions, all_parsed = categorize_images(IMAGE_DIR)
print_dataset_summary(by_chamber_raw, transitions)

# Filter bad images
by_chamber, removed = filter_bad_images(
    by_chamber_raw, quality_scores,
    brightness_range=(15, 240), min_sharpness=0.3,
)
print(f"\nFiltered: kept {sum(len(v) for v in by_chamber.values())} images, "
      f"removed {len(removed)}:")
for name, reason in removed[:10]:
    print(f"  ✗ {name} ({reason})")
if len(removed) > 10:
    print(f"  ... and {len(removed) - 10} more")

# Save session
state['by_chamber'] = {str(k): [i['filename'] for i in v]
                        for k, v in by_chamber.items()}
state['transitions'] = {f"{k[0]}_{k[1]}": [i['filename'] for i in v]
                         for k, v in transitions.items()}
state['quality_scores'] = quality_scores
save_checkpoint(CHECKPOINT, state)
print("\n✅ Cell 3 complete")

# %% [markdown]
# ## Cell 4: Per-Chamber COLMAP
# Each chamber gets its own independent SfM reconstruction.
# This avoids the lighting inconsistency problem — dark chambers don't
# pollute bright ones.

# %%
CHAMBER_COLMAP_DIR = WORKSPACE / 'chamber_colmap'
state.setdefault('chamber_models', {})
completed = state.get('completed_steps', [])

if 'per_chamber_colmap' in completed:
    print("Per-chamber COLMAP already done. Loading results...")
    chamber_models = {}
    for cid_str, model_info in state['chamber_models'].items():
        cid = int(cid_str)
        model_dir = Path(model_info['path'])
        db_path = model_dir.parent / 'database.db'
        if model_dir.exists():
            chamber_models[cid] = load_colmap_model(
                model_dir / 'txt' if (model_dir / 'txt').exists() else model_dir,
                db_path=db_path if db_path.exists() else None,
            )
            print(f"  Chamber {cid}: {model_info['n_reg']}/{model_info['n_total']} registered")
        else:
            print(f"  Chamber {cid}: model dir not found, will re-run")
else:
    chamber_models = {}
    for cid in sorted(by_chamber.keys()):
        imgs = by_chamber[cid]
        print(f"\n{'='*60}")
        print(f"  COLMAP for Chamber {cid} ({len(imgs)} images)")
        print(f"{'='*60}")

        model_dir = run_chamber_colmap(
            cid, imgs, CHAMBER_COLMAP_DIR, IMAGE_DIR,
            options={'max_features': 32768, 'peak_threshold': 0.015,
                     'ba_local_iters': 25, 'ba_global_iters': 50},
        )

        if model_dir is not None:
            db_path = model_dir.parent / 'database.db'
            model = load_colmap_model(
                model_dir / 'txt' if (model_dir / 'txt').exists() else model_dir,
                db_path=db_path if db_path.exists() else None,
            )
            chamber_models[cid] = model
            n_total = len(imgs)
            n_reg = len(model['images'])
            state['chamber_models'][str(cid)] = {
                'path': str(model_dir),
                'n_reg': n_reg,
                'n_total': n_total,
            }
            print(f"  ✅ Chamber {cid}: {n_reg}/{n_total} images registered")
        else:
            print(f"  ❌ Chamber {cid}: COLMAP failed")

    completed.append('per_chamber_colmap')
    state['completed_steps'] = list(set(completed))
    save_checkpoint(CHECKPOINT, state)

print(f"\n✅ Cell 4 complete: {len(chamber_models)} chambers processed")

# %% [markdown]
# ## Cell 5: Cross-Chamber Alignment
# Uses transition images (ChamberX→ChamberY) and binary COLMAP descriptors
# to find 3D-3D correspondences, then RANSAC for rigid transforms.
#
# **Algorithm:**
# 1. For adjacent chambers X and Y, identify bridge points (3D points observed
#    by transition images showing the other chamber)
# 2. Extract 128-byte SIFT descriptors from COLMAP binary models
# 3. Match descriptors between chambers using brute-force + Lowe's ratio test
# 4. RANSAC on matched 3D positions → rigid transform (R, t)
# 5. Chain transforms: all chambers → Chamber1's coordinate frame

# %%
CHAMBER_MODELS_DIR = CHAMBER_COLMAP_DIR

state.setdefault('transforms', {})
state.setdefault('chamber_models', {})

if 'cross_chamber_alignment' in completed:
    print("Cross-chamber alignment already done. Loading transforms...")
    transforms = {}
    for cid_str, t_data in state.get('transforms', {}).items():
        cid = int(cid_str)
        R = np.array(t_data['R'])
        t = np.array(t_data['t'])
        transforms[cid] = (R, t)
        print(f"  Chamber {cid} → Chamber1: {t_data.get('inliers', 0)} inliers")
else:
    # Build adjacency list from naming convention
    adjacency = []
    for pair_key in sorted(state.get('transitions', {}).keys()):
        parts = pair_key.split('_')
        a, b = int(parts[0]), int(parts[1])
        if abs(a - b) == 1:  # Ensure adjacent
            adjacency.append((a, b))

    print(f"Adjacent chamber pairs: {adjacency}")
    print()

    # Compute transforms between adjacent pairs
    raw_transforms = {}  # (from_cid, to_cid) -> (R, t)
    chamber_order = sorted(chamber_models.keys())
    print(f"Chambers with models: {chamber_order}")

    for a, b in adjacency:
        if a not in chamber_models or b not in chamber_models:
            print(f"  Skipping {a}↔{b}: not all models available")
            continue

        # Compute transform to align Chamber{B} into Chamber{A}'s frame
        print(f"\n{'─'*50}")
        print(f"  Aligning Chamber {b} → Chamber {a}")
        print(f"{'─'*50}")

        result = compute_chamber_transform(
            chamber_models[a], chamber_models[b], a, b
        )

        if result[0] is not None:
            (R, t), matched_pairs = result
            raw_transforms[(a, b)] = (R, t, matched_pairs)
            print(f"  ✅ Transform found: {len(matched_pairs)} inlier matches")
            print(f"  {format_transform(R, t)}")
        else:
            # Try the other direction
            print(f"  Trying reverse alignment: Chamber {a} → Chamber {b}")
            result_rev = compute_chamber_transform(
                chamber_models[b], chamber_models[a], b, a
            )
            if result_rev[0] is not None:
                R_rev, t_rev = result_rev[0]
                # Invert: we want b→a, but got a→b
                R = R_rev.T
                t = -R_rev.T @ t_rev
                raw_transforms[(a, b)] = (R, t, result_rev[1])
                print(f"  ✅ Transform found (inverted): {len(result_rev[1])} inliers")
            else:
                print(f"  ❌ No transform found between {a} and {b}")
                raw_transforms[(a, b)] = None

    # Chain transforms: all → Chamber1
    BASE_CHAMBER = 1
    transforms = {BASE_CHAMBER: (np.eye(3), np.zeros(3))}

    if BASE_CHAMBER in chamber_models:
        # Build directional chain
        for cid in sorted(chamber_models.keys()):
            if cid == BASE_CHAMBER:
                continue

            # Find path through adjacency chain
            T_cum = np.eye(3)
            t_cum = np.zeros(3)
            current = cid
            path = [current]

            while current != BASE_CHAMBER:
                # Find adjacent chamber closer to BASE
                neighbor = None
                for (a, b), result in raw_transforms.items():
                    if result is None:
                        continue
                    R, t, _ = result
                    if b == current and a < current:
                        neighbor = a
                        T_cum = R @ T_cum
                        t_cum = R @ t_cum + t
                        break
                    elif a == current and b < current:
                        # Need to invert
                        R_inv = R.T
                        t_inv = -R.T @ t
                        neighbor = b
                        T_cum = R_inv @ T_cum
                        t_cum = R_inv @ t_cum + t_inv
                        break

                if neighbor is None or neighbor in path:
                    print(f"  ⚠ Cannot chain Chamber {cid} → Chamber1 (break at {current})")
                    break
                current = neighbor
                path.append(current)

            if current == BASE_CHAMBER:
                transforms[cid] = (T_cum, t_cum)
                print(f"  Chamber {cid} → Chamber1: chained via {path}")
            else:
                print(f"  ❌ Chamber {cid}: no chain found")

    # Save transforms
    for cid, (R, t) in transforms.items():
        if cid != BASE_CHAMBER:
            state['transforms'][str(cid)] = {
                'R': R.tolist(),
                't': t.tolist(),
                'inliers': 0,
            }

    completed.append('cross_chamber_alignment')
    state['completed_steps'] = list(set(completed))
    save_checkpoint(CHECKPOINT, state)

print(f"\n✅ Cell 5 complete: transforms for {len(transforms)} chambers")

# %% [markdown]
# ## Cell 6: Unified Sparse Model
# Transform all chamber models into Chamber1's coordinate frame and merge.

# %%
if 'unified_model' in completed:
    print("Unified model already built. Loading...")
    unified_dir = WORKSPACE / 'unified_model' / 'txt'
    if unified_dir.exists():
        unified = load_colmap_model(unified_dir)
    else:
        print("Unified model not found, rebuilding...")
else:
    # Filter to chambers with valid transforms
    valid_chambers = {}
    for cid in sorted(chamber_models.keys()):
        if cid in transforms:
            valid_chambers[cid] = chamber_models[cid]
        else:
            print(f"  Skipping Chamber {cid}: no transform to base frame")

    print(f"\nBuilding unified model from {len(valid_chambers)} chambers...")

    unified = merge_models_into_unified(
        valid_chambers, transforms, base_chamber=1
    )

    # Export for 3DGS training
    unified_dir = WORKSPACE / 'unified_model'
    export_colmap_model_text(unified, unified_dir / 'txt')
    export_colmap_model_binary(unified, unified_dir / 'bin')

    completed.append('unified_model')
    state['completed_steps'] = list(set(completed))
    state['unified_stats'] = {
        'n_cameras': len(unified['cameras']),
        'n_images': len(unified['images']),
        'n_points': len(unified['points3d']),
    }
    save_checkpoint(CHECKPOINT, state)

print(f"\n✅ Cell 6 complete: unified model with "
      f"{state.get('unified_stats', {}).get('n_points', 0):,} points")

# %% [markdown]
# ## Cell 7: Setup 3DGS Training Input
# Prepares the `gaussian-splatting/input/` directory with images + COLMAP model.

# %%
if HAS_GPU or DRIVE_MOUNT.exists():
    print("Setting up 3DGS training input...")
    d = setup_3dgs_input(unified, IMAGE_DIR, WORKSPACE / 'unified_3dgs_input')
    print(f"Input ready at: {d}")
else:
    print("GPU not available. Training will need to be done on Colab T4.")
    print("Exporting unified model for transfer...")
    export_path = WORKSPACE / 'unified_model'
    print(f"  Unified COLMAP model: {export_path}")

    # Create a zip for easy transfer
    import zipfile
    zip_path = WORKSPACE / 'unified_model_for_colab.zip'
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for f in (export_path / 'txt').iterdir():
                zf.write(f, f'unified_model/sparse/0/{f.name}')
            for img_info in list(unified['images'].values())[:5]:
                img_path = IMAGE_DIR / img_info['name']
                if img_path.exists():
                    zf.write(img_path, f'unified_model/images/{img_info['name']}')
        print(f"  Zip: {zip_path}")

# %% [markdown]
# ## Cell 8A: Train Unified 3DGS (GPU required)
# Single end-to-end 3DGS training on the unified sparse model.
# This gives the best quality result.

# %%
if 'unified_3dgs_trained' in completed:
    print("Unified 3DGS already trained.")
    ply_files = list(WORKSPACE.glob('arena_unified_*.ply'))
    for pf in ply_files:
        print(f"  {pf} ({pf.stat().st_size / 1024**2:.1f} MB)")
else:
    if HAS_GPU:
        print("Starting unified 3DGS training...")

        input_dir = WORKSPACE / 'unified_3dgs_input'
        output_dir = WORKSPACE

        # Quick test (3K iters)
        print("\nQuick test: 3K iterations")
        from scripts.train_3dgs_enhanced import train
        import argparse
        args = argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            iterations=3000,
            max_gaussians=50000,
            log_interval=500,
        )
        train(args)

        # Rename quick output
        quick_ply = output_dir / 'arena_3dgs.ply'
        if quick_ply.exists():
            quick_ply.rename(output_dir / 'arena_unified_quick.ply')

        # Full training (30K iters)
        print("\nFull training: 30K iterations")
        args = argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            iterations=30000,
            max_gaussians=500000,
            log_interval=1000,
        )
        train(args)

        full_ply = output_dir / 'arena_3dgs.ply'
        if full_ply.exists():
            full_ply.rename(output_dir / 'arena_unified_30K.ply')

        completed.append('unified_3dgs_trained')
        state['completed_steps'] = list(set(completed))
        save_checkpoint(CHECKPOINT, state)
    else:
        print("No GPU available. Upload to Colab for training:")
        print(f"  1. Download unified model: {WORKSPACE}/unified_model_for_colab.zip")

# %% [markdown]
# ## Cell 8B: Per-Chamber 3DGS Training (Optional)
# Trains individual 3DGS models for each chamber.
# Useful for inspection or if you want the experimental stitch-then-fine-tune approach.

# %%
PER_CHAMBER_3DGS_DIR = WORKSPACE / 'per_chamber_3dgs'

if state.get('per_chamber_3dgs_done'):
    print("Per-chamber 3DGS already done.")
else:
    if HAS_GPU:
        from scripts.train_3dgs_enhanced import train
        import argparse

        per_chamber_models = {}
        for cid, model in sorted(chamber_models.items()):
            if len(model['images']) < 5:
                print(f"Chamber {cid}: too few images ({len(model['images'])}), skipping")
                continue

            chamber_out = PER_CHAMBER_3DGS_DIR / f'chamber_{cid}'
            chamber_input = setup_3dgs_input(
                model, IMAGE_DIR,
                PER_CHAMBER_3DGS_DIR / f'input_{cid}'
            )

            print(f"\nTraining Chamber {cid} ({len(model['images'])} views)...")
            args = argparse.Namespace(
                input_dir=str(chamber_input),
                output_dir=str(chamber_out),
                iterations=3000,
                max_gaussians=50000,
                log_interval=500,
            )
            train(args)

            ply_path = chamber_out / 'arena_3dgs.ply'
            if ply_path.exists():
                ply_path.rename(chamber_out / f'chamber_{cid}_3dgs.ply')
                per_chamber_models[cid] = chamber_out / f'chamber_{cid}_3dgs.ply'

        state['per_chamber_3dgs_done'] = True
        state['per_chamber_models'] = {
            str(k): str(v) for k, v in per_chamber_models.items()
        }
        save_checkpoint(CHECKPOINT, state)
    else:
        print("No GPU available. Skip per-chamber training or use Colab.")

# %% [markdown]
# ## Cell 9: Gaussian Stitching (Experimental)
# If you trained per-chamber 3DGS models, this cell stitches them into
# a unified gaussian cloud by applying the computed transforms.
#
# **Method:** Load each chamber's PLY, apply (R, t) to gaussian means and
# quaternions, concatenate all gaussians into one big cloud. Optionally
# prune gaussians in overlap regions.

# %%
if state.get('stitched_ply_done'):
    print("Stitched PLY already built.")
else:
    per_chamber_paths = state.get('per_chamber_models', {})
    if per_chamber_paths and HAS_GPU:
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            !pip install -q plyfile
            from plyfile import PlyData, PlyElement

        print("Loading per-chamber models and applying transforms...")
        stitched_arrays = []
        for cid_str, ply_path_str in sorted(per_chamber_paths.items()):
            cid = int(cid_str)
            ply_path = Path(ply_path_str)
            if not ply_path.exists():
                continue

            print(f"  Chamber {cid}: loading {ply_path.name}")

            R, t = transforms.get(cid, (np.eye(3), np.zeros(3)))
            ply = PlyData.read(str(ply_path))
            vertex = ply['vertex'].data

            n = len(vertex)
            dtype = vertex.dtype
            new_data = np.zeros(n, dtype=dtype)

            for name in dtype.names:
                new_data[name] = vertex[name]

            # Transform positions
            xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            xyz_transformed = (R @ xyz.T).T + t
            new_data['x'] = xyz_transformed[:, 0]
            new_data['y'] = xyz_transformed[:, 1]
            new_data['z'] = xyz_transformed[:, 2]

            # Transform rotations
            # quat is (w, x, y, z). We need to rotate the quat by R.
            # The gaussian rotation matrix is derived from the quat.
            # new_quat = quaternion_from_matrix(R @ matrix_from_quat(quat))
            from scipy.spatial.transform import Rotation as R_scipy
            quats = np.column_stack([vertex['rot_0'], vertex['rot_1'],
                                      vertex['rot_2'], vertex['rot_3']])
            rot_matrices = R_scipy.from_quat(quats).as_matrix()
            rot_transformed = R @ rot_matrices
            new_quats = R_scipy.from_matrix(rot_transformed).as_quat()
            new_data['rot_0'] = new_quats[:, 0]
            new_data['rot_1'] = new_quats[:, 1]
            new_data['rot_2'] = new_quats[:, 2]
            new_data['rot_3'] = new_quats[:, 3]

            stitched_arrays.append(new_data)

        if stitched_arrays:
            stitched = np.concatenate(stitched_arrays)
            print(f"\nStitched model: {len(stitched)} gaussians")

            # Prune distant / low-opacity gaussians
            if len(stitched) > 1000000:
                print("  Pruning low-opacity gaussians...")
                mask = stitched['opacity'] > 0.01
                stitched = stitched[mask]
                print(f"  After pruning: {len(stitched)} gaussians")

            output_path = WORKSPACE / 'arena_stitched.ply'
            PlyData([PlyElement.describe(stitched, 'vertex')]).write(str(output_path))
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Stitched PLY: {output_path} ({size_mb:.1f} MB, "
                  f"{len(stitched):,} gaussians)")

            state['stitched_ply_done'] = True
            state['stitched_ply_path'] = str(output_path)
            state['stitched_gaussian_count'] = len(stitched)
            save_checkpoint(CHECKPOINT, state)
    else:
        print("No per-chamber models found. Run Cell 8B first, or use unified training.")

# %% [markdown]
# ## Cell 10: Compress & Export

# %%
# Find the best PLY to compress
ply_candidates = list(WORKSPACE.glob('arena_*.ply'))
ply_candidates += list(BASE.glob('output/arena_*.ply'))
ply_candidates = [p for p in ply_candidates if 'compressed' not in p.stem]

if ply_candidates:
    best_ply = max(ply_candidates, key=lambda p: p.stat().st_size)
    print(f"Best PLY: {best_ply.name} ({best_ply.stat().st_size / 1024**2:.1f} MB)")

    print("\nCompressing (medium quality)...")
    !python3 "{SCRIPTS}/compress_splat.py" "{best_ply}" --quality medium -o "{WORKSPACE}"

    compressed = WORKSPACE / f"{best_ply.stem}_compressed.splat"
    if compressed.exists():
        print(f"\n✅ Compressed: {compressed}")

    if DRIVE_MOUNT.exists():
        print("\nDownload to local machine:")
        print(f"  from google.colab import files")
        print(f"  files.download('{compressed}')")
else:
    print("No PLY files found. Train first (Cell 8A or 8B).")

# %% [markdown]
# ## Summary: Algorithm Overview
#
# ```
#                  ┌──────────────────┐
#                  │  91 arena photos │
#                  │  (naming parsed) │
#                  └────────┬─────────┘
#                           │
#              ┌────────────┼────────────┐
#              ▼            ▼            ▼
#        Chamber 1     Chamber 2 ... Chamber 9
#              │            │            │
#         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
#         │ COLMAP  │  │ COLMAP  │  │ COLMAP  │
#         │ (SfM)   │  │ (SfM)   │  │ (SfM)   │
#         └────┬────┘  └────┬────┘  └────┬────┘
#              │            │            │
#         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
#         │ Sparse  │  │ Sparse  │  │ Sparse  │
#         │ PtCloud │  │ PtCloud │  │ PtCloud │
#         └────┬────┘  └────┬────┘  └────┬────┘
#              │            │            │
#              └──────┬─────┴──────┬─────┘
#                     │           │
#               ┌─────▼─────┐ ┌───▼────┐
#               │ Descriptor│ │RANSAC  │
#               │ Matching  │ │Rigid   │
#               │ (SIFT)    │ │Transform│
#               └─────┬─────┘ └───┬────┘
#                     │           │
#               ┌─────▼───────────▼────┐
#               │  Unified Sparse      │
#               │  Point Cloud         │
#               │  (all in Chamber1    │
#               │   coordinate frame)  │
#               └─────┬───────────┬────┘
#                     │           │
#          ┌──────────▼┐    ┌─────▼──────┐
#          │  3DGS     │    │ Per-Chamber│
#          │  Unified  │    │ 3DGS +     │
#          │  Training │    │ Stitching  │
#          └───────────┘    └────────────┘
# ```
#
# **Key novel contribution:** Using the naming convention as a structural prior
# to decompose a large, lighting-inconsistent scene into smaller consistent
# sub-models, then recomposing via COLMAP descriptor matching.
#
# This avoids the common failure mode where COLMAP on the full 91-image set
# only registers ~37% of images (due to the ×6 EV brightness range across
# chambers and 34 top-views with different perspective).

# %% [markdown]
# ## Appendix: Local Viewing

# %%
print("To view the result locally (CPU only, no GPU needed):")
print()
print("  # Decompress and view")
print(f"  python3 scripts/decompress_splat.py {WORKSPACE}/arena_unified_30K_compressed.splat")
print()
print("  # Or view the stitched per-chamber PLY")
print(f"  python3 scripts/decompress_splat.py {WORKSPACE}/arena_stitched.ply --export")
print()
print("  # View in SuperSplat (no install)")
print("  # → https://supersplat.com/")
