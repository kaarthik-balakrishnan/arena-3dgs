#!/usr/bin/env python3
"""
COLMAP exhaustive matching + reconstruction on all 91 images.
Finds ALL cross-chamber image pairs, not just sequential neighbors.

Usage:
  python3 colmap_exhaustive.py
"""

import subprocess, sys, os, shutil, struct, json, sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent
IMAGES = BASE / "splat-files-processed"
WORKSPACE = BASE / "colmap_workspace"
DATABASE = WORKSPACE / "database_exhaustive.db"
SPARSE_OUT = WORKSPACE / "sparse_exhaustive"
REPORT = WORKSPACE / "colmap_exhaustive_report.json"

CAMERA_MODEL = "SIMPLE_RADIAL"
SINGLE_CAMERA = 1
MAX_FEATURES = 8192
PEAK_THRESHOLD = 0.004

MAPPER_PARAMS = {
    "multiple_models": 1,
    "max_num_models": 50,
    "min_model_size": 5,
    "init_min_tri_angle": 4,
    "init_min_num_inliers": 15,
    "abs_pose_min_num_inliers": 8,
    "abs_pose_min_inlier_ratio": 0.25,
    "ba_local_max_num_iterations": 25,
    "ba_global_max_num_iterations": 50,
    "ba_global_frames_freq": 500,
    "ba_global_points_freq": 250000,
}


def run(cmd, desc, fatal=True):
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"{'─'*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED (code {result.returncode})")
        if fatal:
            sys.exit(1)
    return result.returncode


# ── Step 1: Feature extraction ──────────────────────────────────
if DATABASE.exists():
    DATABASE.unlink()
if SPARSE_OUT.exists():
    shutil.rmtree(SPARSE_OUT)
WORKSPACE.mkdir(parents=True, exist_ok=True)
SPARSE_OUT.mkdir(parents=True, exist_ok=True)

run(
    f"colmap feature_extractor "
    f"--database_path {DATABASE} "
    f"--image_path {IMAGES} "
    f"--ImageReader.camera_model {CAMERA_MODEL} "
    f"--ImageReader.single_camera {SINGLE_CAMERA} "
    f"--SiftExtraction.max_num_features {MAX_FEATURES} "
    f"--SiftExtraction.peak_threshold {PEAK_THRESHOLD} "
    f"--SiftExtraction.edge_threshold 10",
    "Feature extraction (all 91 images)"
)

conn = sqlite3.connect(str(DATABASE))
num_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
total_feats = conn.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0]
print(f"  {num_images} images, {total_feats} total features")
conn.close()

# ── Step 2: Exhaustive matching ─────────────────────────────────
run(
    f"colmap exhaustive_matcher "
    f"--database_path {DATABASE} "
    f"--FeatureMatching.max_num_matches 32768 ",
    "Exhaustive matching (all pairwise)"
)

conn = sqlite3.connect(str(DATABASE))
verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
print(f"\n  Verified match pairs: {verified}")
conn.close()

# ── Step 3: Reconstruction ─────────────────────────────────────
param_str = " ".join(f"--Mapper.{k} {v}" for k, v in MAPPER_PARAMS.items())
run(
    f"colmap mapper "
    f"--database_path {DATABASE} "
    f"--image_path {IMAGES} "
    f"--output_path {SPARSE_OUT} "
    f"{param_str} ",
    "Reconstruction (exhaustive matching)"
)

# ── Step 4: Analyze results ─────────────────────────────────────
def get_registered_names(model_path):
    txt_dir = Path(str(model_path) + "_txt")
    txt_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap model_converter --input_path {model_path} --output_path {txt_dir} --output_type TXT",
        f"Convert {Path(model_path).name} to text",
        fatal=False
    )
    names = set()
    imgtxt = txt_dir / "images.txt"
    if imgtxt.exists():
        with open(imgtxt) as f:
            lines = [l.rstrip() for l in f if l.strip() and not l.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) >= 10:
                names.add(parts[9])
    return names


models = []
for sub in sorted(SPARSE_OUT.iterdir()):
    img_bin = sub / "images.bin"
    if img_bin.exists():
        with open(img_bin, "rb") as f:
            n_imgs = struct.unpack("Q", f.read(8))[0]
        names = get_registered_names(sub)
        models.append({"name": sub.name, "images": n_imgs, "points": "...", "names": names})
models.sort(key=lambda m: -m["images"])

print(f"\n\n{'='*60}")
print(f"  RESULTS: {len(models)} model(s)")
print(f"{'='*60}")

all_fnames = {f.name for f in IMAGES.glob("*.jpg")}
all_registered = set()

for m in models:
    all_registered |= m["names"]
    print(f"\n  Model {m['name']}: {m['images']} images")
    for name in sorted(m["names"]):
        print(f"    {name}")

print(f"\n  Total unique registered: {len(all_registered)}/{len(all_fnames)}")
unregistered = all_fnames - all_registered
print(f"  Unregistered: {len(unregistered)}")

# By chamber
from collections import defaultdict
unreg_by_chamber = defaultdict(list)
for name in sorted(unregistered):
    chamber = None
    for c in range(1, 10):
        if name.startswith(f"Chamber{c}_"):
            chamber = f"Chamber{c}"
            break
    if chamber:
        unreg_by_chamber[chamber].append(name)

print(f"\n  By chamber:")
for c in sorted(unreg_by_chamber):
    primary_of_c = [f for f in all_fnames if f.startswith(c + "_")]
    reg_c = len(primary_of_c) - len([u for u in unreg_by_chamber[c] if u in primary_of_c])
    # Actually simpler:
    reg_c = sum(1 for f in primary_of_c if f in all_registered)
    bar = "█" * reg_c + "░" * (len(primary_of_c) - reg_c)
    print(f"    {c}: {bar}  {reg_c}/{len(primary_of_c)}")

# Save report
report = {
    "total": len(all_fnames),
    "registered": len(all_registered),
    "unregistered": len(unregistered),
    "models": len(models),
    "match_pairs": verified,
}
with open(REPORT, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Report: {REPORT}")

if unregistered:
    print(f"\n  Unregistered images:")
    for name in sorted(unregistered):
        print(f"    {name}")

print(f"\n  Done!")
