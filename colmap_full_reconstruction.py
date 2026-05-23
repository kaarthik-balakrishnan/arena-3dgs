#!/usr/bin/env python3
"""
COLMAP Full Reconstruction — 91 images across 9 chambers.

Strategy: sequential matching with chamber-aware ordering.
No exhaustive matching — only adjacent images in the ordered walkthrough
are matched, which covers intra-chamber + inter-chamber overlaps.

Usage:
  python3 colmap_full_reconstruction.py --fresh
"""

import subprocess, sys, os, shutil, struct, json, sqlite3
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE / "colmap_workspace"
IMAGES = BASE / "splat-files-processed"
DATABASE = WORKSPACE / "database_full.db"
SPARSE_OUT = WORKSPACE / "sparse_full"
REPORT = WORKSPACE / "colmap_report.json"

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
        print(f"  ⚠ FAILED (code {result.returncode})")
        if fatal:
            sys.exit(1)
    return result.returncode


# ── Image ordering —─────────────────────────────────────────────
def get_chambers():
    """Return sorted list of chamber names."""
    chambers = set()
    for f in IMAGES.glob("*.jpg"):
        chamber = f.stem.split("_")[0]
        if chamber.startswith("Chamber") and chamber[7:].isdigit():
            chambers.add(chamber)
    return sorted(chambers)


def build_ordered_image_list():
    """
    Order images for sequential matching:
    Chamber1 images, then Chamber1↔Chamber2 overlap, then Chamber2 images, etc.
    This ensures sequential matching with overlap=N covers both intra-chamber
    and inter-chamber connections.
    """
    chambers = get_chambers()
    by_chamber = defaultdict(list)
    for f in sorted(IMAGES.glob("*.jpg")):
        chamber = f.stem.split("_")[0]
        if chamber in chambers:
            by_chamber[chamber].append(f.name)

    ordered = []
    for i, c in enumerate(chambers):
        ordered.extend(sorted(by_chamber[c]))
        # Also include first images of next chamber for overlap
        if i + 1 < len(chambers):
            next_c = chambers[i + 1]
            overlap_images = [f for f in by_chamber[next_c]
                             if f.split("_")[0] == next_c]
            # Only add a few transition images to maintain adjacency
            for of in sorted(overlap_images)[:3]:
                if of not in ordered:
                    ordered.append(of)

    # Deduplicate while preserving order
    seen = set()
    ordered_dedup = []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            ordered_dedup.append(name)
    return ordered_dedup


# ── COLMAP pipeline —────────────────────────────────────────────
def step_extract_features():
    if DATABASE.exists():
        DATABASE.unlink()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap feature_extractor "
        f"--database_path {DATABASE} "
        f"--image_path {IMAGES} "
        f"--ImageReader.camera_model {CAMERA_MODEL} "
        f"--ImageReader.single_camera {SINGLE_CAMERA} "
        f"--SiftExtraction.max_num_features {MAX_FEATURES} "
        f"--SiftExtraction.peak_threshold {PEAK_THRESHOLD} "
        f"--SiftExtraction.edge_threshold 10",
        "Feature extraction"
    )

    # Quick sanity check
    conn = sqlite3.connect(DATABASE)
    num_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()
    print(f"  {num_images} images in database")
    if num_images < 91:
        print(f"  ⚠ Expected 91 images, got {num_images}. Something is wrong.")
        sys.exit(1)


def step_sequential_matching():
    ordered = build_ordered_image_list()
    print(f"\n  Image order ({len(ordered)} entries):")
    for name in ordered:
        print(f"    {name}")

    run(
        f"colmap sequential_matcher "
        f"--database_path {DATABASE} "
        f"--FeatureMatching.max_num_matches 32768 "
        f"--SequentialMatching.overlap 15 "
        f"--SequentialMatching.loop_detection 1 "
        f"--SequentialMatching.loop_detection_period 20 ",
        "Sequential matching (alphabetical order = chamber traversal, overlap=15)"
    )

    conn = sqlite3.connect(DATABASE)
    verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    conn.close()
    print(f"\n  Verified image pairs: {verified}")
    if verified == 0:
        print("  ⚠ No verified matches — reconstruction will fail.")
        sys.exit(1)


def step_reconstruction(name, extra_params=None):
    out_dir = SPARSE_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(MAPPER_PARAMS)
    if extra_params:
        params.update(extra_params)
    param_str = " ".join(f"--Mapper.{k} {v}" for k, v in params.items())
    run(
        f"colmap mapper "
        f"--database_path {DATABASE} "
        f"--image_path {IMAGES} "
        f"--output_path {out_dir} "
        f"{param_str} ",
        f"Reconstruction '{name}'"
    )
    return list_models(out_dir)


def list_models(sparse_dir):
    models = []
    for sub in sorted(sparse_dir.iterdir()):
        img_bin = sub / "images.bin"
        if img_bin.exists():
            with open(img_bin, "rb") as f:
                n_imgs = struct.unpack("Q", f.read(8))[0]
            pts_bin = sub / "points3D.bin"
            n_pts = struct.unpack("Q", pts_bin.read_bytes()[:8])[0] if pts_bin.exists() else 0
            models.append({"name": sub.name, "images": n_imgs, "points": n_pts, "path": str(sub)})
    models.sort(key=lambda m: -m["images"])
    return models


def get_registered_names(model_path):
    """Read list of registered image filenames from a COLMAP model."""
    # Convert .bin to .txt
    txt_dir = Path(model_path) / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap model_converter "
        f"--input_path {model_path} "
        f"--output_path {txt_dir} "
        f"--output_type TXT",
        f"Convert {Path(model_path).name} to text",
        fatal=False
    )
    names = set()
    imgtxt = txt_dir / "images.txt"
    if imgtxt.exists():
        with open(imgtxt) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) >= 10:
                names.add(parts[9])
    return names


# ── Analysis —────────────────────────────────────────────────────
def analyze_results():
    print("\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    all_fnames = {f.name for f in IMAGES.glob("*.jpg")}
    print(f"  Total images: {len(all_fnames)}")

    models = []
    for sub in SPARSE_OUT.rglob("images.bin"):
        with open(sub, "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        models.append((sub.parent, n))
    models.sort(key=lambda x: -x[1])

    if not models:
        print("  No models produced.")
        return set(), all_fnames

    print(f"\n  Models: {len(models)}")
    registered_all = set()
    for model_dir, n in models:
        names = get_registered_names(model_dir)
        registered_all |= names
        print(f"    {model_dir.name}: {n} images, {len(names)} unique cumulative")

    unregistered = all_fnames - registered_all
    print(f"\n  Registered: {len(registered_all)}/{len(all_fnames)}")
    print(f"  Unregistered: {len(unregistered)}")

    if unregistered:
        print("\n  By chamber:")
        unreg_by_chamber = defaultdict(list)
        for name in sorted(unregistered):
            unreg_by_chamber[name.split("_")[0]].append(name)
        for c in sorted(unreg_by_chamber):
            print(f"    {c}: {len(unreg_by_chamber[c])}")

    # Database stats
    try:
        conn = sqlite3.connect(DATABASE)
        total_feats = conn.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0]
        print(f"\n  Total features across DB: {total_feats}")
        print(f"  Avg per image: {total_feats // max(len(all_fnames), 1)}")
        verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        print(f"  Verified matches: {verified}")
        conn.close()
    except Exception:
        pass

    report = {
        "total": len(all_fnames),
        "registered": len(registered_all),
        "unregistered": len(unregistered),
        "models": [{"name": m[0].name, "images": m[1]} for m in models],
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {REPORT}")

    return registered_all, unregistered


def print_summary(registered, unregistered):
    total = len(registered) + len(unregistered)
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(registered)}/{total} registered ({100*len(registered)//total}%)")
    print(f"{'='*60}")

    chambers = get_chambers()
    for c in chambers:
        imgs = [f.name for f in IMAGES.glob("*.jpg") if f.stem.split("_")[0] == c]
        reg = sum(1 for i in imgs if i in registered)
        bar = "█" * reg + "░" * (len(imgs) - reg)
        print(f"  {c:12s} {bar}  {reg}/{len(imgs)}")

    if unregistered:
        print(f"""
  Likely causes:
  - Chambers are separate rooms with limited visual overlap
  - Top-view/outside images have extreme perspective change
  - Sequential matching can't bridge all gaps (try vocab tree matcher
    for non-sequential connections)

  Next steps:
  1. Use these registered images as-is for 3DGS training
  2. Try Global SfM (colmap global_mapper) as an alternative
  3. Split into per-chamber reconstructions
""")


# ── Main —────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.fresh:
        for p in [SPARSE_OUT, WORKSPACE / "sparse_extended", WORKSPACE / "optimized_model"]:
            if p.exists():
                shutil.rmtree(p)
        if DATABASE.exists():
            DATABASE.unlink()

    WORKSPACE.mkdir(parents=True, exist_ok=True)

    step_extract_features()
    step_sequential_matching()

    models = step_reconstruction("round1")
    if models:
        print(f"\n  Best: {models[0]['images']} images, {models[0]['points']} points")

    # Try image registrator on best model
    best_models = sorted(
        [m for m in SPARSE_OUT.rglob("images.bin")],
        key=lambda p: struct.unpack("Q", p.read_bytes()[:8])[0]
    )
    if best_models:
        best = best_models[-1].parent
        extended = WORKSPACE / "sparse_extended"
        extended.mkdir(parents=True, exist_ok=True)
        run(
            f"colmap image_registrator "
            f"--database_path {DATABASE} "
            f"--input_path {best} "
            f"--output_path {extended} "
            f"--Mapper.init_min_num_inliers 15 "
            f"--Mapper.init_min_tri_angle 4 "
            f"--Mapper.abs_pose_min_num_inliers 6 "
            f"--Mapper.abs_pose_min_inlier_ratio 0.15 ",
            "Register remaining images into best model",
            fatal=False
        )

    registered, unregistered = analyze_results()
    print_summary(registered, unregistered)

    return 0


if __name__ == "__main__":
    main()
