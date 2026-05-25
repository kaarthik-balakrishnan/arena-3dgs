#!/usr/bin/env python3
"""
Run COLMAP on a set of images and produce 3DGS-ready input.

Produces:
  {name}_3dgs_input/          ← zip this and upload to Colab/Drive
    images/                   all input images
    sparse/0/                 COLMAP model (text format)
      cameras.txt
      images.txt
      points3D.txt

  {name}_colmap_workspace/    ← intermediate files (debugging only)
    database.db               features + matches
    sparse_global/            global SfM output
    optimized_model/          best reconstruction (SIMPLE_RADIAL)
    sparse.ply                sparse point cloud visualization

What 3DGS actually needs:
  - images/*.jpg              input photos
  - sparse/0/cameras.txt      camera intrinsics (SIMPLE_RADIAL: f, cx, cy, k)
  - sparse/0/images.txt       image poses & camera bindings
  - sparse/0/points3D.txt     sparse 3D points

Usage:
  python3 scripts/run_colmap.py -i sharp-frames
  python3 scripts/run_colmap.py -i manual-selection --skip-matching
  python3 scripts/run_colmap.py -i sharp-frames -o /tmp/my_output
"""
import subprocess, sys, os, shutil, argparse
from pathlib import Path

CAMERA_MODEL = "SIMPLE_RADIAL"
MAX_FEATURES = 8192

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Command failed with code {result.returncode}")
    return result.returncode

def colmap_binary():
    for p in ["colmap", "/usr/local/bin/colmap", "/opt/homebrew/bin/colmap"]:
        if shutil.which(p):
            return p
    return "colmap"

CM = colmap_binary()

def extract_features(images, database):
    run(
        f"{CM} feature_extractor "
        f"--database_path {database} "
        f"--image_path {images} "
        f"--ImageReader.camera_model {CAMERA_MODEL} "
        f"--ImageReader.single_camera 1 "
        f"--SiftExtraction.max_num_features {MAX_FEATURES} "
        f"--SiftExtraction.first_octave -1 "
        f"--SiftExtraction.peak_threshold 0.01",
        "Extracting SIFT features"
    )

def match_exhaustive(database):
    run(
        f"{CM} exhaustive_matcher "
        f"--database_path {database} ",
        "Exhaustive matching (all pairs)"
    )

def run_global_mapper(database, images, sparse_out):
    sparse_out.mkdir(parents=True, exist_ok=True)
    run(
        f"{CM} global_mapper "
        f"--database_path {database} "
        f"--image_path {images} "
        f"--output_path {sparse_out}",
        "Running global SfM"
    )

def convert_to_text(model_dir, txt_dir):
    """Convert binary model to text format."""
    txt_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"{CM} model_converter "
        f"--input_path {model_dir} "
        f"--output_path {txt_dir} "
        f"--output_type TXT",
        "Converting model to text format"
    )

def export_ply(model_dir, ply_path):
    run(
        f"{CM} model_converter "
        f"--input_path {model_dir} "
        f"--output_path {ply_path} "
        f"--output_type PLY",
        "Exporting sparse point cloud PLY"
    )

def copy_images(images, dst):
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in sorted(images.glob(ext)):
            target = dst / f.name
            if not target.exists():
                shutil.copy2(str(f), str(target))
            count += 1
    return count

def find_best_model(sparse_out):
    """Find best reconstruction by image count."""
    best_count = 0
    best_path = None
    for sub in sorted(sparse_out.iterdir()):
        imgtxt = sub / "images.txt"
        if not imgtxt.exists():
            imgtxt = sub / "images.bin"
        if not imgtxt.exists():
            continue
        if imgtxt.suffix == ".txt":
            with open(imgtxt) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            n = len(lines) // 2
        else:
            import struct
            with open(imgtxt, "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
        if n > best_count:
            best_count = n
            best_path = sub
    return best_count, best_path

def build_3dgs_input(gs_dir, model_dir):
    """Copy text-format COLMAP model into gs_dir for 3DGS training.
    
    Keeps SIMPLE_RADIAL camera model (train_3dgs_enhanced.py expects
    params ordering [f, cx, cy, k]).
    """
    gs_dir.mkdir(parents=True, exist_ok=True)

    # Convert to text if needed
    has_txt = all((model_dir / f).exists() for f in ["cameras.txt", "images.txt", "points3D.txt"])
    if not has_txt:
        txt_dir = model_dir / "txt"
        convert_to_text(model_dir, txt_dir)
        src = txt_dir
    else:
        src = model_dir

    # Copy text files
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        shutil.copy2(src / fname, gs_dir / fname)

    with open(gs_dir / "images.txt") as f:
        n_imgs = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
    print(f"\n  3DGS input ready at {gs_dir} ({n_imgs} images, SIMPLE_RADIAL)")

def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP and produce 3DGS-ready input folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python3 scripts/run_colmap.py -i sharp-frames\n  python3 scripts/run_colmap.py -i manual-selection\n  python3 scripts/run_colmap.py -i sharp-frames -o /tmp/my_3dgs_input",
    )
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Directory containing input images")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip feature extraction and matching if database exists")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory for 3DGS input (default: {input_parent}/{name}_3dgs_input)")
    args = parser.parse_args()

    images = Path(args.input_dir).resolve()
    if not images.exists():
        print(f"ERROR: Input directory not found: {images}")
        return 1

    name = images.name
    workspace = images.parent / f"{name}_colmap_workspace"
    database = workspace / "database.db"
    sparse_out = workspace / "sparse_global"
    optimized = workspace / "optimized_model"

    # ── Determine 3DGS output directory ────────────────────────
    gs_input_dir = Path(args.output_dir) if args.output_dir else images.parent / f"{name}_3dgs_input"

    print("=" * 60)
    print(f"  COLMAP → 3DGS Pipeline")
    print(f"  Input:         {images}/")
    print(f"  Workspace:     {workspace}/")
    print(f"  3DGS output:   {gs_input_dir}/")
    print("=" * 60)

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(sorted(images.glob(ext)))
    print(f"\n  Found {len(image_files)} images")

    # ── Step 1: Feature extraction & matching ──────────────────
    workspace.mkdir(parents=True, exist_ok=True)
    if database.exists() and args.skip_matching:
        print(f"\n  Found existing database: {database}")
        print("  Skipping feature extraction and matching.")
    else:
        extract_features(images, database)
        match_exhaustive(database)

    # ── Step 2: Global SfM ─────────────────────────────────────
    model_bin = optimized / "images.bin"
    model_txt = optimized / "images.txt"
    if model_bin.exists() or model_txt.exists():
        print(f"\n  Reconstruction already exists at {optimized}")
    else:
        run_global_mapper(database, images, sparse_out)
        count, best = find_best_model(sparse_out)
        if best:
            print(f"\n  Registered: {count} images in {best.name}")
            if optimized.exists():
                shutil.rmtree(optimized)
            shutil.copytree(best, optimized)
        else:
            print("\n  No model produced. Try removing --skip-matching.")
            return 1

    # ── Step 3: Copy images ────────────────────────────────────
    gs_images = gs_input_dir / "images"
    n_copied = copy_images(images, gs_images)
    print(f"\n  Copied {n_copied} images to {gs_images}")

    # ── Step 4: Build 3DGS input (SIMPLE_RADIAL text model) ───
    gs_sparse = gs_input_dir / "sparse" / "0"
    build_3dgs_input(gs_sparse, optimized)
    export_ply(optimized, workspace / "sparse.ply")

    # ── Print summary ──────────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  3DGS input folder: {gs_input_dir}/")
    print(f"    {gs_images}/")
    print(f"    {gs_sparse}/")
    print(f"      cameras.txt")
    print(f"      images.txt")
    print(f"      points3D.txt")
    print(f"  {'='*50}")
    print(f"\n  Train locally:")
    print(f"    python3 scripts/train_3dgs_enhanced.py -i {gs_input_dir}")
    print(f"\n  Upload to Colab:")
    print(f"    cd {gs_input_dir.parent} && zip -r {name}_3dgs_input.zip {name}_3dgs_input/")
    print(f"    Upload the zip to Google Drive, then in Colab:")
    print(f"    - Extract to /content/gaussian-splatting/input")
    print(f"    - Run Steps 1, 2, then skip to Step 6")
    print(f"\n  Workspace (intermediate files, can be deleted):")
    print(f"    {workspace}/")

    return 0

if __name__ == "__main__":
    sys.exit(main())
