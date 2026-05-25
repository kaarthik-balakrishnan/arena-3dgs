#!/usr/bin/env python3
"""
Run COLMAP on a set of images and produce 3DGS-ready output.

Usage:
  python3 scripts/run_colmap.py -i sharp-frames
  python3 scripts/run_colmap.py -i manual-selection
  python3 scripts/run_colmap.py -i manual-selection --skip-matching

Output:
  {name}_colmap_workspace/
    images/           # copies of input images
    sparse/0/         # 3DGS-ready COLMAP model (PINHOLE, binary + text)
      cameras.txt     # PINHOLE format
      images.txt
      points3D.txt
      cameras.bin
      images.bin
      points3D.bin
    optimized_model/  # raw COLMAP output (SIMPLE_RADIAL)
      txt/
        cameras.txt
        images.txt
        points3D.txt
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
    """Find colmap binary."""
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
        "Running global SfM (rotation averaging + global positioning)"
    )

def convert_to_pinhole(model_dir):
    """Convert SIMPLE_RADIAL → PINHOLE in cameras.txt in-place."""
    cam_path = model_dir / "cameras.txt"
    if not cam_path.exists():
        print(f"  WARNING: cameras.txt not found at {cam_path}")
        return False
    with open(cam_path) as f:
        lines = f.readlines()
    modified = False
    with open(cam_path, "w") as f:
        for line in lines:
            if line.startswith("#") or not line.strip():
                f.write(line)
            else:
                parts = line.strip().split()
                if parts[1] == "SIMPLE_RADIAL":
                    f.write(f"{parts[0]} PINHOLE {parts[2]} {parts[3]} {parts[4]} {parts[4]} {parts[5]} {parts[6]}\n")
                    modified = True
                else:
                    f.write(line)
    if modified:
        print("  Converted camera model: SIMPLE_RADIAL → PINHOLE")
    return True

def rebuild_binaries(model_dir):
    """Rebuild cameras.bin, images.bin, points3D.bin from text files."""
    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        p = model_dir / fn
        if p.exists():
            os.remove(p)
    tmp = model_dir.parent / "_tmp_bin"
    tmp.mkdir(parents=True, exist_ok=True)
    run(
        f"{CM} model_converter "
        f"--input_path {model_dir} "
        f"--output_path {tmp} "
        f"--output_type BIN",
        "Rebuilding binary files from text"
    )
    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = tmp / fn
        if src.exists():
            shutil.copy2(src, model_dir / fn)
    shutil.rmtree(tmp, ignore_errors=True)

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
    """Find the best reconstruction in sparse_out (highest image count)."""
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

def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP pipeline and produce 3DGS-ready output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python3 scripts/run_colmap.py -i sharp-frames\n  python3 scripts/run_colmap.py -i manual-selection --skip-matching",
    )
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Directory containing input images")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip feature extraction and matching if database exists")
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
    gs_dir = workspace / "sparse" / "0"
    images_dst = workspace / "images"

    print("=" * 60)
    print(f"  COLMAP → 3DGS Pipeline")
    print(f"  Input:  {images}/")
    print(f"  Output: {workspace}/")
    print("=" * 60)

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(sorted(images.glob(ext)))
    print(f"\n  Found {len(image_files)} images")

    # ── Step 1: Feature extraction ────────────────────────────
    workspace.mkdir(parents=True, exist_ok=True)
    if database.exists() and args.skip_matching:
        print(f"\n  Found existing database: {database}")
        print("  Skipping feature extraction and matching.")
    else:
        extract_features(images, database)
        match_exhaustive(database)

    # ── Step 2: Global SfM ────────────────────────────────────
    if (optimized / "images.bin").exists() or (optimized / "images.txt").exists():
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

    # ── Step 3: Copy images ───────────────────────────────────
    n_copied = copy_images(images, images_dst)
    print(f"\n  Copied {n_copied} images to {images_dst}")

    # ── Step 4: Build 3DGS input (PINHOLE + binaries) ────────
    if optimized.exists():
        model_txt = optimized / "txt"
        if not model_txt.exists():
            model_txt.mkdir(exist_ok=True)
            convert_to_txt = lambda: run(
                f"{CM} model_converter --input_path {optimized} --output_path {model_txt} --output_type TXT",
                "Converting model to text format"
            )
            convert_to_txt()

        # Create sparse/0/ as a copy of the text model, then convert
        if gs_dir.exists():
            shutil.rmtree(gs_dir)
        shutil.copytree(model_txt, gs_dir)
        print(f"\n  Preparing 3DGS input at {gs_dir}")

        convert_to_pinhole(gs_dir)
        rebuild_binaries(gs_dir)
        export_ply(gs_dir, workspace / "sparse.ply")

        # Count registered images
        with open(gs_dir / "images.txt") as f:
            n_imgs = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
        print(f"\n  {'='*50}")
        print(f"  3DGS-ready output: {workspace}/")
        print(f"  {n_imgs} images, PINHOLE camera model")
        print(f"  {'='*50}")
        print(f"\n  Train locally:")
        print(f"    python3 scripts/train_3dgs_enhanced.py -i {workspace}")
        print(f"\n  View sparse point cloud:")
        print(f"    python3 scripts/viz_colmap.py --input {gs_dir}")
        print(f"\n  Or upload to Colab notebook (zip images + sparse/0/)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
