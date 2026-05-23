#!/usr/bin/env python3
"""
Convert COLMAP sparse model (global mapper output, 91 images) to:
  1. output/arena_sparse_pointcloud.ply — simple XYZ+RGB for reference
  2. output/arena_3dgs_input/ — full 3DGS training input:
       images/            → all 91 images (symlinks)
       sparse/0/
         cameras.txt/bin  → PINHOLE (converted from SIMPLE_RADIAL)
         images.txt/bin   → same as COLMAP
         points3D.txt/bin → original COLMAP points
         points3D.ply     → 3DGS format: x,y,z,nx,ny,nz,f_dc_0/1/2

Usage:
  python3 scripts/export_3dgs_input.py
"""

import shutil, subprocess, struct
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

BASE = Path(__file__).resolve().parent.parent
MODEL_TXT = BASE / "colmap_workspace" / "sparse_global" / "0" / "txt"
IMAGES_SRC = BASE / "splat-files-processed"
OUTPUT = BASE / "output"
SIMPLE_PLY = OUTPUT / "arena_sparse_pointcloud.ply"
TRAINING_DIR = OUTPUT / "arena_3dgs_input"


def read_points_txt(path):
    points, colors = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                points.append([x, y, z])
                colors.append([r, g, b])
    return np.array(points), np.array(colors)


def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                cam_id = int(parts[0])
                cameras[cam_id] = {
                    "model": parts[1],
                    "width": int(parts[2]),
                    "height": int(parts[3]),
                    "params": list(map(float, parts[4:])),
                }
    return cameras


def write_cameras_pinhole(path, cameras):
    with open(path, "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            c = cameras[cam_id]
            if c["model"] == "SIMPLE_RADIAL":
                fx, cx, cy, k = c["params"]
                fh.write(f"{cam_id} PINHOLE {c['width']} {c['height']} {fx} {fx} {cx} {cy}\n")
            else:
                params_str = " ".join(str(p) for p in c["params"])
                fh.write(f"{cam_id} {c['model']} {c['width']} {c['height']} {params_str}\n")


def write_ply_3dgs(path, points, colors):
    """Write 3DGS-format PLY: x,y,z,nx,ny,nz,f_dc_0,f_dc_1,f_dc_2."""
    normals = np.zeros((len(points), 3), dtype=np.float32)
    f_dc = np.array(colors, dtype=np.float32) / 255.0

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    vertex = np.zeros(len(points), dtype=dtype)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["nx"] = normals[:, 0]
    vertex["ny"] = normals[:, 1]
    vertex["nz"] = normals[:, 2]
    vertex["f_dc_0"] = f_dc[:, 0]
    vertex["f_dc_1"] = f_dc[:, 1]
    vertex["f_dc_2"] = f_dc[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")],
            text=False, byte_order="<").write(str(path))
    print(f"  {path.name}: {len(points)} points, {path.stat().st_size / 1024 / 1024:.1f} MB")


def write_ply_simple(path, points, colors):
    """Write simple XYZ+RGB PLY for reference."""
    vertex = np.zeros(len(points), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
    print(f"  {path.name}: {len(points)} points, {path.stat().st_size / 1024 / 1024:.1f} MB")


def convert_binary(txt_dir, bin_dir):
    """Use colmap model_converter to build binary files from text."""
    result = subprocess.run(
        ["colmap", "model_converter",
         "--input_path", str(txt_dir),
         "--output_path", str(bin_dir),
         "--output_type", "BIN"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: Binary conversion failed: {result.stderr.strip()}")
        return False
    return True


def organize_images(src_dir, dst_dir):
    """Copy images to training input directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in sorted(src_dir.glob(ext)):
            dst = dst_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            count += 1
    print(f"  {count} images copied to {dst_dir}")
    return count


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COLMAP → 3DGS Input Converter")
    print("=" * 60)

    # ── Read COLMAP model ──────────────────────────────────────
    print(f"\n[1] Reading COLMAP model from {MODEL_TXT}...")
    points, colors = read_points_txt(MODEL_TXT / "points3D.txt")
    cameras = read_cameras_txt(MODEL_TXT / "cameras.txt")
    print(f"  {len(points)} points, {len(cameras)} camera(s)")

    # ── Simple reference PLY ────────────────────────────────────
    print(f"\n[2] Writing reference PLY...")
    write_ply_simple(SIMPLE_PLY, points, colors)

    # ── Build 3DGS training input ──────────────────────────────
    print(f"\n[3] Building 3DGS training input at {TRAINING_DIR}...")
    if TRAINING_DIR.exists():
        shutil.rmtree(TRAINING_DIR)

    sparse_dir = TRAINING_DIR / "sparse" / "0"
    sparse_dir.mkdir(parents=True)

    # Copy images.txt, points3D.txt as-is
    for fn in ["images.txt", "points3D.txt"]:
        shutil.copy2(str(MODEL_TXT / fn), str(sparse_dir / fn))
        print(f"  Copied {fn}")

    # Write PINHOLE cameras.txt
    write_cameras_pinhole(sparse_dir / "cameras.txt", cameras)
    print(f"  cameras.txt (converted to PINHOLE)")

    # Write 3DGS-format PLY
    write_ply_3dgs(sparse_dir / "points3D.ply", points, colors)

    # Convert to binary
    print(f"  Converting to binary...")
    bin_dir = OUTPUT / "_bin_temp"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if convert_binary(sparse_dir, bin_dir):
        for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = bin_dir / fn
            if src.exists():
                shutil.copy2(str(src), str(sparse_dir / fn))
                print(f"  Copied {fn}")
        shutil.rmtree(bin_dir, ignore_errors=True)
    else:
        print(f"  Binary files not available (model_converter may need output dir to exist)")

    # Organize images
    organize_images(IMAGES_SRC, TRAINING_DIR / "images")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    n_imgs = 0
    img_txt = sparse_dir / "images.txt"
    if img_txt.exists():
        with open(img_txt) as f:
            n_imgs = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
    print(f"  Training input ready at: {TRAINING_DIR}")
    print(f"    {n_imgs} images")
    print(f"    {len(points)} points")
    print(f"    {len(cameras)} camera(s) (PINHOLE)")
    print(f"    points3D.ply (3DGS format)")
    print(f"    *.bin files")
    print(f"\n  Reference PLY: {SIMPLE_PLY}")
    print(f"\n  To use: zip -r output/arena_3dgs_input.zip output/arena_3dgs_input/")
    print(f"  Upload to Colab, extract to /content/gaussian-splatting/input/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
