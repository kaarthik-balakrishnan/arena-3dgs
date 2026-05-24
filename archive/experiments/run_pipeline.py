#!/usr/bin/env python3
"""
Master pipeline: images → optimized COLMAP → 3DGS training → Unity-ready PLY

Orchestrates the full reconstruction pipeline for the arena dataset.
Can run individual stages or the full pipeline.

Usage:
  # Run full pipeline (COLMAP + 3DGS training + export)
  python run_pipeline.py --full

  # Only optimize COLMAP (merge existing models)
  python run_pipeline.py --colmap-only

  # Only train 3DGS (requires GPU)
  python run_pipeline.py --train-only --iterations 30000

  # Only validate/export PLY for Unity
  python run_pipeline.py --export-only

  # Quick test (3K iterations)
  python run_pipeline.py --full --quick
"""
import sys, os, subprocess, argparse, shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPTS = BASE / "scripts"
WORKSPACE = BASE / "colmap_workspace"
IMAGES = BASE / "splat-files-processed"
COLMAP_DATA = BASE / "colmap_data"
OUTPUT = BASE / "output"

def check_colmap():
    try:
        subprocess.run(["colmap", "help"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_pytorch():
    try:
        import torch
        return True
    except ImportError:
        return False

def run_script(name, extra_args=None):
    script = SCRIPTS / name
    if not script.exists():
        print(f"Error: {script} not found")
        return False
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'#'*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'#'*60}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0

def stage_colmap():
    print("\n" + "=" * 60)
    print("  STAGE 1: COLMAP Optimization")
    print("=" * 60)

    if not check_colmap():
        print("  COLMAP not found. Install with: brew install colmap")
        return False

    # Run the colmap optimization script
    return run_script("run_colmap.py")

def stage_training(args):
    print("\n" + "=" * 60)
    print("  STAGE 2: 3D Gaussian Splatting Training")
    print("=" * 60)

    if not check_pytorch():
        print("  PyTorch not found. This stage requires a GPU (CUDA/MPS).")
        print("  Use the Colab notebook for training, or install PyTorch + CUDA locally.")
        print()
        print("  For Colab training:")
        print("    1. Open arena_3dgs_colab.ipynb in Colab")
        print("    2. Run cells 1-6 to set up data")
        print("    3. Run cell 7B for 30K iterations")
        print("    4. Run cell 8 to download the PLY")
        print()
        
        answer = input("  Continue anyway? (y/N): ").strip().lower()
        if answer != 'y':
            return False

    iters = 3000 if args.quick else 30000
    print(f"  Training for {iters} iterations")

    # Determine input directory - prefer optimized model
    input_dir = BASE / "gaussian-splatting" / "input"
    
    # If we have an optimized model, copy it to the expected location
    opt_model = WORKSPACE / "optimized_model" / "txt"
    if opt_model.exists() and (opt_model / "cameras.txt").exists():
        print(f"  Using optimized model from {opt_model}")
    
    success = run_script("train_3dgs_enhanced.py", [
        "--input-dir", str(input_dir),
        "--output-dir", str(OUTPUT),
        "--iterations", str(iters),
        "--max-gaussians", str(500000 if not args.quick else 50000),
    ])
    return success

def stage_export():
    print("\n" + "=" * 60)
    print("  STAGE 3: Unity Export & Validation")
    print("=" * 60)

    return run_script("export_unity.py")

def stage_compression(args):
    print("\n" + "=" * 60)
    print("  STAGE 4: Post-Training Compression")
    print("=" * 60)

    ply_files = sorted(OUTPUT.glob("arena_3dgs.ply"))
    if not ply_files:
        ply_files = sorted(OUTPUT.glob("*.ply"))
    if not ply_files:
        print("  ERROR: No PLY files found in output/")
        return False

    ply_path = ply_files[-1]
    quality = args.compress_quality
    print(f"  Compressing: {ply_path.name} (quality: {quality})")

    extra = [str(ply_path), "--quality", quality, "--output-dir", str(OUTPUT)]
    return run_script("compress_splat.py", extra)

def prepare_gaussian_splatting_input():
    """Set up the gaussian-splatting/input directory structure."""
    gs_dir = BASE / "gaussian-splatting"
    input_dir = gs_dir / "input"
    images_dir = input_dir / "images"
    sparse_dir = input_dir / "sparse" / "0"

    print("\n--- Preparing 3DGS input directory ---")

    # Create images directory
    if IMAGES.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in IMAGES.glob("*.jpg"):
            if not (images_dir / img.name).exists():
                shutil.copy2(img, images_dir / img.name)
        n_imgs = len(list(images_dir.glob("*.jpg")))
        print(f"  Copied {n_imgs} images to {images_dir}")
    else:
        print(f"  WARNING: No images found at {IMAGES}")

    # Create sparse directory from optimized model
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Prefer optimized model, then registered, then colmap_data
    for src in [
        WORKSPACE / "optimized_model" / "txt",
        WORKSPACE / "sparse_registered" / "txt",
        COLMAP_DATA,
        WORKSPACE / "sparse" / "0" / "txt",
    ]:
        if src.exists() and (src / "cameras.txt").exists():
            for f in ["cameras.txt", "images.txt", "points3D.txt"]:
                src_f = src / f
                if src_f.exists():
                    shutil.copy2(src_f, sparse_dir / f)
            # Count registered images
            with open(sparse_dir / "images.txt") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            n_reg = len(lines) // 2
            print(f"  Using COLMAP model from {src} ({n_reg} images)")
            break
    else:
        print("  WARNING: No COLMAP data found")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Arena 3DGS Pipeline: images → COLMAP → 3DGS → Unity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --full                  # Full pipeline (incl. compress)
  python run_pipeline.py --full --quick          # Quick test (3K iterations)  
  python run_pipeline.py --colmap-only           # Only optimize COLMAP
  python run_pipeline.py --train-only            # Only train 3DGS
  python run_pipeline.py --compress              # Compress trained PLY
  python run_pipeline.py --compress --compress-quality low  # Aggressive compression
  python run_pipeline.py --export-only           # Validate PLY for Unity
  python run_pipeline.py --setup-only            # Prepare 3DGS input dir
        """
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3K iterations)")
    parser.add_argument("--colmap-only", action="store_true", help="Only COLMAP")
    parser.add_argument("--train-only", action="store_true", help="Only 3DGS training")
    parser.add_argument("--export-only", action="store_true", help="Only export/validate")
    parser.add_argument("--setup-only", action="store_true", help="Only prepare input dirs")
    parser.add_argument("--iterations", type=int, default=30000, 
                        help="Training iterations (default: 30000)")
    parser.add_argument("--compress", action="store_true",
                        help="Run post-training compression (SH clustering + quantization)")
    parser.add_argument("--compress-quality", "-cq", default="medium",
                        choices=["very_high", "high", "medium", "low", "very_low"],
                        help="Compression quality preset (default: medium)")

    args = parser.parse_args()

    # If no args, show help
    if not any([args.full, args.colmap_only, args.train_only, 
                args.export_only, args.setup_only, args.compress]):
        parser.print_help()
        print("\n  Current state:")
        
        # Check for PLY files
        ply_files = list(OUTPUT.glob("*.ply"))
        if ply_files:
            for pf in ply_files:
                print(f"    PLY: {pf} ({pf.stat().st_size / 1024**2:.1f} MB)")
        
        # Check merged model
        merged = WORKSPACE / "sparse_registered"
        if merged.exists():
            import struct
            with open(merged / "images.bin", "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
            print(f"    Optimized COLMAP: {n} registered images")
        
        print()
        return 0

    if args.setup_only:
        prepare_gaussian_splatting_input()
        return 0

    if args.colmap_only or args.full:
        if not stage_colmap():
            print("  COLMAP stage failed")
            if not args.full:
                return 1

    if args.setup_only or (args.full and args.colmap_only 
                           and stage_colmap.__code__.co_filename == "COLOPT"):
        pass  # fall through

    # Prepare 3DGS input
    if args.full or args.train_only:
        prepare_gaussian_splatting_input()

    if args.train_only or args.full:
        # Copy input args for sub-stages
        stage_training(args)

    if args.export_only or args.full:
        stage_export()

    if args.compress or args.full:
        stage_compression(args)

    if args.full:
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
        ply_files = sorted(OUTPUT.glob("*.ply"))
        if ply_files:
            print(f"\n  Output PLY files:")
            for pf in ply_files:
                print(f"    {pf} ({pf.stat().st_size / 1024**2:.1f} MB)")
        print("\n  Next steps:")
        print("    1. Open the PLY in SuperSplat: https://supersplat.com/")
        print("    2. Or use Unity: https://github.com/aras-p/UnityGaussianSplatting")
        print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
