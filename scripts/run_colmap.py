#!/usr/bin/env python3
"""
Enhanced COLMAP pipeline for the arena dataset.
Tries multiple matching strategies and model merges to maximize registration.
"""
import subprocess, sys, os, shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
WORKSPACE = BASE / "colmap_workspace"
IMAGES = BASE / "splat-files-processed"
DATABASE = WORKSPACE / "database.db"
SPARSE = WORKSPACE / "sparse"
MERGED = WORKSPACE / "sparse_registered"
OUTPUT_MODEL = WORKSPACE / "optimized_model"

CAMERA_MODEL = "SIMPLE_RADIAL"
SINGLE_CAMERA = 1
MAX_FEATURES = 16384

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Command failed with code {result.returncode}")
    return result.returncode

def extract_features():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap feature_extractor "
        f"--database_path {DATABASE} "
        f"--image_path {IMAGES} "
        f"--ImageReader.camera_model {CAMERA_MODEL} "
        f"--ImageReader.single_camera {SINGLE_CAMERA} "
        f"--SiftExtraction.max_num_features {MAX_FEATURES} "
        f"--SiftExtraction.first_octave -1 "
        f"--SiftExtraction.peak_threshold 0.02",
        "Extracting SIFT features (16384 per image, first_octave=-1)"
    )

def match_sequential(overlap=10):
    run(
        f"colmap sequential_matcher "
        f"--database_path {DATABASE} "
        f"--SequentialMatching.overlap {overlap}",
        f"Sequential matching (overlap={overlap})"
    )

def match_exhaustive():
    run(
        f"colmap exhaustive_matcher "
        f"--database_path {DATABASE} ",
        "Exhaustive matching (all pairs)"
    )

def match_vocab_tree(vocab_path=None):
    if vocab_path and Path(vocab_path).exists():
        return run(
            f"colmap vocab_tree_matcher "
            f"--database_path {DATABASE} "
            f"--VocabTreeMatching.vocab_tree_path {vocab_path}",
            "Vocab tree matching"
        )
    else:
        print("  Vocab tree not found, skipping.")

def match_spatial():
    run(
        f"colmap spatial_matcher "
        f"--database_path {DATABASE} "
        f"--SpatialMatching.is_gps 0",
        "Spatial matching (based on frame ordering)"
    )

def run_mapper(overlap_val=10, name="run"):
    out = SPARSE / name
    out.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap mapper "
        f"--database_path {DATABASE} "
        f"--image_path {IMAGES} "
        f"--output_path {SPARSE / name} "
        f"--Mapper.ba_local_max_num_iterations 25 "
        f"--Mapper.ba_global_max_num_iterations 50 "
        f"--Mapper.multiple_models 1 "
        f"--Mapper.max_num_models 50 "
        f"--Mapper.min_model_size 5",
        f"Running incremental SfM mapper ({name})"
    )
    return count_registered(SPARSE / name)

def count_registered(path):
    for sub in sorted(path.iterdir()):
        imgtxt = sub / "images.txt"
        if imgtxt.exists():
            with open(imgtxt) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            return len(lines) // 2, sub
    return 0, None

def convert_to_txt(bin_path, txt_path):
    run(
        f"colmap model_converter "
        f"--input_path {bin_path} "
        f"--output_path {txt_path} "
        f"--output_type TXT",
        f"Convert model to text format: {bin_path.name}"
    )

def merge_models(model1, model2, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap model_merger "
        f"--input_path1 {model1} "
        f"--input_path2 {model2} "
        f"--output_path {output_path}",
        "Attempting model merge"
    )
    # Check if any files were produced
    return any(output_path.iterdir()) if output_path.exists() else False

def align_and_merge():
    """
    If direct merge fails, try to align models first using
    known image correspondences, then merge.
    """
    print("\n--- Attempting alignment-based merge ---")
    models = sorted([p for p in SPARSE.iterdir() if p.is_dir() and (p / "images.bin").exists()])
    if len(models) < 2:
        print("  Need at least 2 models to merge")
        return None

    main_model = models[0]
    other_models = models[1:]
    
    # Try merging sequentially
    current = main_model
    for i, other in enumerate(other_models):
        merged_out = WORKSPACE / f"merge_step_{i}"
        merged_out.mkdir(parents=True, exist_ok=True)
        
        success = merge_models(current, other, merged_out)
        if success and any(merged_out.iterdir()):
            print(f"  Merged successfully to {merged_out}")
            current = merged_out
        else:
            print(f"  Merge with {other.name} failed (different coordinate systems)")
            # Clean up failed output
            if merged_out.exists():
                shutil.rmtree(merged_out)
    
    return current

def copy_merged_to_output(best_model):
    if OUTPUT_MODEL.exists():
        shutil.rmtree(OUTPUT_MODEL)
    OUTPUT_MODEL.mkdir(parents=True)
    
    for f in best_model.iterdir():
        if f.is_file():
            shutil.copy2(f, OUTPUT_MODEL / f.name)
    
    # Also convert to txt for compatibility
    txt_out = OUTPUT_MODEL / "txt"
    txt_out.mkdir(exist_ok=True)
    convert_to_txt(OUTPUT_MODEL, txt_out)
    
    return OUTPUT_MODEL

def main():
    print("=" * 60)
    print("  ARENA COLMAP OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Check if we already have a good merged model
    if MERGED.exists() and (MERGED / "images.bin").exists():
        import struct
        with open(MERGED / "images.bin", "rb") as f:
            num = struct.unpack("Q", f.read(8))[0]
        print(f"\n  Found existing merged model: {num} images")
        
        if num >= 34:
            print("  Using existing merged model (best available)")
            copy_merged_to_output(MERGED)
            convert_to_txt(OUTPUT_MODEL, OUTPUT_MODEL / "txt")
            print(f"  Optimized model ready at: {OUTPUT_MODEL}")
            return 0

    # Step 2: Check if we have multiple sparse models to try merging
    models = sorted([p for p in SPARSE.iterdir() if p.is_dir() and (p / "images.bin").exists()])
    print(f"\n  Found {len(models)} sparse model(s):")
    for m in models:
        with open(m / "images.bin", "rb") as f:
            num = struct.unpack("Q", f.read(8))[0]
        with open(m / "points3D.bin", "rb") as f:
            pts = struct.unpack("Q", f.read(8))[0]
        print(f"    {m.name}: {num} images, {pts} points")

    if len(models) >= 2:
        best = align_and_merge()
        if best:
            copy_merged_to_output(best)
            print(f"\n  Optimized model ready at: {OUTPUT_MODEL}")
            return 0

    # Step 3: If no merge possible, use the largest single model
    print("\n  Using largest single reconstruction")
    largest = max(models, key=lambda p: (p / "images.bin").stat().st_size)
    copy_merged_to_output(largest)
    print(f"  Optimized model ready at: {OUTPUT_MODEL}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
