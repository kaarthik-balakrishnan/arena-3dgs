# 3D Arena Reconstruction with Gaussian Splatting

Reconstruct a 10-15m arena from 91 photos using COLMAP + 3D Gaussian Splatting.

## Quick Start (Google Colab)

**Total time:** ~20 minutes on free Colab T4 GPU

| Step | Cell | What it does | Time |
|------|------|-------------|------|
| 1 | Mount Drive | Mount Google Drive for checkpoint saving | 10s |
| 2 | Install deps | Installs COLMAP, PyTorch, 3DGS | 3 min |
| 3B | Download images | Pulls 91 resized images from GitHub | 2 min |
| 5 | Download COLMAP data | Uses pre-computed camera poses (skips 10 min COLMAP run) | 10s |
| 6 | Convert format | Prepares data for 3DGS training | 10s |
| 7A | Train (3000 iters) | Quick training run | 7 min |
| 8 | Export results | Downloads .ply point cloud | 10s |

### Instructions

1. Open the notebook in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaarthik-balakrishnan/arena-3dgs/blob/main/arena_3dgs_colab.ipynb)

2. Run cells **in order** from top to bottom.

3. When prompted, mount your Google Drive (needed for checkpoint backups).

4. For **Step 3B** (download images): The notebook will download the 91 resized images from this GitHub repo automatically.

5. For **Step 5** (download COLMAP data): The notebook downloads pre-computed camera poses from this repo. This skips the 10-minute COLMAP step.

6. Run **Step 7A** (3000 iterations, ~7 min) or **Step 7B** (7000 iterations, ~15 min) for training.

7. **Step 8** exports and downloads the resulting `.ply` point cloud.

### Alternative: Manual Upload

If the GitHub download doesn't work, use the included `colab_training_data.zip` (26 MB):

1. In Colab, use **Step 3A** instead of 3B to upload the ZIP
2. Use **Step 5** to download pre-computed COLMAP data
3. Continue with Steps 6-8

## Files

| File | Description |
|------|-------------|
| `arena_3dgs_colab.ipynb` | **Main Colab notebook** — run this in Google Colab |
| `colab_training_data.zip` | Ready-to-upload ZIP with 91 images (for manual Colab use) |
| `splat-files/` | Original 91 photos |
| `splat-files-processed/` | Resized to 1920px width |
| `colmap_workspace/sparse/0/` | COLMAP output (camera poses + sparse point cloud) |
| `output/arena_sparse_pointcloud.ply` | Sparse point cloud from COLMAP (15,906 points) |
| `scripts/prepare_images.py` | Resize images for Colab upload |
| `scripts/train_3dgs.py` | Local training script (requires GPU / gsplat) |

## Local Run (CPU only)

COLMAP SfM runs fine on CPU. GPU-accelerated 3DGS training requires a CUDA GPU (not available on Intel Macs).

```bash
# Resize images
python scripts/prepare_images.py --input splat-files --output splat-files-processed

# Run COLMAP
colmap feature_extractor --database_path colmap_workspace/database.db \
    --image_path splat-files-processed --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path colmap_workspace/database.db
colmap mapper --database_path colmap_workspace/database.db \
    --image_path splat-files-processed --output_path colmap_workspace/sparse

# Export sparse point cloud
python scripts/export_pointcloud.py
```

## Viewing Results

Open the `.ply` file in:
- [MeshLab](https://www.meshlab.net/) — free 3D mesh viewer
- [CloudCompare](https://www.cloudcompare.org/) — point cloud processing
- [Potree](https://potree.org/) — web-based point cloud viewer

## Requirements

- **GPU**: Google Colab (free T4) recommended for training
- **VRAM**: 12GB+ (Colab T4 has 15GB)
- **Local**: Python 3.9+, COLMAP 4.x (for SfM only)

## References

- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) — Kerbl et al. 2023
- [Official 3DGS Repo](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP](https://colmap.github.io/) — SfM pipeline
