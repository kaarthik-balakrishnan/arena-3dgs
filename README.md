# 3D Arena Reconstruction with Gaussian Splatting

Reconstruct a 10-15m arena from 91 photos using COLMAP + 3D Gaussian Splatting.

**Camera:** Google Pixel 10 Pro XL (6.9mm, 24mm equiv., f/1.68)  
**COLMAP model:** SIMPLE_RADIAL (radial distortion: k1=0.0065)  
**Images:** 91 at 1920×1445, **30 registered** by COLMAP  
**Sparse cloud:** ~16K points → **200K+ after 3DGS densification**

## Quick Start (Google Colab)

**Total time:** ~35 minutes on free Colab T4 GPU (30K iterations)

| Step | Cell | What it does | Time |
|------|------|-------------|------|
| 1 | Mount Drive | Mount Google Drive for checkpoint saving | 10s |
| 2 | Install deps | Installs COLMAP, PyTorch, 3DGS + gsplat | 3 min |
| 3B | Download images | Pulls 91 resized images from GitHub | 2 min |
| 5 | Download COLMAP data | Uses pre-computed camera poses | 10s |
| 6 | Convert format | Prepares data for 3DGS training | 10s |
| 7B | Train (30,000 iters) | Full quality training with densification | 30 min |
| 8 | Export results | Downloads .ply point cloud | 10s |

### Instructions

1. Open the notebook in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaarthik-balakrishnan/arena-3dgs/blob/main/arena_3dgs_colab.ipynb)

2. Run cells **in order** from top to bottom.

3. Mount Google Drive when prompted.

4. **Step 3B** downloads 91 images from GitHub.

5. **Step 5** downloads pre-computed COLMAP data (skips ~20 min COLMAP run).

6. **Step 7B** trains for 30K iterations (recommended). Use **7A** (3K) for a quick test.

7. **Step 8** exports and downloads the `.ply` point cloud.

## Pipeline Details

### Camera Model

The Pixel 10 Pro XL has a 6.9mm lens (24mm full-frame equivalent). We use `SIMPLE_RADIAL` 
(fx, cx, cy, k1) to model the slight radial distortion (k1=0.0065). The `PINHOLE` model
previously used ignored this distortion. Using `OPENCV` (k1,k2,p1,p2) is also valid.

### COLMAP Registration

Only 30/91 images are registered by COLMAP due to the arena's repetitive textures
(uniform green floor, plain walls). This is a known limitation of SfM on such scenes.
The 3DGS training compensates through its densification process (growing from ~16K to 
200K+ Gaussians), filling in unobserved regions.

To improve registration:
- **Sequential matcher** connects consecutive frames (walking path around arena)
- **Exhaustive matcher** provides long-range loop closure
- **More SIFT features** (16384 vs default 8192)

### Training Quality

| Iterations | Gaussians | Quality | Time (T4) |
|-----------|-----------|---------|-----------|
| 3,000 | ~50K | Test quality, blocky | ~7 min |
| 30,000 | ~200K+ | Full quality, detailed | ~30 min |

The official 3DGS training uses adaptive density control (clone/split/prune) which 
grows the model from the initial sparse cloud to a dense representation over 30K iterations.

## Viewing & Exploring the 3D Scene

### SuperSplat (Recommended, No Install)
- Go to [SuperSplat](https://supersplat.com/)
- Drag & drop your `.ply` file
- Interactive 3DGS viewer with editing tools
- Works in browser, free

### Unity Walk-through (Best for Exploration)
Unity is excellent for walking through the arena:
1. Install [Unity 2022.3+](https://unity.com/) with URP
2. Use [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) by Aras Pranckevičius
3. Drop your `.ply` into the Assets folder
4. Full WASD + mouse-look controls, WebGL export possible

### Other Options
- **gsplat.js** — embed 3DGS in web pages
- **CloudCompare** — point cloud analysis
- **MeshLab** — basic PLY viewer

## Files

| File | Description |
|------|-------------|
| `arena_3dgs_colab.ipynb` | **Main Colab notebook** |
| `colmap_data/` | Pre-computed COLMAP output (cameras.txt, images.txt, points3D.txt) |
| `splat-files/` | Original 91 photos |
| `splat-files-processed/` | Resized to 1920px width |
| `colmap_workspace/` | Full COLMAP workspace with binary + text output |
| `output/arena_sparse_pointcloud.ply` | Sparse point cloud from COLMAP |
| `scripts/prepare_images.py` | Resize images for Colab upload |
| `scripts/train_3dgs.py` | Local training script using gsplat |
| `scripts/export_pointcloud.py` | Export COLMAP sparse cloud to PLY |

## Local Run (CPU only)

COLMAP SfM runs fine on CPU. 3DGS training requires a CUDA GPU (use Colab).

```bash
# Resize images
python3 scripts/prepare_images.py --input splat-files --output splat-files-processed

# Run COLMAP with correct camera model
colmap feature_extractor --database_path colmap_workspace/database.db \
    --image_path splat-files-processed \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_num_features 16384

colmap sequential_matcher --database_path colmap_workspace/database.db \
    --SequentialMatching.overlap 10

colmap exhaustive_matcher --database_path colmap_workspace/database.db

colmap mapper --database_path colmap_workspace/database.db \
    --image_path splat-files-processed --output_path colmap_workspace/sparse

# Export sparse point cloud
python3 scripts/export_pointcloud.py
```

## Requirements

- **GPU**: Google Colab (free T4) recommended for training
- **VRAM**: 15GB+ for 30K iterations
- **Local**: Python 3.9+, COLMAP 4.x (for SfM only)

## References

- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) — Kerbl et al. 2023
- [Official 3DGS Repo](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Faster CUDA rasterizer
- [COLMAP](https://colmap.github.io/) — SfM pipeline
- [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) — Unity viewer
- [SuperSplat](https://supersplat.com/) — Web-based 3DGS viewer
