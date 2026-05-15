# 3D Arena Reconstruction with Gaussian Splatting

Reconstruct a 10-15m arena from 91 photos using COLMAP + 3D Gaussian Splatting. Explore the result in Unity, browser, or desktop viewer.

**Camera:** Google Pixel 10 Pro XL (6.9mm, 24mm equiv., f/1.68)  
**COLMAP model:** SIMPLE_RADIAL (k1=0.0065) → **merged 34 registered images**  
**Images:** 91 at 1920×1445 → 30+4 registered after model merging  
**Sparse cloud:** ~16K points → **200K-500K after 3DGS densification**

## Quick Start

```bash
# Full pipeline (COLMAP → 3DGS → Unity export)
python3 run_pipeline.py --full

# COLMAP optimization only (merges multiple reconstructions)
python3 run_pipeline.py --colmap-only

# Requires GPU for training:
python3 run_pipeline.py --train-only --iterations 50000
```

### Google Colab (free GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaarthik-balakrishnan/arena-3dgs/blob/main/arena_3dgs_colab.ipynb)

| Step | Cell | What | Time |
|------|------|------|------|
| 1 | Mount Drive | Google Drive checkpoint save | 10s |
| 2 | Install | COLMAP + PyTorch + 3DGS + gsplat | 3 min |
| 3B | Images | Download 91 images from GitHub | 2 min |
| 5 | COLMAP | Pre-computed 34-camera merged model | 10s |
| 6 | Convert | Prep data for 3DGS format | 10s |
| 7B | Train 30K | Full quality densification | 30 min |
| 8 | Export | Download `.ply` for Unity/SuperSplat | 10s |

## Training Quality

| Config | Iters | Gaussians | T4 Time | Quality |
|--------|-------|-----------|---------|---------|
| Quick | 3,000 | ~50K | ~7 min | Test, blocky |
| Standard | 30,000 | ~200K+ | ~30 min | Production |
| **High** | **50,000** | **~500K** | **~50 min** | **Unity-ready** |
| gsplat | 30,000 | ~200K+ | ~15 min | Same, 2x faster |

## Pipeline Improvements

| Improvement | What | Gain |
|------------|------|------|
| **Model merging** | Two independent COLMAP runs merged → **34 registered images** | +13% views |
| **Enhanced training** | gsplat with densification, L1+L2 loss, scale reg | Sharper details |
| **Pipeline** | `run_pipeline.py` orchestrates everything | 1-command end-to-end |

## Viewing the 3D Scene

### Unity (Best Exploration)
```bash
# Validate PLY format first
python3 scripts/export_unity.py output/arena_3dgs.ply
```
- Install [Unity 2022.3+](https://unity.com/) with URP
- Clone [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)
- Drop `.ply` into `Assets/GaussianAssets/`
- WASD + mouse-look walkthrough

### SuperSplat (No Install)
- Go to [SuperSplat](https://supersplat.com/)
- Drag & drop your `.ply` file

## Files

| File | Description |
|------|-------------|
| `arena_3dgs_colab.ipynb` | **Main Colab notebook** |
| `run_pipeline.py` | **Master pipeline orchestrator** |
| `scripts/run_colmap.py` | COLMAP optimization + model merging |
| `scripts/train_3dgs_enhanced.py` | Enhanced GPU training with densification |
| `scripts/export_unity.py` | Unity PLY validator + exporter |
| `colmap_data/` | Pre-computed COLMAP (30 registered) |
| `colmap_workspace/optimized_model/` | **Merged model (34 registered)** |
| `splat-files-processed/` | 91 resized images (1920px) |
| `output/arena_sparse_pointcloud.ply` | Sparse COLMAP cloud (16K pts) |
| `output/arena_3dgs.ply` | **Trained 3DGS model (requires GPU)** |

## Requirements

- **GPU**: Google Colab (free T4) or local NVIDIA GPU with 15GB+ VRAM
- **Local**: Python 3.9+, COLMAP 4.x (for SfM only)

## References

- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) — Kerbl et al. 2023
- [gsplat](https://github.com/nerfstudio-project/gsplat) — 2x faster CUDA rasterizer
- [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) — Unity viewer
- [SuperSplat](https://supersplat.com/) — Web-based 3DGS viewer
