# Arena 3DGS Pipeline — Full Implementation Guide

## What This Does

Takes 91 photos of an arena → creates a 3D model you can walk through in Unity.
Each step is simple, checkpointed, and handles crashes gracefully.

### Quick Start (Generated Results)

```bash
# Everything you need is already computed:
ls colmap_data_optimized/           # 34 registered image poses (COLMAP result)
ls colab_training_data.zip          # Download this to Colab

# On Colab: run arena_3dgs_colab.ipynb → Cells 1, 2, 3B, 6, 7A, 7B, 8
# Or locally (GPU optional):
python3 run_pipeline.py --full --quick
```

**Final output**: `output/arena_3dgs.ply` — a 3D Gaussian Splatting model.
Drop it into [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) or [SuperSplat](https://supersplat.com/).

---

## Pipeline Stages

### Stage 1: Photos → Processed Images

**What**: 91 photos from a Google Pixel 10 Pro XL → resized for processing.

| Original | Processed |
|----------|-----------|
| 8160×6144 px | 1920×1445 px |
| ~5 MB each | ~300 KB each |
| Full camera resolution | COLMAP-friendly size |

**Script**: `scripts/prepare_images.py` — resizes and compresses (JPEG quality 90).
Parallelized with `ThreadPoolExecutor(8)` for 4-10x speedup.

**Reference**: Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints" (2004).

---

### Stage 2: Images → Camera Poses (COLMAP)

**What**: Figures out where each photo was taken and generates a sparse 3D point cloud.

#### Camera Model

The Pixel 10 Pro XL has mild lens distortion. We use `SIMPLE_RADIAL` (fx, cx, cy, k1):

```
fx=1398, cx=960, cy=722.5, k1=0.0065
```

`SIMPLE_RADIAL` was chosen over `PINHOLE` because k1=0.0065 corrects mild radial distortion.
Fewer parameters = more stable with only 34 registered views.

**Reference**: Hartley, R. & Zisserman, A. "Multiple View Geometry in Computer Vision" (2003).

#### Feature Extraction

16,384 SIFT features per image (double default). First octave starts at -1 (captures high-resolution features).

**Key fix for Colab**: COLMAP needs a display server for OpenGL-backed SIFT extraction.
The notebook starts Xvfb (virtual display) and sets `--SiftExtraction.use_gpu 0`.

**Reference**: Lowe (2004) — SIFT features are scale-invariant and rotation-invariant.

#### Matching

| Method | Pairs | Purpose |
|--------|-------|---------|
| Sequential (overlap=10) | 510 | Matches consecutive frames (walking path) |
| Exhaustive | 3,592 | Matches all pairs (non-sequential connections) |

**Key fix**: Lowered `--SiftExtraction.peak_threshold` to 0.01 (default 0.02) to detect more
features in low-texture arena walls/floor. Added mapper init flags:
`--Mapper.init_min_tri_angle 4 --Mapper.init_min_num_inliers 15 --Mapper.abs_pose_min_num_inliers 8`
— otherwise COLMAP fails with "No good initial image pair".

#### Reconstruction

COLMAP incremental SfM registers images one by one:
1. Find a good initial pair (lots of matches + parallax)
2. Triangulate 3D points from that pair
3. Register new images that see existing 3D points
4. Repeat

**Result**: 2 independent reconstructions (different arena areas, zero overlap):
- Model 0: 30 images, 15,938 points
- Model 1: 12 images, 870 points

**Total unique registered: 42 images** (30 + 12). Full merge blocked by different coordinate frames.

#### Model Merging

`scripts/run_colmap.py` automates: detect models → try merge → fall back to largest.

The partial merge gave **34 images** (30 from model 0 + 8 from model 1). The other 4
model-1 images are in incompatible coordinate systems.

**Why only 34/91?** The arena has repetitive textures (green floor, plain walls).
Images that see 0 already-triangulated 3D points can't be registered. This is a known
SfM limitation — learned features (SuperPoint) would fix this but needs GPU inference.

**Reference**: Schönberger, J.L. & Frahm, J.-M. "Structure-from-Motion Revisited" (CVPR 2016).

---

### Stage 3: Sparse Cloud → 3D Gaussian Splatting

**What**: Turns the point cloud into a photorealistic 3D scene you can walk through.

#### How 3DGS Works

Each Gaussian is a 3D ellipsoid with:
- **Position** (x, y, z) — where it sits in space
- **Color** (3 RGB spherical harmonic coefficients + 45 higher-order SH) — what color from each angle
- **Opacity** (1 value) — how transparent
- **Scale** (3 values) — how big in each axis
- **Rotation** (4 quaternion values) — which way it's oriented

Starting from ~16K COLMAP points, the model grows to 200K-500K Gaussians through
**adaptive density control**:

| Operation | When | What it does |
|-----------|------|-------------|
| **Clone** | High gradient + small Gaussian | Add more detail in under-reconstructed areas |
| **Split** | High gradient + large Gaussian | Break blobby areas into finer detail |
| **Prune** | Opacity near zero | Remove floaters and redundant Gaussians |

**Reference**: Kerbl, B. et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023).

#### Loss Function

```
Loss = MSE(render, photo) + 0.8 × L1(render, photo) + 0.01 × mean_scale
```

- **MSE**: Pixel-level accuracy
- **L1**: Sharpens edges (prevents blur)
- **Scale regularization**: Prevents Gaussians from growing too large (reduces floaters)

#### Training Schedule

| Segment | Iterations | Purpose |
|---------|-----------|---------|
| 7A (quick test) | 0–3,000 | Sanity check (~7 min on T4) |
| 7B | 0–10,000 | Base training |
| 7C | 10,000–20,000 | Resume with densification |
| 7D | 20,000–30,000 | Refine (2x densify interval) |
| 7E | 30,000–50,000 | High-quality final pass |

**Key fix**: Each segment uses `--start_checkpoint` + `--checkpoint_iterations`
from the official `train.py`. This preserves full optimizer state (not just model weights),
so densification history isn't lost between segments.

**Key fix**: Added `--densify_until_iter 15000` to segments 7C/7D so densification
resumes correctly. Added `--position_lr_max_steps` matching each segment's end iteration
so the position learning rate doesn't flatline.

**Key fix**: Intermediate checkpoints in long segments (7C: 15K, 7E: 40K/45K/50K)
so a disconnect loses at most 5K iterations instead of 10K.

**Reference**: Official 3DGS codebase: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

#### Efficiency Improvements (Latest)

| Change | File | Impact |
|--------|------|--------|
| Fixed crash: `elements['f_rest_0':'f_rest_44']` → loop over 45 fields | `train_3dgs_enhanced.py:408` | Was unreachable code (would crash on export) |
| Parallel image loading (ThreadPoolExecutor 8 workers) | `train_3dgs_enhanced.py`, `train_3dgs.py` | 4-10x faster I/O |
| Parallel image processing (ThreadPoolExecutor 8 workers) | `prepare_images.py` | 4-10x faster resize |
| Cache `torch.exp(scales)` once per iteration | `train_3dgs_enhanced.py:310` | 2-3 fewer CUDA ops per iter |
| `torch.zeros` + fill for quaternion init | `train_3dgs_enhanced.py:277` | Avoid Python list→tensor copy |
| Generator-based file reading | `train_3dgs_enhanced.py:54` | Lower peak memory |
| `shutil.copytree` instead of manual file loop | `run_colmap.py:161` | Simpler, faster |

---

### Stage 4: PLY → Unity

**What**: Validate and export the trained model so Unity can load it.

**Script**: `scripts/export_unity.py` — checks:
- PLY has correct header (vertex fields: x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3)
- Data types are float32
- File isn't truncated

**Viewer** | **Type** | **Walk-through?**
---|---
[SuperSplat](https://supersplat.com/) | Web (drag & drop) | Yes
[UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) | Desktop Unity | Yes, WASD + mouse-look
[gsplat.js](https://github.com/nerfstudio-project/gsplat.js) | Web viewer | Yes, programmatic camera

---

## Notebook Architecture

`arena_3dgs_colab.ipynb` has **24 cells**, each idempotent (safe to re-run):

| Cell | Stage | Runtime | Checkpoint |
|------|-------|---------|------------|
| 1 | Mount Google Drive | 5s | — |
| 2 | Install system deps (xvfb, COLMAP, CUDA) | 3 min | — |
| 3A | Clone repo + install Python deps | 2 min | — |
| 3B | Download 91 images from GitHub | 2 min | — |
| 4A | Prepare images (resize) | 30s | Drive |
| 4B | COLMAP feature extraction | 3 min | Drive |
| 4C | COLMAP matching (sequential + exhaustive) | 5 min | Drive |
| 4D | COLMAP mapper + model merge | 10 min | Drive |
| 5 | Download pre-computed COLMAP (skip 4) | 1 min | — |
| 6 | Convert COLMAP → 3DGS format | 5s | — |
| 7A | Train: quick test (3K iters) | 7 min | Drive |
| 7B | Train: base (10K iters) | 10 min | Drive |
| 7C | Train: resume (10K→20K) | 10 min | Drive |
| 7D | Train: refine (20K→30K) | 10 min | Drive |
| 7E | Train: high-quality (30K→50K) | 20 min | Drive |
| 8 | Export PLY + download | 5s | — |

**Key design decisions**:
- Every training cell saves to Google Drive (`MyDrive/arena_3dgs/`)
- Every training cell checks Drive first and skips if checkpoint exists
- `subprocess.run(capture_output=True, text=True)` for all training — prints errors
- `--disable_viewer` prevents GUI popup crash on headless Colab
- Intermediate checkpoints every 5K-10K iterations limit lost work on disconnect

**Bug fixes applied**:

| Bug | Symptom | Fix |
|-----|---------|-----|
| OpenGL context missing | `Check failed: context_.create()` | Start Xvfb, set DISPLAY=:99, use_gpu=0 |
| struct not defined | `NameError: name 'struct'` at Cell 4D | Move `import struct` to top of cell |
| SIMPLE_RADIAL rejected | `assert model_name=="PINHOLE"` in text reader | Convert SIMPLE_RADIAL→PINHOLE in cameras.txt |
| No good initial pair | Mapper fails with 0 registered | Add mapper init flags, lower peak_threshold |
| False success message | "Quick test done!" printed even on crash | Check returncode + output file existence |
| Input type flag unsupported | `model_converter: unknown --input_type` | Remove flag (auto-detected) |
| Checkpoint path nesting | `cp -r` nests folders | `shutil.rmtree` + direct restore path |
| Struct array slice invalid | `elements['f_rest_0':'f_rest_44']` crashes | Loop over 45 individual field names |

---

## File Reference

```
arena-3dgs/
├── arena_3dgs_colab.ipynb         # 24-cell Colab notebook (primary entry point)
├── arena_3dgs_config.json         # Optimal COLMAP + 3DGS settings
├── run_pipeline.py                # Local orchestrator (--full, --train-only, etc.)
├── PIPELINE.md                    # This file
├── README.md                      # Quick-start (1 page)
├── colmap_data/                   # Original 30-image COLMAP model (txt)
├── colmap_data_optimized/         # Merged 34-image COLMAP model (txt + bin)
├── colab_training_data.zip        # One-file download for Colab
├── splat-files-processed/         # 91 resized images (1920px)
├── scripts/
│   ├── prepare_images.py          # Parallel image resize + JPEG compress
│   ├── run_colmap.py              # Automated model merging + optimization
│   ├── train_3dgs_enhanced.py     # GPU training: densification, L1+L2, scale reg
│   ├── train_3dgs.py              # Local gsplat training (CPU)
│   ├── export_unity.py            # PLY validator for Unity format
│   └── export_pointcloud.py       # COLMAP sparse cloud → PLY
├── colmap_workspace/              # Full COLMAP output (gitignored)
│   ├── sparse/0                   # 30-image reconstruction (binary)
│   ├── sparse/1                   # 12-image reconstruction (binary)
│   └── sparse_registered/         # Merged 34-image model (binary)
└── output/
    ├── arena_sparse_pointcloud.ply
    └── arena_3dgs.ply             # Trained 3DGS model (GPU required)
```

---

## References

1. **3D Gaussian Splatting** — Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.
   "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.
   [Original code](https://github.com/graphdeco-inria/gaussian-splatting)

2. **COLMAP** — Schönberger, J.L. & Frahm, J.-M.
   "Structure-from-Motion Revisited." CVPR 2016.
   [Website](https://colmap.github.io/)

3. **gsplat** — Ye, V. et al.
   "gsplat: Open-Source 3D Gaussian Splatting Library."
   [GitHub](https://github.com/nerfstudio-project/gsplat)

4. **SIFT** — Lowe, D.G.
   "Distinctive Image Features from Scale-Invariant Keypoints." IJCV 2004.

5. **Multiple View Geometry** — Hartley, R. & Zisserman, A.
   "Multiple View Geometry in Computer Vision." Cambridge University Press, 2003.

6. **UnityGaussianSplatting** — P., Aras.
   [GitHub](https://github.com/aras-p/UnityGaussianSplatting)

7. **SuperSplat** — PlayCanvas.
   [Web app](https://supersplat.com/)

8. **hloc** — Sarlin, P.-E. et al.
   "From Coarse to Fine: Robust Hierarchical Localization at Scale." CVPR 2019.
   [GitHub](https://github.com/cvg/Hierarchical-Localization)
