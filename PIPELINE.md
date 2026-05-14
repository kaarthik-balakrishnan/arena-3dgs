# Pipeline Documentation

## What's Been Done

### 1. Image Acquisition
- **91 photos** of a 10-15m arena captured with a **Google Pixel 10 Pro XL**
- Camera specs: 6.9mm lens (24mm full-frame equiv.), f/1.68 aperture
- Photos span ~9 minutes walking around the arena (timestamps: 09:23:13 – 09:28:54)
- Original resolution: 8160×6144 → resized to **1920×1445** for processing

### 2. Structure-from-Motion (COLMAP)

#### Camera Model
- EXIF analysis revealed the Pixel 10 Pro XL's lens characteristics
- Changed from `PINHOLE` to **`SIMPLE_RADIAL`** (adds radial distortion parameter k1)
- Final intrinsics: `fx=1398, cx=960, cy=722.5, k1=0.0065`
- `SIMPLE_RADIAL` is the minimum sufficient model for this mild distortion

#### Feature Extraction
- **16,384 SIFT features** per image (double the default of 8,192)
- `SiftExtraction.first_octave = -1` (detects features at higher resolution)
- `SiftExtraction.peak_threshold = 0.02` (standard threshold)

#### Matching Strategy
Three approaches were tested to maximize image registrations:

| Method | Pairs | Result |
|--------|-------|--------|
| Sequential (overlap=10) | 510 pairs | Matches consecutive frames along walking path |
| Exhaustive | 3,592 pairs | Matches all image pairs globally |
| Vocab Tree | N/A | 112MB tree downloaded, but COLMAP 4.0.4 CPU-only crashed |

The **combination of sequential + exhaustive** matching was used.

#### Reconstruction
- **30 out of 91 images registered** (33%)
- 15,938 sparse 3D points
- The remaining 61 images could not be registered due to:
  - Arena's repetitive textures (uniform green floor, plain walls)
  - These images see **0 3D points** from the existing model
  - They cover different viewpoints that COLMAP couldn't connect

This is a **known SfM limitation** with large, texture-poor scenes. 30 views is sufficient for 3DGS.

### 3. 3D Gaussian Splatting Pipeline

#### Training Configurations

| Setting | Quick (7A) | Full (7B) |
|---------|-----------|-----------|
| Iterations | 3,000 | 30,000 |
| Final Gaussians | ~50K | ~200K+ |
| Time on T4 | ~7 min | ~30 min |
| Quality | Blocky, test-grade | Detailed, production-grade |

The official 3DGS training uses **adaptive density control**:
- **Clone**: Duplicate Gaussians in under-reconstructed regions
- **Split**: Break large Gaussians into smaller ones
- **Prune**: Remove Gaussians with near-zero opacity
- This grows the model from 16K → 200K+ Gaussians over 30K iterations

#### Alternative: gsplat
- [gsplat](https://github.com/nerfstudio-project/gsplat) provides a **2x faster** CUDA rasterizer
- Same quality as official renderer
- Available as optional install in the Colab notebook

### 4. Repository Structure

```
arena-3dgs/
├── arena_3dgs_colab.ipynb     # Main Colab notebook (run this)
├── PIPELINE.md                 # This file
├── README.md                   # Quick-start guide
├── colmap_data/                # Pre-computed COLMAP output
│   ├── cameras.txt             # SIMPLE_RADIAL model
│   ├── images.txt              # 30 registered image poses
│   └── points3D.txt            # 15,938 sparse points
├── splat-files-processed/      # 91 resized images (1920px)
├── scripts/
│   ├── prepare_images.py       # Resize images to 1920px
│   ├── export_pointcloud.py    # Export COLMAP sparse cloud to PLY
│   └── train_3dgs.py           # Local CPU gsplat training
├── reconstruct_2d.py           # 2D top-down arena projection
├── output/
│   └── arena_sparse_pointcloud.ply  # Sparse COLMAP point cloud
├── colmap_workspace/           # Full COLMAP workspace (gitignored)
└── colab_training_data.zip     # ZIP for manual Colab upload
```

---

## How to Generate Results

### Colab (Recommended)

1. Open [arena_3dgs_colab.ipynb](https://colab.research.google.com/github/kaarthik-balakrishnan/arena-3dgs/blob/main/arena_3dgs_colab.ipynb) in Colab
2. Run cells in order:
   - **Cell 1**: Mount Google Drive (for checkpoint backups)
   - **Cell 2**: Install dependencies (~3 min)
   - **Cell 3B**: Download 91 images from GitHub (~2 min)
   - **Cell 5**: Download pre-computed COLMAP data (optional, skips COLMAP)
   - **Cell 6**: Convert data to 3DGS format
   - **Cell 7B**: Train 30K iterations (~30 min on T4 GPU)
   - **Cell 8**: Export PLY and download

### Local (COLMAP only, no GPU)

```bash
# 1. Resize images (if you have originals)
python3 scripts/prepare_images.py --input splat-files --output splat-files-processed

# 2. Run COLMAP with correct settings
colmap feature_extractor \
    --database_path colmap_workspace/database.db \
    --image_path splat-files-processed \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_num_features 16384

colmap sequential_matcher \
    --database_path colmap_workspace/database.db \
    --SequentialMatching.overlap 10

colmap exhaustive_matcher \
    --database_path colmap_workspace/database.db

colmap mapper \
    --database_path colmap_workspace/database.db \
    --image_path splat-files-processed \
    --output_path colmap_workspace/sparse

# 3. Export sparse point cloud
python3 scripts/export_pointcloud.py

# 4. Train on GPU (requires CUDA):
#    Use the Colab notebook or gsplat
```

### Viewing Results

| Tool | Type | Setup | Walk-through? |
|------|------|-------|---------------|
| [SuperSplat](https://supersplat.com/) | Web | Drag & drop PLY | Yes, interactive |
| [Unity + UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) | Desktop | Clone repo, drop PLY | **Yes, full WASD + mouse-look** |
| [CloudCompare](https://www.cloudcompare.org/) | Desktop | Open PLY | Limited (point cloud only) |
| [gsplat.js](https://github.com/nerfstudio-project/gsplat.js) | Web | Embed in HTML | Programmatic camera control |

---

## Remaining Work

### Priority: Improve Image Registration

| Approach | Effort | Expected Gain | Notes |
|----------|--------|---------------|-------|
| **SuperPoint + SuperGlue** | Medium | **High** | Learned features get far more matches on low-texture scenes. Use [hloc](https://github.com/cvg/Hierarchical-Localization) to replace SIFT. |
| **Manual initial pair** | Low | Medium | Tell COLMAP which two images to start with (images that overlap well) |
| **COLMAP with CUDA** | Low | Medium | GPU-accelerated COLMAP is faster and may find better matches. Runs on Colab T4. |
| **Split into segments** | Medium | Medium | Run COLMAP on 2-3 overlapping segments, then merge models |
| **Train a custom model** | High | Very High | Use image features from a pretrained model (DINOv2, DenseVLAD) for matching |

### Priority: Improve 3DGS Quality

| Approach | Effort | Expected Gain | Notes |
|----------|--------|---------------|-------|
| **Train longer** | Low | High | 30K iterations standard; try 50K or 70K |
| **gsplat rasterizer** | Low | Medium | 2x faster training = more iterations in same time |
| **Higher res images** | Medium | Medium | Try full 8160×6144 originals (needs more VRAM) |
| **Multiple cameras** | High | Medium | If more angles are needed, take more photos with different framing |
| **Regularization** | Medium | High | Add depth regularization or normal smoothness loss to fill holes |

### Future Ideas

- **Web viewer**: Deploy the trained model with [gsplat.js](https://github.com/nerfstudio-project/gsplat.js) for browser-based exploration
- **Unity build**: Create a standalone Unity application with the walk-through experience
- **Measurements**: Add distance/area measurement tools for arena analysis
- **Video rendering**: Render a fly-through video of the reconstructed arena
- **Multi-scene**: Apply the pipeline to other scenes (not just the arena)

---

## Technical Notes

### Why only 30/91 images register?

The COLMAP incremental SfM pipeline works by:
1. Finding a good initial image pair (two images with many matches and good parallax)
2. Triangulating 3D points from that pair
3. Incrementally registering new images that see the existing 3D points
4. The unregistered images see **0 already-triangulated 3D points** because:
   - They cover different physical areas of the arena
   - The repetitive floor/wall textures produce ambiguous matches
   - The camera moved too far between consecutive registered frames

This is why using **learned features (SuperPoint)** would help — they produce more discriminative matches on low-texture surfaces.

### Camera Model Selection

| Model | Params | When to Use |
|-------|--------|-------------|
| PINHOLE | fx, fy, cx, cy | No distortion (professional cameras with long lenses) |
| SIMPLE_RADIAL | fx, cx, cy, k1 | **Mild radial distortion (our case: k1=0.0065)** |
| OPENCV | fx, fy, cx, cy, k1, k2, p1, p2 | Significant distortion (wide-angle, phone cameras) |

SIMPLE_RADIAL was chosen because:
- Pixel 10 Pro XL has mild distortion (k1=0.0065)
- Fewer parameters = more stable estimation with limited images
- OPENCV's 8 params overfit with only 30 cameras
