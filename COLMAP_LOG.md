# COLMAP Reconstruction Log

## Goal
Reconstruct a 10–15m arena from 91 photos using COLMAP for 3D Gaussian Splatting training.

## Dataset
- **91 images** across **9 chambers** (Chamber1–9), 1920×1445 px, Google Pixel 10 Pro
- Camera model: SIMPLE_RADIAL (f ≈ 2304px, k1 ≈ 0.0065)
- Images numbered by chamber; naming shows overlap: `Chamber1_03_Chamber2.jpg` = taken from Chamber1 looking into Chamber2

## Approach & Results

### 1. Sequential Matching + Incremental SfM
**Script:** `colmap_full_reconstruction.py`

- **Match strategy:** Sequential matching with overlap=15 (adjacent images in alphabetical order = chamber walkthrough)
- **Registration:** **59/91 (64%)** across **4 disconnected models** (28 + 22 + 15 + 9 images)
- **Root cause:** Sequential matching doesn't bridge distant chambers. Chambers are separate rooms with limited visual overlap through doorways.
- **By chamber:** Chamber1=23/26, 2=7/10, 3=3/5, 4=8/9, 5=4/7, 6=7/10, 7=4/8, 8=7/9, 9=0/7

### 2. Per-Chamber Reconstruction + Alignment
**Script:** `colmap_perchamber.py`

- **Strategy:** Reconstruct each chamber independently, include transition images in both adjacent chambers' models, then align via shared camera poses
- **Registration:** Per-chamber models: C1=28/31, C2=13/20, C3=11/12, C4=6/14, C5=10/15, C6=10/18, C7=7/17, C8=7/14, C9=5/10
- **Alignment:** Failed — transition images (showing adjacent chamber) don't register in target chamber's reconstruction because they're taken from a different room with limited overlap
- **Result:** Only **29/91 (31%)** after alignment chain broke at multiple points

### 3. Exhaustive Matching + Incremental SfM
**Script:** `colmap_exhaustive.py`

- **Match strategy:** `exhaustive_matcher` — all 91×90/2 = 4095 image pairs matched (100% verified)
- **Matches:** 505 verified pairs after filtering (every image connects to every other image)
- **Registration:** **82/91 (90%)** — 2 models (Model 0: 76, Model 1: 8)
- **Missed:** 9 mostly outside/topview images — incremental SfM couldn't chain across all weak connections

### 4. Exhaustive Matching + Global SfM (Winner)
**Command:** `colmap exhaustive_matcher` → `colmap global_mapper`

- **Match strategy:** Same exhaustive matching as #3
- **SfM strategy:** Global SfM (rotation averaging + global positioning) instead of incremental
- **Registration:** **91/91 (100%)** — **single unified model with all 9 chambers**
- **Points:** 18,554
- **Why it works:** Global SfM solves all camera poses simultaneously via optimization, handling weak cross-chamber connections that incremental SfM loses

### Format Comparison

| Approach | Images | Points | Models | Key Limitation |
|----------|--------|--------|--------|----------------|
| Sequential + incremental | 59/91 (64%) | 10K | 4 | Sequential misses cross-chamber pairs |
| Per-chamber + alignment | 29/91 (31%) | 13K | 9+ | Transition images don't register cross-chamber |
| Exhaustive + incremental | 82/91 (90%) | 19K | 2 | Incremental can't chain Chamber8→Chamber9 |
| **Exhaustive + global** | **91/91 (100%)** | **19K** | **1** | — |

## Key Files

| File | Description |
|------|-------------|
| `colmap_workspace/sparse_global/0/` | **Final model (binary)** — 91 images, 18,554 points |
| `colmap_workspace/sparse_global/0/txt/` | Same model in COLMAP text format |
| `colmap_workspace/database_exhaustive.db` | Database with features + exhaustive matches |
| `colmap_workspace/final_report.json` | Summary report |
| `colmap_exhaustive.py` | Exhaustive matching + incremental mapper script |
| `colmap_perchamber.py` | Per-chamber reconstruction + alignment script |
| `colmap_full_reconstruction.py` | Original sequential matching script |
| `colmap_merge_models.py` | Model merge attempt (not needed — global mappers 91/91) |

## Lessons Learned

1. **Exhaustive > Sequential** — For datasets with weak connectivity (e.g., rooms connected by doorways), exhaustive matching finds ALL potentially matchable pairs
2. **Global > Incremental SfM** — Global SfM solves all poses simultaneously; it handles weakly connected graphs that incremental SfM can't traverse
3. **Per-chamber alignment is unreliable** — Transition images between chambers don't register well in the target chamber's reconstruction, breaking the alignment chain
4. **Caveat:** Global SfM needs good focal length priors (the warning about <50% cameras with prior focal lengths was benign here because all images share the same intrinsic)

## Next Step

Convert to 3DGS training format:
1. `output/arena_sparse_pointcloud.ply` — Sparse point cloud (XYZ + RGB) for reference
2. Training input directory with PINHOLE camera model + 3DGS-format PLY — ready for Colab notebook
