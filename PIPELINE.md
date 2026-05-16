# Arena 3DGS Pipeline

## Overview

Reconstruct a 10–15m arena from 91 photos using COLMAP + 3D Gaussian Splatting, then compress the model for lightweight local viewing on a CPU-only Mac.

```
Photos → COLMAP SfM → 3DGS Training (GPU) → PLY → Compress → .splat → Decompress & View
```

---

## Repo Structure

```
.
├── arena_3dgs_colab.ipynb      # Colab notebook (GPU training + export + compress)
├── run_pipeline.py              # Local master pipeline (COLMAP + training + compress)
├── scripts/
│   ├── compress_splat.py        # Post-training compression (numpy-only)
│   ├── decompress_splat.py      # Decompression viewer + PLY export (pyglet)
│   ├── run_colmap.py            # COLMAP optimization
│   ├── train_3dgs.py            # Original training wrapper
│   ├── train_3dgs_enhanced.py   # Enhanced training
│   ├── export_unity.py          # PLY validation/export for Unity
│   ├── export_pointcloud.py     # Point cloud export
│   └── prepare_images.py        # Image preparation
├── colmap_data/                 # Pre-computed COLMAP (30-image model)
├── colmap_data_optimized/       # Pre-computed COLMAP (34-image merged model)
├── colmap_workspace/            # COLMAP working directory (local runs)
├── output/                      # COLMAP outputs + training results
├── splat-files-processed/       # Input photos (91 images, 1920px)
├── colab_training_data.zip      # Packaged input for Colab
├── PIPELINE.md                  # This file
└── README.md                    # Project overview
```

---

## Workflow

### Step 1: Train on Colab (GPU required)

1. Open `arena_3dgs_colab.ipynb` in Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells sequentially from Cell 1

**Cell flow:**
- **Cell 1:** Mount Drive → choose **Continue** or **Start Fresh**
- **Cell 2:** Install COLMAP, PyTorch, clone 3DGS repo, build CUDA extensions
- **Cell 3:** Download 91 arena photos from GitHub
- **Cell 4A–4D:** COLMAP SfM (or skip to Step 5 for pre-computed data)
- **Cell 5:** Download pre-computed COLMAP data (alternative to Step 4)
- **Cell 6:** Convert to 3DGS format (SIMPLE_RADIAL → PINHOLE)
- **Cell 7A:** Quick test (3K iters, ~7 min)
- **Cell 7B–7D:** Full training (10K → 20K → 30K, ~30 min total)
- **Cell 7E:** Optional high quality (30K → 50K, ~20 min)
- **Cell 8A:** Export final PLY, download to computer
- **Cell 8B:** Validate PLY format, viewing options
- **Cell 8C:** Compress model → download .splat for local viewer

**Session management:** Progress is tracked in `MyDrive/arena_3dgs/session_state.json`. After a disconnect, run Cell 1 → choose **Continue** → run remaining cells.

### Step 2: View locally (CPU, no GPU)

```bash
# Option A: View compressed .splat directly
python3 scripts/decompress_splat.py path/to/model_compressed.splat
# Controls: drag=orbit, scroll=zoom, R=reset, Q=quit

# Option B: Export to PLY
python3 scripts/decompress_splat.py path/to/model_compressed.splat --export

# Option C: Stats only
python3 scripts/decompress_splat.py path/to/model_compressed.splat --stats
```

### Step 3: Compress existing PLY locally

```bash
python3 scripts/compress_splat.py path/to/model.ply --quality medium
# Presets: very_high, high, medium, low, very_low
```

---

## Compression Format

The `.splat` binary format implements SH clustering + attribute quantization (Aras' blog):

| Quality | SH mode | Pos bits | Scale bits | Rot bits | Ratio |
|---------|---------|----------|------------|----------|-------|
| very_high | norm11 | 16 | 11 | 10 | ~2× |
| high | norm11 | 16 | 11 | 10 | ~2–3× |
| medium | norm565 | 11 | 11 | 10 | ~5× |
| low | cluster 16K | 11 | 11 | 10 | ~15× |
| very_low | cluster 4K | 11 | 11 | 10 | ~18× |

---

## Recent Changes (git log)

| Commit | Date | Description |
|--------|------|-------------|
| `f1c4875` | Now | PyTorch 2.6 compat patch + mark_step only on success |
| `b5a9b97` | Now | Fixed COLMAP-to-training directory gap |
| `c038114` | Now | Cell 2: verify colmap binary before trusting session |
| `6e44e23` | Now | Session management, fresh/continue choice |
| `563bec2` | Now | Added missing checkpoint_exists(), colmap guards |
| `3d503c6` | Now | Submodule init for stale clones |
| `691bbbd` | Now | Clone with --recursive for submodules |
| `9efd14b` | Now | **Added compression pipeline** (compress + decompress) |
| `c339e64` | Older | Notebook rewrite: checkpoint bug fixes |
| `29f8f73` | Older | Parallel I/O, export bug fixes |

---

## Known Limitations

- **Scale dequantization** (v1 .splat): No per-file scale range stored, decompressed scales normalized to [0,1]
- **Color format inference** (v1 .splat): Heuristic based on remaining bytes (no format byte in header)
- **Local training:** Not possible — requires CUDA/MPS. Colab T4 is the only path.
- **OpenGL viewer:** Requires pyglet 2.x; tested on macOS Intel Iris Plus Graphics 645
