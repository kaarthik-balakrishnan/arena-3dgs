# How to Add a New Pre-computed COLMAP Model

## Overview

The pipeline supports selecting from multiple pre-computed COLMAP models via a GUI dropdown. Adding a new model requires three steps: prepare the data, register it in the code, and add it to the dropdown.

---

## Step 1: Prepare the COLMAP Model

Your model directory must be committed to the repo with three text files in **SIMPLE_RADIAL** format:

```
colmap_data_<name>/
  cameras.txt      # One camera, model=SIMPLE_RADIAL
  images.txt       # One line per image + one line per 2D point
  points3D.txt     # One line per 3D point
```

These are standard COLMAP text-export files. To export from a COLMAP binary workspace:

```bash
colmap model_converter \
  --input_path <sparse_dir> \
  --output_path colmap_data_<name> \
  --output_type TXT
```

**Important:** All pre-computed models must use `SIMPLE_RADIAL` camera model because Step 6 converts `SIMPLE_RADIAL → PINHOLE` uniformly. If your source model uses a different camera model, convert it first (or edit the text files manually).

---

## Step 2: Register in `scripts/colab_pipeline.py`

Open `scripts/colab_pipeline.py` and add an entry to the `COLMAP_MODELS` dictionary (around line 382):

```python
COLMAP_MODELS = {
    "30-image original": {
        "dir": "colmap_data",
        "expected_imgs": 30,
        "desc": "Original COLMAP model (30 registered images)",
    },
    "34-image merged": {
        "dir": "colmap_data_optimized",
        "expected_imgs": 34,
        "desc": "Merged/optimized COLMAP model (34 registered images)",
    },
    "84-image full": {
        "dir": "colmap_data_84",
        "expected_imgs": 84,
        "desc": "Full arena model with 84 registered images (recommended)",
    },
    "<your-label>": {                              # ← Add yours
        "dir": "colmap_data_<name>",               # Directory name in repo
        "expected_imgs": <N>,                      # Number of registered images
        "desc": "<Short description>",             # Shown to the user
    },
}
```

Fields:
| Field | Purpose |
|---|---|
| `dir` | Subdirectory in the repo root containing the three `.txt` files |
| `expected_imgs` | Image count (shown in logs; used for downstream validation) |
| `desc` | Human-readable description shown when the user selects this model |

---

## Step 3: Add to the GUI Dropdown

### `arena_3dgs_colab.ipynb`

In Cell 5 (around line 221), append your label to the `@param` list:

```python
MODEL_CHOICE = "84-image full" #@param ["30-image original", "34-image merged", "84-image full", "<your-label>"]
```

### `arena_3dgs_84.ipynb` (if applicable)

If the 84-image notebook also offers model selection, update its `@param` list the same way.

---

## How It Works

When the user selects a model and runs Cell 5, `download_precomputed_colmap(session, model_choice=...)` looks up the entry in `COLMAP_MODELS`, then downloads `cameras.txt`, `images.txt`, and `points3D.txt` from:

```
https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main/<dir>/<file>
```

No additional code changes are needed — the download function is fully generic.

---

## Verifying

1. Commit the new directory, edit `colab_pipeline.py`, and push to GitHub.
2. Open the notebook on Colab.
3. The new option should appear in the Step 5 dropdown.
4. Select it, run the cell, and confirm all three text files are downloaded and the image count matches `expected_imgs`.

---

## Training Troubleshooting

### Loss oscillates / doesn't converge

The notebook has two training cells — a **quick test** (3000 iterations) and **full training** (30000 iterations). 3000 iterations is far too few for 84 diverse views; the 30000-iteration cell is the one that produces a converged model.

At 3000 iterations the loss typically plateaus or rises because:

- **Densification instability** — New Gaussians are added every 100 iterations where gradients are high. With only 3000 iterations, later densification steps (iter 2000+) add Gaussians that never get enough remaining iterations to settle, causing the loss to oscillate upward.
- **Learning rate barely decays** — The position LR scheduler decays over 30K steps. At 3K it is still at ~95% of the initial LR, so the optimizer keeps making large positional updates and never converges.
- **Gaussian count grows unchecked** — Gaussians grow from ~18K → ~63K in 3000 iters with insufficient pruning time.

**Solution:** Run Cell 8 (`30000 iterations`) instead of Cell 7 (`quick_test`). The loss should converge to ~0.05–0.1.

### Loss still poor after 30K iterations

If the full 30K training also converges poorly, the likely causes are:

- **Non-integer downscale factor** — Images are downscaled from 1920×1445 → 1600×1204 (0.833×). The camera intrinsics (fx, fy, cx, cy) from the SIMPLE_RADIAL → PINHOLE conversion are multiplied by this scale, but a non-integer factor can cause sub-pixel misalignment between projected Gaussians and ground truth pixels.
- **Single shared intrinsics** — All 84 images share one camera model from COLMAP. If the source photos had varying focus, cropping, or non-uniform resizing, a single set of intrinsics won't fit every view perfectly.
- **Too few initial points** — The pre-computed COLMAP model provides ~18.5K initial Gaussians. For a scene with 84 views covering a large area, starting with more points (via a denser COLMAP reconstruction) can improve convergence.

---

## Viewing the Result

### .splat file fails to load in SuperSplat

SuperSplat expects a **standard 32-byte-per-Gaussian** format where `filesize % 32 == 0`. The custom compressed `.splat` from `compress_splat.py` has a header + palette that breaks this alignment, causing the error:

```
filesize is not a multiple of 32 bytes
```

**Fix:** Run compression with the `--standard` flag to export a SuperSplat-compatible file:

```bash
python3 scripts/compress_splat.py output/arena_3dgs_pointcloud.ply --standard
```

This writes `arena_3dgs_pointcloud_standard.splat` (32 bytes per Gaussian, no header), which you can drag directly into https://supersplat.com/editor.

The notebook Cell 8C now exports **both** formats automatically:
| File | Format | Viewer |
|------|--------|--------|
| `*_standard.splat` | 32 B/Gaussian, raw | SuperSplat, PlayCanvas |
| `*_compressed.splat` | Custom (header + SH palette) | `scripts/decompress_splat.py` |

### Other viewing options

1. **SuperSplat** (no install) — drag `*_standard.splat` onto https://supersplat.com/editor
2. **Unity** — Clone https://github.com/aras-p/UnityGaussianSplatting, drop the PLY into `Assets/GaussianAssets/`
3. **Local viewer** — `python3 scripts/decompress_splat.py path/to/compressed.splat` (drag to orbit, scroll to zoom, R=reset, Q=quit)
