# Session Context — 3D Gaussian Splatting Training

## Goal
Fix 3D Gaussian Splatting training quality, bugs, and memory issues in a gsplat-based custom implementation, with Colab notebook integration.

## Repo
`github.com/kaarthik-balakrishnan/arena-3dgs`  
Branch: `main`

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train_3dgs_enhanced.py` | Core training script (densification, export_ply, checkpoints, all bug fixes) |
| `scripts/colab_pipeline.py` | Colab wrapper: install deps, COLMAP pipeline, `train_3dgs()` constructs Namespace → calls `train()` |
| `scripts/visualize_coverage.py` | Standalone coverage visualization (project Gaussians into camera views) |
| `arena_3dgs_colab.ipynb` | Colab notebook — Quick Test Cell 7A, Full Training Cell 7B, Coverage Cell 8D |
| `docs/bugs-fixed.md` | Catalog of 7 bugs fixed |
| `AGENTS.md` | This file — session progress for resuming in a new session |

## Architecture

### Training data flow
1. COLMAP sparse model → `cameras.txt`, `images.txt`, `points3D.txt` (text format)
2. Images loaded in parallel, downscaled to `max_res` (default 800 on longest side)
3. Camera intrinsics (fx, cx, cy) scaled by same factor as image
4. Gaussians initialized from COLMAP 3D points → `nn.Parameter` tensors
5. `gsplat.rasterization` renders one random view per iteration
6. Densification (clone/split/prune) every 100 iters, opacity reset every 3000 iters
7. SH degree increases every `sh_degree_interval` iters (1000 default)
8. Checkpoints: init (iter 0), good (first loss < 0.1), final (iter N)
9. `export_ply()` writes PLY with all 48 SH columns (zero-padded if needed)

### How params are passed
- **CLI:** `train_3dgs_enhanced.py` uses `argparse` directly
- **Colab:** `colab_pipeline.train_3dgs()` builds an `argparse.Namespace` programmatically
- **Rule:** New CLI params must use `getattr(args, name, default)` so partial Namespace objects don't crash. The colab_pipeline must explicitly list every param in its Namespace construction.

## Current Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_res` | 800 | Max image side in px (downscaled before training) |
| `iterations` | 30000 | Total training iterations |
| `densify_from_iter` | 500 | Start densification |
| `densify_until_iter` | 15000 | Stop densification (paper: 15000) |
| `densify_interval` | 100 | Densify every N iterations |
| `densify_grad_threshold` | 0.0002 | Gradient norm threshold for clone/split |
| `prune_opacity_threshold` | 0.005 | Remove Gaussians with sigmoid(opacity) below this (paper: 0.005) |
| `opacity_reset_interval` | 3000 | Reset opacities every N iters (paper: 3000) |
| `max_gaussians` | 0 (unlimited) | Hard cap (set 250000–350000 for T4 OOM safety) |
| `force_split_scale` | 0.02 × scene_extent | Force-split Gaussians exceeding this size |
| `sh_degree` | 3 | Max SH degree (3 = full 48-coeff view-dependent color) |
| `sh_degree_interval` | 1000 | Iterations between SH degree increments |
| `random_background` | False | Random BG color per iter (regularization) |
| `position_lr_init` | 0.00016 | Initial position learning rate |
| `position_lr_final` | 0.0000016 | Final position learning rate |
| `position_lr_delay_mult` | 0.01 | Position LR delay multiplier (ramp up from 1% over 500 iters) |
| `position_lr_delay_steps` | 500 | Iterations for position LR warmup |
| `feature_lr` | 0.0025 | SH feature learning rate |
| `opacity_lr` | 0.05 | Opacity learning rate (paper: 0.05) |
| `scaling_lr` | 0.005 | Scale learning rate |
| `rotation_lr` | 0.001 | Rotation learning rate |
| `lambda_dssim` | 0.2 | SSIM loss weight (1-λ) * L1 + λ * (1-SSIM) |
| `percent_dense` | 0.01 | Size threshold as fraction of scene extent |
| `max_init_points` | 50000 | Cap for COLMAP init points |

## Bugs Found & Fixed (7 total)

1. **SH colors frozen** — `colors_sh_dc`/`colors_sh_rest` not wrapped in `nn.Parameter`
2. **UnboundLocalError** — `active_sh_degree` used before assignment in checkpoint
3. **SH interval too long** — `TEST_SH_DEGREE_INTERVAL=5000` → quick test never uses SH
4. **PLY column mismatch** — `export_ply()` writes 48 columns always (zero-padded if lower SH)
5. **Optimizer state lost** — Adam momentum not transferred through densification
6. **Stale script on Colab** — deps marker skipped script download on re-run
7. **Coverage path hardcoded** — `/content/coverage_views` broken locally

See `docs/bugs-fixed.md` for details.

## Training Quality Issues Fixed (2026-05-27)

After analyzing `BrushTest_3dgs.ply` (362K Gaussians, 85.8 MB) against the official implementation and supersplat.ai examples, 8 root causes for quality degradation were identified and fixed:

### Critical fixes applied to `train_3dgs_enhanced.py` and `colab_pipeline.py`:

| # | Issue | Old | New | Official |
|---|-------|-----|-----|---------|
| 1 | `densify_until_iter` | 10000 | 15000 | 15000 |
| 2 | `prune_opacity_threshold` | 0.01 | 0.005 | 0.005 |
| 3 | `opacity_lr` | 0.025 | 0.05 | 0.05 |
| 4 | `position_lr_delay_mult` | missing (no warmup) | 0.01 with 500 step sin-ramp | 0.01 |
| 5 | Split displacement | both children at same position | Sampled along orientation from N(0, scale) | Same |
| 6 | Scene extent source | Gaussian positions | Camera positions ×1.1 | Camera positions ×1.1 |

### Medium priority (not yet applied):
- Switch gradient tracking from world-space to screen-space (viewspace) — gsplat supports `info["means2d"].absgrad` with `absgrad=True`
- Position LR delay steps tracking after densification rebuild (minor)

See `docs/diagnosis.md` for the full analysis with PLY metrics, root cause details, and the pathology (balloon Gaussians with scale up to 1841 in a 50-unit scene, 18% dead Gaussians with opacity < 0.01).

## Latest Training Run Results

**Config:** 30K iters, max_gaussians=250000, max_res=800, densify_until_iter=10000, prune_opacity_threshold=0.01

```
Gaussians: hit cap at ~254,456 (bouncing off 250K limit)
Loss: range 0.05–0.73, final 0.73 (not converged)
```

**Analysis from BrushTest PLY (362K Gaussians, no cap):** 18% of Gaussians have opacity < 0.01 (dead), 48 Gaussians have scale > 100 (max 1841 in a 50-unit scene). The model wastes capacity on balloon Gaussians and dead splats.

### Root causes fixed (2026-05-27)
1. **`densify_until_iter`** 10000 → 15000 — densification ran only 10K of 30K iters
2. **`prune_opacity_threshold`** 0.01 → 0.005 — was killing useful Gaussians
3. **`opacity_lr`** 0.025 → 0.05 — opacities couldn't recover after reset
4. **Position LR delay** — missing warmup caused early overshoot
5. **Split displacement** — children spawned at same spot, now displaced along orientation
6. **Scene extent** — was from Gaussian positions (inflated by outliers), now from camera positions

## Key Implementation Details

### `densification()` — clone/split/prune logic
- **Clone:** high gradient + small scale → make a copy with noise
- **Split:** high gradient + large scale → replace with 2 smaller copies (scale / 1.6), displaced along parent orientation using N(0, scale) sampling
- **Force-split:** any Gaussian > `force_split_scale * scene_extent` regardless of gradient
- **Prune:** remove low opacity (< `prune_opacity_threshold`) or too large (> 0.1 × scene_extent) or too large in screen space (> `max_screen_size` pixels)
- Returns `old_idx` mapping new→old for optimizer state transfer

### `export_ply()` — PLY format
Always writes:
- `x, y, z` (f32)
- `nx, ny, nz` (f32, always 0)
- `f_dc_0..2` (f32, 3 SH DC coefficients)
- `f_rest_0..44` (f32, 45 SH rest coefficients, zero-padded if sh_degree < 3)
- `opacity` (f32)
- `scale_0..2` (f32)
- `rot_0..3` (f32)

Total: 48 columns, 248 bytes per Gaussian. PLY size ≈ N × 248 B + header overhead.

### Optimizer state transfer
After densification changes Gaussian count:
1. Save `optimizer.state_dict()` (contains `exp_avg`, `exp_avg_sq` for each param)
2. Rebuild optimizer with new tensor shapes
3. For each param group, map old Adam states via `old_idx`: survivors copy their old momentum, new entries (from clone/split) get zero momentum

### Colab script download
`install_dependencies()` in `colab_pipeline.py`:
- Downloads `train_3dgs_enhanced.py` and `visualize_coverage.py` from GitHub EVERY run
- Downloads `colab_pipeline.py` itself via Cell 2 (separate `urllib.request.urlretrieve`)
- First-time CUDA extension compilation for gsplat takes ~800s on Colab T4

## Next Steps

### Immediate
- [ ] **Raise `max_gaussians`** to 300000 or 350000 and re-run 30K training
- [ ] **Compare loss curves** — does the model stay below 0.1 consistently?

### If quality still poor
- [ ] Run coverage visualization on init PLY — which views have <10% coverage?
- [ ] Try `random_background=True` (regularization against dark-region cheating)
- [ ] Increase `position_lr_init` / `position_lr_final` (Gaussians may not move enough)

### If OOM persists
- [ ] Lower `max_res` further (600 or even 500)
- [ ] Reduce `max_gaussians` further
- [ ] Profile with `torch.cuda.memory_summary()` to find the specific bottleneck

### If training too slow
- [ ] Reduce `densify_interval` from 100 to 200 (less frequent densification)
- [ ] Reduce `n_views` (skip every other image for a smaller scene)

## Critical Notes for Resuming

1. **Training script doesn't converge at low Gaussian counts** — the cap of 250K was hit by iter ~13000 of 30000, leaving 17000 iterations with no ability to grow. Watch for `gaussians=N` hitting exactly the `max_gaussians` value.
2. **SH degree 3** is the most memory-efficient path to quality. Degree 2 saves parameter memory but not rasterizer memory.
3. **`getattr(args, name, default)`** is required for any new CLI param used in `train()`. The colab Namespace must include the param explicitly.
4. **Coverage visualization** requires the init PLY (`arena_3dgs_init.ply`) saved at iteration 0.
5. **Colab runtime restarts** wipe all installed packages. Only Drive files persist. The `.deps_installed` marker in Drive skips the heavy apt-get/pip install on re-run, but the training scripts are re-downloaded every time.
6. **macOS local testing**: Use `pip install plyfile numpy opencv-python-headless tqdm scipy` (no torch needed for coverage viz). The coverage script's `--output-dir` defaults to `./coverage_views`.

## Running Locally (macOS)

```bash
# Visualize coverage
python scripts/visualize_coverage.py \
  --sparse-dir ~/path/to/sparse/0 \
  --ply ~/path/to/arena_3dgs_init.ply \
  --images-dir ~/path/to/images

# Training requires CUDA — only works on Colab or Linux with NVIDIA GPU
```
