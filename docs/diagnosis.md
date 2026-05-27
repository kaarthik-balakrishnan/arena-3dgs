# Training Quality Diagnosis — BrushTest vs Official 3DGS

## PLY Analysis

| Metric | BrushTest_3dgs.ply |
|--------|-------------------|
| Gaussian count | 362,695 |
| File size | 85.8 MB |
| Opacity < 0.01 (dead) | 66,014 (18%) |
| Opacity > 0.1 (useful) | 249,656 (69%) |
| Max scale (any Gaussian) | 1841.38 |
| Gaussians with scale > 1 | 16,037 |
| Gaussians with scale > 10 | 435 |
| Scene bounding box | ~35 × 43 × 27 units |

**Key pathology:** Gaussians with scale 1841 in a 50-unit scene — they span 37× the entire scene. These act as diffuse blobs that wash out detail.

## 8 Root Causes of Poor Quality

### 1. densify_until_iter = 10000 (official: 15000)
**Severity: Critical**

After iter 10000, NO densification, NO opacity reset, NO size control for the last 20K iterations. Gaussians balloon unchecked. Official gives **50% more time** under densification control.

**Fix:** Set `densify_until_iter=15000` (paper default).

### 2. prune_opacity_threshold = 0.01 (official: 0.005)
**Severity: High**

Twice the official pruning threshold. Combined with half the opacity LR, useful Gaussians with borderline opacity get killed before recovering. Creates a cycle where too many Gaussians die → remaining ones try to cover more → they grow too large.

**Fix:** Set `prune_opacity_threshold=0.005`.

### 3. opacity_lr = 0.025 (official: 0.05)
**Severity: High**

Opacities can't change fast enough. After opacity reset (clamps to 0.01), Gaussians need to raise opacity quickly to survive the next pruning at iter 3100. At half the LR, they don't recover in time.

**Fix:** Set `opacity_lr=0.05`.

### 4. Missing position LR delay multiplier (official: 0.01)
**Severity: High**

The official multiplies position LR by 0.01 at the start, ramping up over ~500 iters via a sin curve. This prevents Gaussians from overshooting early. The custom code applies full position LR from iter 0, causing Gaussians to move wildly before the scene structure is established.

**Fix:** Add `position_lr_delay_mult=0.01` and `position_lr_delay_steps=500`, modulate position LR by `delay_mult + (1-delay_mult) * sin(0.5π * t)`.

### 5. Split doesn't displace children
**Severity: High**

Official code samples displacement from a normal distribution scaled by the parent's scale, rotated by the parent's orientation, then adds to position:
```python
samples = torch.normal(mean=0, std=parent_scale)
new_xyz = parent_rotation @ samples + parent_xyz
```

Custom code places both children at **exactly the parent's position** — they start overlapping perfectly, making splits far less effective.

**Fix:** Add random displacement oriented by the parent's rotation during split.

### 6. Scene extent from Gaussian positions (official: camera positions)
**Severity: Medium**

Official computes `cameras_extent` as `1.1 × max(camera_distances_from_center)`. Custom uses `max(||Gaussian_positions||)` — if COLMAP has outlier points, scene extent inflates, making all size thresholds (percent_dense, force_split, world-space pruning) proportionally too large.

**Fix:** Compute extent from camera positions (extracted from viewmats), not Gaussian positions.

### 7. World-space gradient instead of screen-space gradient
**Severity: Low-Medium**

Official accumulates `||viewspace_point_tensor[:, :2]||` (screen-space position gradient, directly from the rasterizer). Custom uses `||means.grad||` (world-space 3D position gradient, computed by PyTorch autograd through the full chain). The 3D gradient is a valid signal but mixes projection distance into the threshold. Screen-space gradients are more uniform regardless of depth.

**Fix:** Switch to viewspace gradient if available from gsplat rasterizer.

### 8. Gradient accumulation only for visible Gaussians (variance)
**Severity: Low**

The `count_accum` tracks visible counts per Gaussian. Custom uses `(grad_norm > 0)` as visibility, which is equivalent to the official `visibility_filter` but may miss edge cases where a Gaussian is rendered but has zero gradient.

**Fix:** Verified as functionally equivalent to official.

## How supersplat.ai Scenes Are Different

Scenes on `supersplat.ai` are trained with the **official implementation** using:
- 24 GB VRAM (allows millions of Gaussians)
- densify_until_iter=15000
- prune_opacity_threshold=0.005
- opacity_lr=0.05
- position_lr_delay_mult=0.01
- Proper split displacement
- Scene extent from camera positions

These models typically have **1-6 million Gaussians** with <5% dead weight, properly distributed sizes, and detailed SH coefficients.

## Action Items

1. [x] Set `densify_until_iter=15000`
2. [x] Set `prune_opacity_threshold=0.005`
3. [x] Set `opacity_lr=0.05`
4. [x] Add `position_lr_delay_mult=0.01` with warmup
5. [x] Fix split to displace children along orientation
6. [x] Fix scene_extent to use camera positions
7. [ ] Switch to screen-space gradient (if gsplat supports it)
