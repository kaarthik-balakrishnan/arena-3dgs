# Future 3DGS Improvements

These are improvements identified from the original paper/repo that are not yet implemented. Listed in order of estimated impact.

## High Impact

### 1. Use screen-space (2D) gradients for densification
**File:** `scripts/train_3dgs_enhanced.py` ~line 475  
**What:** The original paper accumulates `viewspace_point_tensor.grad[:, :2]` (the gradient of the 2D projected position from the rasterizer) instead of `means.grad` (3D world-space gradient).  
**Why:** Screen-space gradients account for perspective effects — Gaussians farther from the camera or near image edges have differently-scaled gradients, producing a better densification signal.  
**Implementation:** The `gsplat` rasterizer likely returns view-space points in the `info` dict. Use those gradients instead of `means.grad.norm()`.

### 2. Fix scene_extent to use camera centers
**File:** `scripts/train_3dgs_enhanced.py` ~line 380  
**What:** Currently `scene_extent = means.data.norm(dim=-1).max()` (max distance of any Gaussian from origin). The original uses the max distance between camera centers × 1.1.  
**Why:** For scenes where the point cloud is offset from the origin, the current computation inflates `scene_extent`, which miscalibrates the `percent_dense` threshold (clone vs split boundary) and position learning rate scaling.  
**Implementation:** Compute `scene_extent` from the camera positions (available via `viewmats` inverse) instead of point positions.

### 3. Add exposure compensation
**File:** `scripts/train_3dgs_enhanced.py` (new code needed)  
**What:** Learn per-image affine color transforms (3×4 matrix) to handle varying exposure/white balance across input photos.  
**Why:** Real-world captures (especially outdoor/large scenes) often have inconsistent lighting between views. The network otherwise has to use SH coefficients to compensate, wasting representational capacity.  
**Implementation:** Add a learnable `exposure` parameter of shape `(n_views, 3, 4)`, apply the affine transform to rendered colors before loss computation, and add an exposure optimizer. See `scene/gaussian_model.py` and `train.py` in the original repo.

## Medium Impact

### 4. Split noise in local (rotated) frame
**File:** `scripts/train_3dgs_enhanced.py` ~line 198-201  
**What:** The original samples split noise in each Gaussian's local coordinate frame (using the Gaussian's rotation matrix via `build_rotation(quats)`), then rotates to world space. Currently noise is added isotropically in world space.  
**Why:** For elongated/anisotropic Gaussians (e.g., walls, thin structures), world-space isotropic noise places child Gaussians in wrong directions. Local-frame sampling ensures children are displaced along the Gaussian's principal axes.  
**Implementation:** Build rotation matrix from `quats`, sample noise in local space with std = new scale, then `bmm(rots, noise.unsqueeze(-1)).squeeze()` to get world-space offset.

### 5. Add position LR warmup
**File:** `scripts/train_3dgs_enhanced.py` ~line 394-398  
**What:** The original starts position LR at `0.01 × lr_init` with a smooth cosine ramp over the first few thousand iterations. Currently starts at `lr_init` from step 0.  
**Why:** High initial position LR can destabilize early training before Gaussians have settled into reasonable positions.  
**Implementation:** Add `position_lr_delay_mult = 0.01` and apply a sinusoidal delay function: `delay = delay_mult + (1 - delay_mult) × sin(0.5π × clamp(step / delay_steps, 0, 1))`; multiply position LR by `delay`.

## Low Impact / Nice-to-Have

### 6. Add depth regularization (optional)
**File:** `scripts/train_3dgs_enhanced.py` (new loss term)  
**What:** An optional monocular depth prior loss that compares rendered depth to COLMAP depth or a monocular depth estimator.  
**Why:** Improves geometry on untextured regions (e.g., roads, walls) and reduces floaters. Weight decays from 1.0 to 0.01 during training.  
**Implementation:** Render depth from the rasterizer, compute L1 loss against available depth maps, multiply by a decaying weight schedule.

### 7. White background handling for synthetic data
**File:** `scripts/train_3dgs_enhanced.py` ~line 530  
**What:** When training with white background (NeRF synthetic data), reset opacity at the start of densification.  
**Why:** White backgrounds can cause Gaussians to become permanently transparent without this reset.  
**Implementation:** Add `if white_background and it == densify_from_iter: reset_opacity()`.

### 8. Optimizer type: SparseAdam
**What:** The original repo optionally uses a sparse Adam variant (`SparseGaussianAdam`) that exploits the fact that many Gaussian parameters are updated infrequently.  
**Why:** Can provide ~2.7× training speedup with minimal quality loss.  
**Implementation:** Available in the gsplat library or original repo's `utils/sparse_adam.py`.
