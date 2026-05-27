# Bugs Fixed During 3DGS Training Refactor

## 1. PLY Export Crashes When `sh_degree < 3`

**File:** `scripts/train_3dgs_enhanced.py` — `export_ply()` and all three call sites

**Symptom:** `IndexError: index 9 is out of bounds for axis 1 with size 9` at end of training.

**Root cause:** `export_ply()` hardcodes 45 SH rest coefficients (`for i in range(45)`) corresponding to SH degree 3. When training with `sh_degree=1` (or 2), `colors_sh` has fewer columns — e.g. at degree 1 it has 3 DC + 9 rest = 12 columns. The loop tries `rest[:, 9]` through `rest[:, 44]` which don't exist.

**Diagnosis:** Read `export_ply` at line ~643, saw `dtype_full += [(f'f_rest_{i}', 'f4') for i in range(45)]` and `for i in range(45): elements[f'f_rest_{i}'] = rest[:, i]`. Traced `colors_sh` shape back through the training loop — it's `(N, 3 + n_sh_rest*3)` where `n_sh_rest = (max_sh_degree+1)^2 - 1`.

**Fix:** At all three export sites (init, good-model, and final), pad `colors_sh_flat` to exactly 48 columns (3 DC + 45 rest) before passing to `export_ply`:
```python
n_sh_full = 48
if colors_sh_flat.shape[-1] < n_sh_full:
    pad = torch.zeros(len(colors_sh_flat), n_sh_full - colors_sh_flat.shape[-1],
                      device=colors_sh_flat.device)
    colors_sh_flat = torch.cat([colors_sh_flat, pad], dim=-1)
```

---

## 2. SH Degree Never Activates During Quick Test

**File:** `arena_3dgs_colab.ipynb` — Quick Test cell

**Symptom:** Training for 3000 iterations, SH degree stays at 0. Loss bounces wildly (0.169–0.707) with view-dependent scenes.

**Root cause:** `TEST_SH_DEGREE_INTERVAL = 5000` but `TEST_ITERS = 3000`. SH progression fires when `(it+1) % 5000 == 0` — never reached. The model trains only the DC (Lambertian) term for all 3000 iterations; 3 SH rest bands are allocated but always zeroed in the forward pass.

**Diagnosis:** Observed SH degree never increased in training logs. Checked notebook — interval (5000) > max iterations (3000).

**Fix:** Changed `TEST_SH_DEGREE_INTERVAL = 500`. SH degree now goes 0→1 at iter 499, leaving 2500 iterations to train view-dependent colors.

---

## 3. `UnboundLocalError: active_sh_degree` at Init Checkpoint

**File:** `scripts/train_3dgs_enhanced.py` — init checkpoint dict at line ~452

**Symptom:** `cannot access local variable 'active_sh_degree' where it is not associated with a value`

**Root cause:** The init checkpoint save block was inserted between `optimizer = torch.optim.Adam(...)` and `active_sh_degree = 0`. The checkpoint dict referenced `active_sh_degree` before it was assigned.

**Diagnosis:** The traceback pointed to line 452 in `train()`. Reading the file confirmed `active_sh_degree = 0` was still at its original position (line 479) after the new checkpoint code.

**Fix:** Moved `active_sh_degree = 0` to line 423, adjacent to `max_sh_degree = args.sh_degree`:
```python
max_sh_degree = args.sh_degree
active_sh_degree = 0
```

---

## 4. SH Colors Not Wrapped in `nn.Parameter` at Initialization

**File:** `scripts/train_3dgs_enhanced.py` — lines 425–426

**Symptom:** SH color coefficients are never updated by the optimizer before the first densification (iter 500). The model trains geometry (means, scales, quats, opacities) but the colors stay frozen at their COLMAP-initialized values for 500 iterations.

**Root cause:** `means`, `quats`, `scales`, and `opacities` are wrapped in `torch.nn.Parameter()` (lines 418–421), giving them `requires_grad=True`. But `colors_sh_dc` and `colors_sh_rest` are plain tensors — `torch.zeros()` and arithmetic return tensors with `requires_grad=False` by default. The optimizer stores them but `loss.backward()` only flows through tensors with `requires_grad=True`, and `optimizer.step()` skips params where `p.grad is None`.

After the first densification, they become Parameters (lines 624–625), so the bug only affects iterations 0–499.

**Diagnosis:** Line-by-line trace of parameter initialization showed `colors_sh_dc` and `colors_sh_rest` missing `nn.Parameter` wrapping while all other trainable parameters had it.

**Fix:** Wrapped both in `nn.Parameter`:
```python
colors_sh_dc = torch.nn.Parameter(((colors_init - 0.5) / C0).unsqueeze(1))
colors_sh_rest = torch.nn.Parameter(torch.zeros(n_points, n_sh_rest, 3, dtype=torch.float32, device=device))
```

---

## 5. Accidental Deletion of Critical Variables During Edit

**File:** `scripts/train_3dgs_enhanced.py` — init checkpoint insertion

**Symptom:** After inserting the init checkpoint code, `get_xyz_lr()`, `active_sh_degree`, `imgs_gt`, `n_iterations`, `densify_from_iter`, `densify_until_iter`, and `densify_interval` were all missing from the function body.

**Root cause:** The first `edit` call replaced the text from `optimizer = torch.optim.Adam(params, eps=1e-15)` to `opacity_reset_interval = args.opacity_reset_interval`. The original code between those two lines contained the position LR scheduler, SH degree init, image tensor, and iteration config variables. These were replaced along with the intended target.

**Diagnosis:** `git diff` showed these lines in the removed (`-`) section of the diff. The training would have crashed with `NameError` on the first call to `get_xyz_lr()` at line ~673.

**Fix:** Restored the missing block by inserting it after the init checkpoint/PLY exports and before `opacity_reset_interval`:
```python
def get_xyz_lr(...):
    ...

imgs_gt = ...
n_iterations = ...
densify_from_iter = ...
densify_until_iter = ...
densify_interval = ...
```

---

## Verified Correct (No Bugs Found)

| Component | Conclusion |
|---|---|
| Optimizer state transfer (Adam momentum preservation) | Correct — uses integer IDs from `state_dict()`, remaps via `old_idx` |
| Densification SH inheritance (clone/split) | Correct — `colors_sh_dc` and `colors_sh_rest` are cloned/split alongside other params |
| SH zeroing in forward pass | Correct — in-place `colors_sh_3d[:, n_active:, :] = 0.0` on non-leaf tensor propagates gradients correctly, CopySlices zeros gradients at masked positions |
| View sampling | Correct — `torch.randint(0, n_views, (1,)).item()` with no fixed seed |
| Loss function | Correct — L1 + (1-SSIM) with λ=0.2 (paper default) |
| `get_xyz_lr` default args | Correct — evaluated at function definition time when `args` and `spatial_lr_scale` are in scope |
