# Bugs Fixed

## 1. SH Colors Frozen at Zero — Missing `nn.Parameter`

**Files:** `scripts/train_3dgs_enhanced.py` lines 425–426

**Symptom:** `colors_sh_dc` and `colors_sh_rest` had zero gradients for the first 500 iterations. The optimizer didn't update them.

**Root cause:** Raw tensors assigned to `colors_sh_dc` and `colors_sh_rest` were NOT wrapped in `nn.Parameter`. PyTorch optimizers only track `nn.Parameter` objects. The SH colors stayed at initialization values (zero for rest, mean-shifted for DC).

**Fix:** Wrapped both in `nn.Parameter(...)`.

**Impact:** Without this, the model's view-dependent color (specular, reflections) never trained. This is likely the highest-impact fix.

---

## 2. `active_sh_degree` Used Before Assignment

**File:** `scripts/train_3dgs_enhanced.py` line 452 (old)

**Symptom:** `UnboundLocalError: cannot access local variable 'active_sh_degree' where it is not associated with a value`

**Root cause:** `active_sh_degree = 0` was placed AFTER the init checkpoint block that referenced it. The checkpoint dict at line 452 used the variable before it was defined.

**Fix:** Moved `active_sh_degree = 0` to line 423 (before the checkpoint block, next to `max_sh_degree`).

---

## 3. SH Degree Interval Too Long for Quick Test

**File:** `arena_3dgs_colab.ipynb` — Cell 7A

**Symptom:** Quick test runs 3000 iterations with `TEST_SH_DEGREE_INTERVAL=5000`. SH degree never increases from 0 → 1, so quick test always renders at degree 0 (matte only). User can't tell if SH is working.

**Fix:** Changed `TEST_SH_DEGREE_INTERVAL` from 5000 to 500.

**Note:** `sh_degree_interval=500` means SH degree increments every 500 iters → degree 0→1 at 500, 1→2 at 1000, 2→3 at 1500. By 3000 iters, full degree 3 is active for 1500 iters.

---

## 4. Export PLY Columns Don't Match Viewer Expectation

**File:** `scripts/train_3dgs_enhanced.py` — `export_ply()` at lines 706–739; call sites at 462–471, 566–577, 693–704

**Symptom:** Standard 3DGS viewers (SuperSplat, PlayCanvas, Unity) expect exactly 48 SH columns: `f_dc_0..2` (3) + `f_rest_0..44` (45). At lower SH degrees, the script exported fewer columns, making the PLY unreadable.

**Fix:** `export_ply()` always writes 48 columns (hardcoded `f_rest_0..45`). Before calling, the code pads `colors_sh_flat` to 48 columns with zeros. This padding was added at all three call sites: init checkpoint, good model, and final export.

**Important:** If `--sh-degree` is changed from 3, the export still writes 48 columns with zero-padded rest. Viewers won't display the higher SH bands (they'll be zero), but the PLY will load.

---

## 5. Optimizer State Lost on Densification — Child Gaussians Reset

**File:** `scripts/train_3dgs_enhanced.py` — `densification()` and the optimizer rebuild at lines 614–660

**Symptom:** Every densification step (clone/split/prune) destroyed and recreated the optimizer. Adam momentum/velocity for surviving Gaussians was reset to zero, causing them to re-learn from scratch.

**Root cause:** The old code rebuilt the optimizer with `torch.optim.Adam(params, eps=1e-15)` but never transferred the previous state dict. The `old_idx` tracking existed but wasn't used.

**Fix:** After rebuilding the optimizer, the code iterates through param groups, maps old Adam states (`exp_avg`, `exp_avg_sq`) to new indices via `old_idx`, and copies surviving entries. New Gaussians (from clone/split, `old_idx = -1`) get zero-init Adam state as expected.

**Impact:** Without this fix, every densification step reset momentum for all survivors, effectively restarting their optimization. This is the most likely cause of the "loss never converges" pattern.

---

## 6. Stale Training Script on Colab — Marker Guard Skips Download on Re-Run

**File:** `scripts/colab_pipeline.py` — `install_dependencies()`

**Symptom:** Re-running Cell 2 (Install Dependencies) prints "Dependencies already installed (Drive marker found)" and exits. The `train_3dgs_enhanced.py` file on Colab is never updated, so any code changes pushed to GitHub are ignored.

**Root cause:** The function checked for a `.deps_installed` marker file in Google Drive and returned early (before the `urllib.request.urlretrieve` call that downloads the training script).

**Fix:** Moved the `train_3dgs_enhanced.py` download to run **unconditionally** after the marker check, regardless of whether deps are already installed.

**Lineage:** This bug is why the user kept seeing `UnboundLocalError: active_sh_degree` even after the fix was pushed to `main` — Colab was running the old file cached from the first Cell 2 execution.

---

## 7. Coverage Output Hardcoded to `/content/coverage_views`

**File:** `scripts/visualize_coverage.py` line 96

**Symptom:** Running locally on macOS failed with `OSError: [Errno 30] Read-only file system: '/content'`.

**Fix:** Added `output_dir` parameter to `check_coverage()` (default: `./coverage_views`) and `--output-dir` to CLI. Notebook Cell 8D passes `output_dir="/content/coverage_views"` explicitly so Colab behavior is unchanged.
