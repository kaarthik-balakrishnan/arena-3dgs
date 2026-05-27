## 6. Stale Training Script on Colab — Marker Guard Skips Download on Re-Run

**File:** `scripts/colab_pipeline.py` — `install_dependencies()`

**Symptom:** Re-running Cell 2 (Install Dependencies) prints "Dependencies already installed (Drive marker found)" and exits. The `train_3dgs_enhanced.py` file on Colab is never updated, so any code changes pushed to GitHub are ignored.

**Root cause:** The function checked for a `.deps_installed` marker file in Google Drive and returned early (line 34–37) if found — **before** the `urllib.request.urlretrieve` call that downloads the training script.

**Diagnosis:** Read `install_dependencies()`, saw the early-return guard at line 34–37 and the download at line 56–59. The guard prevented the download from ever executing on re-run.

**Fix:** Moved the `train_3dgs_enhanced.py` download to run **unconditionally** after the marker check, regardless of whether deps are already installed. The install step uses `if/else` to skip the expensive OS/pip installs, but the script download always executes.

**Lineage:** This bug is why the user kept seeing `UnboundLocalError: active_sh_degree` even after the fix was pushed to `main` — Colab was running the old file cached from the first Cell 2 execution.