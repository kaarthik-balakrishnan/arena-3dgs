# colab_pipeline.py — API Reference

All pipeline functions accept a `Session` object (from `scripts.session`) for idempotency and Google Drive checkpointing. Path parameters are keyword-only with Colab defaults; pass custom paths to run outside Colab.

---

## `install_dependencies(session, *, scripts_dir, repo_url)`

| Parameter | Default | Purpose |
|---|---|---|
| `session` | (required) | Session for idempotency |
| `scripts_dir` | `/content/scripts` | Where to download `train_3dgs_enhanced.py` |
| `repo_url` | `https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main` | GitHub raw base URL |

**What it does:**
1. Installs COLMAP, Xvfb, libgl via apt
2. Installs PyTorch 2.1.0 + CUDA 11.8 wheels
3. Installs `plyfile numpy pillow opencv-python-headless tqdm gsplat scipy`
4. Downloads `train_3dgs_enhanced.py` from GitHub to `scripts_dir`
5. Sets up virtual display via Xvfb

**Generates:** System-level COLMAP install, Python packages, `{scripts_dir}/train_3dgs_enhanced.py`, a Drive checkpoint marker.

---

## `download_images(session, *, input_dir, repo, expected_images)`

| Parameter | Default | Purpose |
|---|---|---|
| `session` | (required) | Session for idempotency |
| `input_dir` | `/content/gaussian-splatting/input` | Where to save downloaded images |
| `repo` | `kaarthik-balakrishnan/arena-3dgs` | GitHub repo name |
| `expected_images` | `91` | Stop early if this many exist |

**What it does:** Lists `splat-files-processed/` via GitHub API, downloads all `.jpg/.jpeg/.png` files.

**Generates:** Image files in `input_dir`, sets `num_images` session param.

---

## `download_precomputed_colmap(session, model_choice, *, sparse_dir, repo_url)`

| Parameter | Default | Purpose |
|---|---|---|
| `session` | (required) | Session for idempotency |
| `model_choice` | `"84-image full"` | Key into `COLMAP_MODELS` dict |
| `sparse_dir` | `/content/gaussian-splatting/input/sparse/0` | Output for `cameras.txt`, `images.txt`, `points3D.txt` |
| `repo_url` | `https://raw.githubusercontent.com/...` | Base URL for model files |

**Available models** (from `COLMAP_MODELS`):

| Key | Directory | Images |
|---|---|---|
| `"30-image original"` | `colmap_data` | 30 |
| `"34-image merged"` | `colmap_data_optimized` | 34 |
| `"84-image full"` | `colmap_data_84` | 84 |

**Generates:** `cameras.txt`, `images.txt`, `points3D.txt` in `sparse_dir`. Sets `colmap_download_choice` and `expected_images` session params.

---

## `run_colmap_features(session, *, input_dir, colmap_dir, db_path)`

| Parameter | Default | Purpose |
|---|---|---|
| `session` | (required) | Session for idempotency |
| `input_dir` | `/content/gaussian-splatting/input` | Image directory |
| `colmap_dir` | `/content/gaussian-splatting/sparse` | COLMAP workspace root |
| `db_path` | `/content/gaussian-splatting/sparse/database.db` | SQLite database path |

**What it does:** Runs `colmap feature_extractor` with SIMPLE_RADIAL model, single camera, SiftExtraction GPU=0, max 8192 features, first_octave -1, peak_threshold 0.01.

**Generates:** SQLite database at `db_path` with keypoints tables. Saves DB checkpoint to Drive.

---

## `run_colmap_matching(session, *, db_path)`

| Parameter | Default |
|---|---|
| `session` | (required) |
| `db_path` | `/content/gaussian-splatting/sparse/database.db` |

**What it does:** Runs both `colmap sequential_matcher` (overlap 20) then `colmap exhaustive_matcher`. Saves DB to Drive after each.

**Generates:** Match tables in the SQLite database. Sets `colmap_matching` session step.

---

## `run_colmap_reconstruction(session, *, input_dir, colmap_dir, db_path, sparse_dir)`

| Parameter | Default |
|---|---|
| `session` | (required) |
| `input_dir` | `/content/gaussian-splatting/input` |
| `colmap_dir` | `/content/gaussian-splatting/sparse` |
| `db_path` | `/content/gaussian-splatting/sparse/database.db` |
| `sparse_dir` | `/content/gaussian-splatting/input/sparse/0` |

**What it does:** Runs `colmap mapper` with multiple models, up to 50 models, min triangulation angle 4°, min 15 inliers for init, min 8 for absolute pose, BA iterations 25/50. Finds the best model (most images) and copies it to `sparse_dir` as text files.

**Generates:** COLMAP sparse models in `colmap_dir/{0,1,...}`. Best model's `cameras.txt`, `images.txt`, `points3D.txt` in `sparse_dir`. Saves sparse model checkpoint to Drive.

---

## `run_colmap_merge(session, *, colmap_dir, merged_dir, sparse_dir)`

| Parameter | Default |
|---|---|
| `session` | (required) |
| `colmap_dir` | `/content/gaussian-splatting/sparse` |
| `merged_dir` | `/content/gaussian-splatting/sparse_merged` |
| `sparse_dir` | `/content/gaussian-splatting/input/sparse/0` |

**What it does:** Finds models with ≥5 images, merges up to 3 best models pairwise via `colmap model_merger`. Converts merged result to text and copies to `sparse_dir`.

**Generates:** Merged model at `merged_dir`, text files in `sparse_dir`. Sets `merged_images`, `merged_points3d` session params.

---

## `convert_to_3dgs_format(session, *, sparse_dir, input_dir, images_dir)`

| Parameter | Default |
|---|---|
| `session` | (required) |
| `sparse_dir` | `/content/gaussian-splatting/input/sparse/0` |
| `input_dir` | `/content/gaussian-splatting/input` |
| `images_dir` | `/content/gaussian-splatting/input/images` |

**What it does:**
1. Moves images from `input_dir` into `images_dir/`
2. Converts COLMAP camera model from SIMPLE_RADIAL → PINHOLE (in `cameras.txt`)
3. Removes stale `.bin` files
4. Runs `colmap model_converter --output_type BIN` to rebuild binary files

**Generates:** `cameras.txt` (PINHOLE model), `cameras.bin`, `images.bin`, `points3D.bin` in `sparse_dir`. Sets `training_images` session param.

---

## `train_3dgs(session, iterations, max_gaussians, log_interval, max_res, output_name, *, input_dir, output_base)`

| Parameter | Default | Purpose |
|---|---|---|
| `session` | (required) | |
| `iterations` | `30000` | Training iterations |
| `max_gaussians` | `500000` | Max Gaussian count |
| `log_interval` | `1000` | Steps between logs |
| `max_res` | `1600` | Max image resolution |
| `output_name` | `"arena_3dgs"` | Subdirectory name under `output_base` |
| `input_dir` | `/content/gaussian-splatting/input` | Must contain `images/` and `sparse/0/*.bin` |
| `output_base` | `/content/gaussian-splatting/output` | Parent of `{output_name}/` |

**What it does:** Calls `train()` from `train_3dgs_enhanced.py` with the converted data.

**Generates:** `{output_base}/{output_name}/{output_name}.ply`, checkpoints, logs. Sets session step `training_{N}k`.

---

## `export_pointcloud(session, *, output_dirs, dst)`

| Parameter | Default |
|---|---|
| `session` | (required) |
| `output_dirs` | `["/content/.../arena_3dgs.ply", "/content/.../quick_test.ply"]` |
| `dst` | `/content/arena_3dgs_pointcloud.ply` |

**What it does:** Finds first existing `.ply` from `output_dirs`, copies to `dst`, saves to Drive, and triggers browser download (Colab).

**Generates:** `{dst}` PLY file, Drive checkpoint.

---

## `validate_pointcloud(ply_path)`

| Parameter | Default |
|---|---|
| `ply_path` | `/content/arena_3dgs_pointcloud.ply` |

**What it does:** Reads PLY with `plyfile`, checks for required fields (`x, y, z, f_dc_[0-2], opacity, scale_[0-2], rot_[0-4]`), prints stats.

**Generates:** Console output only (no files).

---

## Helper functions (internal)

`_organize_images(*, input_dir, images_dir)` — Moves images from `input_dir` to `images_dir/`. Called by `convert_to_3dgs_format`.

`_restore_colmap_data(session, *, db_path)` — Restores `database.db` from Drive checkpoint if available.

`_restore_sparse_model(session, *, colmap_dir)` — Restores sparse model `{colmap_dir}/0/` from Drive checkpoint if available.

`_copy_sparse_to_input(best_model, *, colmap_dir, merged_dir, sparse_dir)` — Copies COLMAP output (text format) from the best/merged model into `sparse_dir` for 3DGS consumption.

---

## `COLMAP_MODELS` (dict)

Maps model choice strings to `{dir, expected_imgs, desc}` for pre-computed COLMAP data download.

---

## Related Scripts

| Script | Purpose |
|---|---|
| `scripts/session.py` | `Session` class (Google Drive checkpointing, idempotency) |
| `scripts/train_3dgs_enhanced.py` | Actual 3DGS training loop |
| `scripts/run_colmap.py` | Local COLMAP CLI (`-i <images_dir>`) |
| `scripts/compress_splat.py` | Convert PLY → .splat (standard or compressed) |
| `scripts/export_3dgs_input.py` | Convert raw COLMAP + images → 3DGS input format |
| `scripts/chamber_splat.py` | End-to-end alternative pipeline |
| `scripts/viz_colmap.py` | Visualize COLMAP sparse model |
