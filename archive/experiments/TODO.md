# Arena 3DGS — TODO

## Critical Path (must do, in order)

- [ ] **Run the chamber-stitch pipeline on Colab**
  - Open `chamber_splat_stitch.ipynb` in Colab with T4 GPU
  - Runtime → Change runtime type → T4 GPU
  - Follow the step-by-step instructions in the notebook
  - Expected: ~1.5-2 hours for full pipeline, 30 min for quick test

- [ ] **Download results**
  - Trained PLY: `arena_unified_30K.ply` (or `arena_unified_quick.ply`)
  - Compressed: `arena_unified_30K_compressed.splat`
  - From Google Drive: `MyDrive/arena_3dgs/`

- [ ] **View locally**
  ```bash
  python3 scripts/decompress_splat.py ~/Downloads/arena_unified_30K_compressed.splat
  ```
  - Controls: drag=orbit, scroll=zoom, R=reset, Q=quit

## High Priority

- [ ] **Run the original COLMAP pipeline for comparison**
  - Open `arena_3dgs_colab.ipynb` in Colab
  - Runs the original end-to-end pipeline (no chamber decomposition)
  - Compare registration rate vs the per-chamber approach

## Medium Priority

- [ ] **Bump .splat format to v2**
  - Add color format byte to header (eliminate byte-count heuristic)
  - Store per-file scale range in header (fix approximate dequantization)
  - Update both `compress_splat.py` and `decompress_splat.py`
  - Keep backward compatibility with v1 reader

- [ ] **Test SH clustering presets on real model**
  - `--quality low` (16K SH clusters) on 500K+ gaussians
  - `--quality very_low` (4K SH clusters)
  - Measure compression ratio vs visual quality

- [ ] **Profile k-means performance**
  - Current: numpy mini-batch k-means, chunked centroid processing
  - Evaluate if sklearn's MiniBatchKMeans would be faster
  - Target: < 5 min for 500K gaussians × 45 SH coeffs × 16K centroids

- [ ] **Benchmark unified vs per-chamber vs original approaches**
  - Metrics: registration rate, PSNR, SSIM, training time
  - Compare: (a) original single COLMAP, (b) per-chamber COLMAP + alignment
  - Determine if the chamber decomposition actually improves results

## Low Priority

- [ ] **Benchmark all 5 quality presets**
  - Metrics: PSNR, SSIM, compression ratio, encode/decode time
  - Test on real arena model (500K+ gaussians)

- [ ] **Add SuperSplat-compatible export**
  - Export to SuperSplat's JSON format for web inspection
  - Or add `.ply` export from decompressor with sorted properties

- [ ] **Improve viewer controls**
  - Add WASD movement
  - Add click-to-select gaussian info
  - Add gamma/exposure controls

- [ ] **Write unit tests for compression**
  - Round-trip test: compress → decompress → compare stats
  - Test all quality presets on synthetic data
  - Test edge cases: 1 gaussian, degenerate rotations, zero opacity
