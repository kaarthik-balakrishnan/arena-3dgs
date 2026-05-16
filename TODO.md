# Arena 3DGS — TODO

## High Priority

- [ ] **Train real arena model on Colab**
  - Open `arena_3dgs_colab.ipynb` in Colab with T4 GPU
  - Run cells 1→8A sequentially (Cell 1 → choose Start Fresh)
  - Expected: ~30 min for 30K iterations, outputs `arena_3dgs_pointcloud.ply`

- [ ] **Download trained PLY to local machine**
  - Colab Cell 8A auto-downloads the PLY
  - Alternatively grab from `MyDrive/arena_3dgs/arena_3dgs_pointcloud.ply`
  - Expected size: ~100–500 MB

- [ ] **Compress PLY locally**
  ```bash
  python3 scripts/compress_splat.py arena_3dgs_pointcloud.ply --quality medium
  ```
  - Output: `arena_3dgs_pointcloud_compressed.splat`
  - Also test `--quality high` and `--quality very_high`

- [ ] **View compressed model locally**
  ```bash
  python3 scripts/decompress_splat.py arena_3dgs_pointcloud_compressed.splat
  ```
  - Controls: drag=orbit, scroll=zoom, R=reset, Q=quit
  - Also test `--export` flag for PLY round-trip

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
  - Evaluate if sklearn's MiniBatchKMeans would be faster (add optional dependency)
  - Target: < 5 min for 500K gaussians × 45 SH coeffs × 16K centroids

## Low Priority

- [ ] **Benchmark all 5 quality presets**
  - Metrics: PSNR, SSIM, compression ratio, encode/decode time
  - Test on real arena model (500K+ gaussians)
  - Generate comparison table

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
