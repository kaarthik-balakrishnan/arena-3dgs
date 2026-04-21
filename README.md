# 3D Arena Reconstruction with Gaussian Splatting

Reconstruct a 3D arena from 93 images using Gaussian Splatting on Google Colab.

## Overview

This project reconstructs a 10-15m arena from photos using:
1. **COLMAP** - Structure-from-Motion to estimate camera poses and sparse 3D points
2. **3D Gaussian Splatting** - Train a differentiable radiance field
3. **Point Cloud** - Extract final 3D point cloud

## Workflow

### Step 1: Preprocess Images (Local Machine)

Resize images to reduce file size for Colab upload:

```bash
python prepare_images.py --input splat-files --output splat-files-processed --width 1920
```

Then zip the processed images:
```bash
cd splat-files-processed && zip -r ../images.zip *
```

### Step 2: Run on Google Colab

1. Open `notebook_colab.ipynb` in Google Colab
2. Run cells in order
3. When prompted, upload the ZIP file containing images
4. After training, download the point cloud (.ply file)

### Step 3: View Results

The output `.ply` file can be viewed in:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.cloudcompare.org/)
- [PyMeshLab](https://github.com/cdmsPyLabs/PyMeshLab)

## Files

| File | Description |
|------|-------------|
| `prepare_images.py` | Resize images for Colab |
| `notebook_colab.ipynb` | Main Colab notebook |
| `splat-files/` | Original 93 images |
| `reconstruct_2d.py` | Alternative 2D reconstruction |

## GPU Requirements

- **Recommended**: Google Colab with T4 or A100 GPU
- **VRAM**: Minimum 12GB recommended
- **Training Time**: ~10-15 minutes for 3000 iterations

## Alternative Approaches

### Using gsplat (modern library)

```bash
pip install gsplat
```

### Using Nerfstudio

```bash
pip install nerfstudio
ns-train gaussian-splatting --data /path/to/images
```

## Troubleshooting

- **COLMAP fails**: Ensure images have sufficient overlap and features
- **Out of memory**: Reduce image resolution or number of images
- **Poor quality**: Increase training iterations

## References

- [3D Gaussian Splatting Paper](https://arxiv.org/abs/2308.04079)
- [Official Repo](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat Library](https://github.com/nerfstudio-project/gsplat)