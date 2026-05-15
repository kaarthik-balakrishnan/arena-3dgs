#!/usr/bin/env python3
"""
Export 3DGS PLY for UnityGaussianSplatting and optionally for gsplat.js web viewer.
Validates the PLY format and provides quality info.
"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

BASE = Path(__file__).resolve().parent.parent

def validate_and_report(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    data = vertex.data
    n = len(data)

    print(f"Point cloud: {ply_path.name}")
    print(f"  Gaussians: {n:,}")
    print(f"  Properties: {list(data.dtype.names)}")

    # Check required properties for UnityGaussianSplatting
    required = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    missing = [p for p in required if p not in data.dtype.names]
    if missing:
        print(f"  WARNING: Missing properties for Unity: {missing}")
        return False
    
    has_sh = 'f_rest_0' in data.dtype.names
    print(f"  SH coefficients: {'Yes (full)' if has_sh else 'No (RGB only)'}")

    # Stats
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
    print(f"  Bounds: X[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] "
          f"Y[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] "
          f"Z[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    print(f"  Extent: {xyz.ptp(0)}")

    scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
    print(f"  Scale range: [{scales.min():.6f}, {scales.max():.4f}]")

    opacities = data['opacity']
    print(f"  Opacity range: [{opacities.min():.4f}, {opacities.max():.4f}]")
    print(f"  Visible (>0.01): {(opacities > 0.01).sum():,}")

    # Check for obviously bad values
    bad_scales = (scales <= 0).any() or (scales > 100).any()
    bad_opac = (opacities < 0).any() or (opacities > 1).any()
    bad_rots = np.isnan(data['rot_0']).any()

    if bad_scales: print("  WARNING: Invalid scale values detected")
    if bad_opac: print("  WARNING: Opacity out of [0,1] range")
    if bad_rots: print("  WARNING: NaN rotation values detected")

    size_mb = ply_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    print("\n" + "=" * 50)
    print("  UnityGaussianSplatting Compatible: YES" if not missing else "  Compatible: NO")
    print("=" * 50)
    print("\n  To view in Unity:")
    print(f"    1. Clone https://github.com/aras-p/UnityGaussianSplatting")
    print(f"    2. Drop {ply_path.name} into Assets/GaussianAssets/")
    print(f"    3. Open SampleScene or drag the asset into the scene")
    print("\n  To view in browser (SuperSplat):")
    print(f"    1. Go to https://supersplat.com/")
    print(f"    2. Drag & drop {ply_path.name}")
    print()
    return True


def convert_colmap_sparse_to_ply(colmap_txt_dir, output_path):
    """Convert COLMAP sparse point cloud to a simple RGB PLY for reference."""
    from scripts.export_pointcloud import read_points3d_text
    points, colors = read_points3d_text(Path(colmap_txt_dir) / "points3D.txt")
    
    vertex = np.zeros(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(output_path))
    print(f"Saved sparse point cloud to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export/validate 3DGS for Unity")
    parser.add_argument("ply", nargs="?", default=None, 
                        help="Path to 3DGS PLY file (optional, validates all in output/)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate PLY format for Unity compatibility")
    
    args = parser.parse_args()
    
    if args.ply:
        validate_and_report(Path(args.ply))
    else:
        # Find all PLY files in output directory
        output_dir = BASE / "output"
        ply_files = sorted(output_dir.glob("*.ply"))
        if not ply_files:
            print("No PLY files found in output/. Running sparse export...")
            from scripts.export_pointcloud import read_points3d_text
            for sparse_dir in [BASE / "colmap_workspace" / "optimized_model" / "txt",
                                BASE / "colmap_workspace" / "sparse_registered" / "txt",
                                BASE / "colmap_workspace" / "sparse" / "0" / "txt",
                                BASE / "colmap_data"]:
                if (sparse_dir / "points3D.txt").exists():
                    convert_colmap_sparse_to_ply(sparse_dir, output_dir / "arena_sparse_pointcloud.ply")
                    break
            ply_files = sorted(output_dir.glob("*.ply"))
        
        for pf in ply_files:
            print()
            validate_and_report(pf)
