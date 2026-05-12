import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

SPARSE_TXT = Path("colmap_workspace/sparse/0/txt")

def read_points3d_text(path):
    points, colors = [], []
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for line in lines:
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        points.append([x, y, z])
        colors.append([r, g, b])
    return np.array(points), np.array(colors)

print("Reading COLMAP sparse point cloud...")
points, colors = read_points3d_text(SPARSE_TXT / "points3D.txt")
print(f"Points: {len(points)}")

vertex = np.zeros(len(points), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
])
vertex['x'] = points[:, 0]
vertex['y'] = points[:, 1]
vertex['z'] = points[:, 2]
vertex['red'] = colors[:, 0]
vertex['green'] = colors[:, 1]
vertex['blue'] = colors[:, 2]

ply_path = Path("output/arena_sparse_pointcloud.ply")
ply_path.parent.mkdir(exist_ok=True)
PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(ply_path))
print(f"Saved: {ply_path} ({ply_path.stat().st_size / 1024 / 1024:.1f} MB)")
