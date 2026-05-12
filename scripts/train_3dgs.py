import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import struct
from plyfile import PlyData, PlyElement
import math

BASE = Path(__file__).parent / "scene"
SPARSE = Path(__file__).parent / "colmap_workspace/sparse/0"

def read_cameras_text(path):
    cameras = {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for line in lines:
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        cameras[cam_id] = {'model': model, 'width': w, 'height': h, 'params': params}
    return cameras

def read_images_text(path):
    images = {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        qvec = np.array([qw, qx, qy, qz])
        tvec = np.array([tx, ty, tz])
        images[img_id] = {'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id, 'name': name}
    return images

def read_points3d_text(path):
    points, colors = [], []
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for line in lines:
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        points.append([x, y, z])
        colors.append([r/255.0, g/255.0, b/255.0])
    return np.array(points), np.array(colors)

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy]
    ])

print("Loading COLMAP data...")
cameras = read_cameras_text(SPARSE / "txt/cameras.txt")
images_data = read_images_text(SPARSE / "txt/images.txt")
points, colors = read_points3d_text(SPARSE / "txt/points3D.txt")

print(f"Cameras: {len(cameras)}, Images: {len(images_data)}, Points: {len(points)}")

from gsplat import rasterization

sorted_ids = sorted(images_data.keys(), key=lambda x: images_data[x]['name'])
valid_ids = []
valid_imgs = []

for img_id in sorted_ids:
    name = images_data[img_id]['name']
    full_path = BASE / "input" / name
    if full_path.exists():
        valid_ids.append(img_id)
        img_bgr = cv2.imread(str(full_path))
        valid_imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    else:
        alt_path = BASE / "input" / name.split('/')[-1]
        if alt_path.exists():
            valid_ids.append(img_id)
            img_bgr = cv2.imread(str(alt_path))
            valid_imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

n_views = len(valid_ids)
print(f"Found {n_views} valid images with COLMAP poses")

img_h, img_w = valid_imgs[0].shape[:2]
cam = cameras[list(cameras.keys())[0]]
fx = cam['params'][0]
cx = cam['params'][1]
cy = cam['params'][2]

K = torch.tensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=torch.float32)

viewmats = []
for img_id in valid_ids:
    info = images_data[img_id]
    R = qvec2rotmat(info['qvec'])
    t = info['tvec']
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    viewmat = np.eye(4)
    viewmat[:3] = c2w[:3]
    viewmats.append(torch.from_numpy(viewmat).float())

viewmats = torch.stack(viewmats)
K = K.unsqueeze(0).repeat(n_views, 1, 1)

N = len(points)
if N > 50000:
    idx = np.random.choice(N, 50000, replace=False)
    points = points[idx]
    colors = colors[idx]

means = torch.tensor(points, dtype=torch.float32)
colors_init = torch.tensor(colors, dtype=torch.float32)
n_points = len(means)
print(f"Using {n_points} Gaussian primitives")

quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_points, dtype=torch.float32)
opacities = torch.tensor([0.1] * n_points, dtype=torch.float32)
scales = torch.tensor([[0.001, 0.001, 0.001]] * n_points, dtype=torch.float32)

means = torch.nn.Parameter(means)
quats = torch.nn.Parameter(quats)
scales = torch.nn.Parameter(torch.log(scales))
opacities = torch.nn.Parameter(torch.log(opacities / (1 - opacities + 1e-10)))
colors_sh = torch.nn.Parameter(torch.cat([colors_init, torch.zeros(n_points, 15)], dim=-1))

params = [means, quats, scales, opacities, colors_sh]
optimizer = torch.optim.Adam(params, lr=1e-3)

imgs_gt = torch.tensor(np.stack(valid_imgs) / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)

n_iterations = 1000

device = torch.device("cpu")
print(f"Training on {device} for {n_iterations} iterations")

for iter in tqdm(range(n_iterations)):
    optimizer.zero_grad()
    renders, alphas, info = rasterization(
        means=means,
        quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-10),
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities),
        colors=colors_sh,
        viewmats=viewmats,
        Ks=K,
        width=img_w,
        height=img_h,
    )
    loss = torch.nn.functional.mse_loss(renders, imgs_gt)
    loss.backward()
    optimizer.step()
    if (iter + 1) % 100 == 0:
        print(f"  Iter {iter+1}: loss = {loss.item():.6f}")

print(f"\nTraining done! Final loss: {loss.item():.6f}")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

opacities_np = torch.sigmoid(opacities).detach().numpy().squeeze()
means_np = means.detach().numpy()
colors_np = colors_sh.detach().numpy()[:, :3]
colors_np = np.clip(colors_np, 0, 1)

mask = opacities_np > 0.01
print(f"Exporting {mask.sum()} gaussians with opacity > 0.01")

final_xyz = means_np[mask]
final_rgb = (colors_np[mask] * 255).astype(np.uint8)

vertex = np.zeros(len(final_xyz), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
])
vertex['x'] = final_xyz[:, 0]
vertex['y'] = final_xyz[:, 1]
vertex['z'] = final_xyz[:, 2]
vertex['red'] = final_rgb[:, 0]
vertex['green'] = final_rgb[:, 1]
vertex['blue'] = final_rgb[:, 2]

ply_path = output_dir / "arena_point_cloud.ply"
PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(ply_path))
print(f"Saved point cloud to {ply_path}")
