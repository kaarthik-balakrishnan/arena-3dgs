#!/usr/bin/env python3
"""
Enhanced 3D Gaussian Splatting training using gsplat.
Features improved densification, more iterations, and better defaults.

Usage:
  python scripts/train_3dgs_enhanced.py --input /path/to/input --iterations 30000

This script auto-installs dependencies (PyTorch, gsplat) on first run if missing.
Designed for GPU (CUDA/MPS) machines.
"""
import os, sys
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).resolve().parent.parent

# ── Dependency setup ──────────────────────────────────────────────
def ensure_deps():
    try:
        import torch
    except ImportError:
        print("Installing PyTorch...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    try:
        import gsplat
    except ImportError:
        print("Installing gsplat...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gsplat"])
    try:
        import plyfile
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])

ensure_deps()

import torch
import torch.nn.functional as F
import cv2
from plyfile import PlyData, PlyElement
from tqdm import trange, tqdm

# ── COLMAP data loading (generator-based for memory efficiency) ────
def read_cameras_text(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#': continue
            parts = line.split()
            cameras[int(parts[0])] = {
                'model': parts[1], 'width': int(parts[2]), 'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]]
            }
    return cameras

def read_images_text(path):
    images = {}
    with open(path) as f:
        pairs = [(line, next(f, '')) for line in f if line.strip() and line[0] != '#']
    for data_line, _ in pairs:
        parts = data_line.strip().split()
        img_id = int(parts[0])
        images[img_id] = {
            'qvec': np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]),
            'tvec': np.array([float(parts[5]), float(parts[6]), float(parts[7])]),
            'camera_id': int(parts[8]), 'name': parts[9]
        }
    return images

def read_points3d_text(path):
    points, colors = [], []
    with open(path) as f:
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

# ── Adaptive density control ───────────────────────────────────────
@torch.no_grad()
def densification(means, scales, opacities, grad_accum, count_accum,
                  densify_grad_threshold=0.00002, densify_size_threshold=0.0001,
                  opacity_reset_interval=3000, prune_opacity_threshold=0.005,
                  iteration=0, max_gaussians=500000):
    if iteration < 500:
        return means, scales, opacities, grad_accum, count_accum

    n = len(means)
    if n >= max_gaussians:
        return means, scales, opacities, grad_accum, count_accum

    grad_avg = grad_accum / count_accum.clamp(min=1)
    grad_norm = grad_avg.norm(dim=-1)

    clone_mask = (grad_norm >= densify_grad_threshold) & (scales.mean(dim=-1) <= densify_size_threshold)
    split_mask = (grad_norm >= densify_grad_threshold) & (scales.mean(dim=-1) > densify_size_threshold)

    n_clone = clone_mask.sum().item()
    n_split = split_mask.sum().item()

    # Build gaussian lists from originals; split replaces its originals
    keep_mask = torch.ones(n, dtype=torch.bool, device=means.device)
    parts_m = [means]
    parts_s = [scales]
    parts_o = [opacities]

    if n_clone > 0:
        clone_m = means[clone_mask]
        clone_s = scales[clone_mask]
        clone_o = opacities[clone_mask]
        noise = torch.randn_like(clone_m) * 0.001 * clone_s.mean(dim=-1, keepdim=True)
        parts_m.append(clone_m + noise)
        parts_s.append(clone_s * 0.5)
        parts_o.append(clone_o)

    if n_split > 0:
        keep_mask[split_mask] = False
        split_m = means[split_mask].repeat(2, 1)
        split_s = (scales[split_mask] / 1.6).repeat(2, 1)
        split_o = opacities[split_mask].repeat(2)
        noise = torch.randn_like(split_m) * 0.001
        parts_m.append(split_m + noise)
        parts_s.append(split_s)
        parts_o.append(split_o)

    # Apply keep_mask to the original chunk (index 0)
    parts_m[0] = means[keep_mask]
    parts_s[0] = scales[keep_mask]
    parts_o[0] = opacities[keep_mask]

    means = torch.cat(parts_m)
    scales = torch.cat(parts_s)
    opacities = torch.cat(parts_o)

    # Prune: remove low-opacity gaussians
    prune_mask = torch.sigmoid(opacities) < prune_opacity_threshold
    means = means[~prune_mask]
    scales = scales[~prune_mask]
    opacities = opacities[~prune_mask]

    # Reset gradient accumulators
    n_final = len(means)
    grad_accum = torch.zeros(n_final, 3, device=means.device)
    count_accum = torch.zeros(n_final, 1, device=means.device)

    # Periodic opacity reset
    if iteration % opacity_reset_interval == 0 and iteration > 0:
        opacities = torch.sigmoid(opacities)
        opacities = torch.clamp(opacities, max=0.01)
        opacities = torch.log(opacities / (1 - opacities + 1e-10))

    return means, scales, opacities, grad_accum, count_accum


# ── SSIM loss (1D Gaussian window, same as original 3DGS) ──────────
def _gaussian_window(window_size, sigma, channel, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window.expand(channel, 1, window_size, window_size)

def ssim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    channel = img1.shape[1]
    window = _gaussian_window(window_size, sigma, channel, img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))

# ── Main training ──────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dir = Path(args.input_dir)
    sparse_dir = input_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = input_dir / "sparse" / "0" / "txt"
    if not sparse_dir.exists():
        # Try colmap_data or sparse_registered
        for p in [BASE / "colmap_data", BASE / "colmap_workspace" / "sparse_registered",
                  BASE / "colmap_workspace" / "optimized_model"]:
            if p.exists():
                sparse_dir = p / "txt" if (p / "txt").exists() else p
                if sparse_dir.exists(): 
                    break

    images_dir = input_dir / "images"
    if not images_dir.exists():
        images_dir = input_dir
    if not images_dir.exists():
        images_dir = BASE / "splat-files-processed"

    print(f"Images: {images_dir}")
    print(f"Sparse: {sparse_dir}")

    # Load COLMAP data
    print("Loading COLMAP data...")
    cams = read_cameras_text(sparse_dir / "cameras.txt")
    imgs_data = read_images_text(sparse_dir / "images.txt")
    points, colors = read_points3d_text(sparse_dir / "points3D.txt")
    print(f"  {len(cams)} cameras, {len(imgs_data)} images, {len(points)} points")

    # Load images (parallel I/O)
    sorted_ids = sorted(imgs_data.keys(), key=lambda x: imgs_data[x]['name'])

    def load_one(img_id):
        name = imgs_data[img_id]['name']
        for base_dir in [images_dir, input_dir / "images"]:
            img_path = base_dir / name
            if img_path.exists():
                break
        else:
            return None
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        return img_id, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load_one, img_id) for img_id in sorted_ids]
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda x: sorted_ids.index(x[0]))
    valid_ids = [r[0] for r in results]
    valid_imgs = [r[1] for r in results]
    n_views = len(valid_ids)
    assert n_views > 0, "No valid images found!"
    print(f"  Loaded {n_views} training views")

    img_h, img_w = valid_imgs[0].shape[:2]
    max_res = getattr(args, 'max_res', 1600)
    scale = min(max_res / max(img_h, img_w), 1.0)
    if scale < 1.0:
        new_w = round(img_w * scale)
        new_h = round(img_h * scale)
        print(f"  Downscaling images: {img_w}x{img_h} → {new_w}x{new_h} (scale={scale:.3f})")
        for i in range(n_views):
            valid_imgs[i] = cv2.resize(valid_imgs[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_w, img_h = new_w, new_h

    cam = cams[list(cams.keys())[0]]
    fx = cam['params'][0] * scale
    cx = cam['params'][1] * scale
    cy = cam['params'][2] * scale

    K = torch.tensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=torch.float32, device=device)

    # Build view matrices
    viewmats = []
    for img_id in valid_ids:
        info = imgs_data[img_id]
        R = qvec2rotmat(info['qvec'])
        t = info['tvec']
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        viewmat = torch.eye(4)
        viewmat[:3] = torch.from_numpy(c2w[:3]).float()
        viewmats.append(viewmat)
    viewmats = torch.stack(viewmats).to(device)
    Ks = K.unsqueeze(0).repeat(n_views, 1, 1)

    # Initialize Gaussians from sparse point cloud
    N = len(points)
    max_init = 50000
    if N > max_init:
        idx = np.random.choice(N, max_init, replace=False)
        points = points[idx]
        colors = colors[idx]

    means = torch.tensor(points, dtype=torch.float32, device=device)
    colors_init = torch.tensor(colors, dtype=torch.float32, device=device)
    n_points = len(means)
    print(f"  Initialized {n_points} Gaussians")

    quats = torch.zeros(n_points, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n_points,), 0.1, dtype=torch.float32, device=device)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=2)
    nn_dist = np.maximum(dist[:, 1], 1e-7)
    scales_init = np.log(np.sqrt(nn_dist))
    scales = torch.tensor(np.tile(scales_init[:, np.newaxis], (1, 3)), dtype=torch.float32, device=device)

    means = torch.nn.Parameter(means)
    quats = torch.nn.Parameter(quats)
    scales = torch.nn.Parameter(scales)
    opacities = torch.nn.Parameter(torch.log(opacities / (1 - opacities + 1e-10)))
    n_sh_rest = 45  # total 48 SH coeffs = 3 DC + 45 rest (degree 3)
    C0 = 0.28209479177387814
    colors_sh_dc = (colors_init - 0.5) / C0
    colors_sh = torch.nn.Parameter(torch.cat([colors_sh_dc, torch.zeros(n_points, n_sh_rest, device=device)], dim=-1))

    params = [
        {'params': [means], 'lr': 0.00016, 'name': 'xyz'},
        {'params': [quats], 'lr': 0.001, 'name': 'rotation'},
        {'params': [scales], 'lr': 0.005, 'name': 'scaling'},
        {'params': [opacities], 'lr': 0.05, 'name': 'opacity'},
        {'params': [colors_sh], 'lr': 0.0025, 'name': 'features'},
    ]
    optimizer = torch.optim.Adam(params)

    def get_xyz_lr(iteration, lr_init=0.00016, lr_final=0.0000016, max_steps=30000):
        t = min(iteration, max_steps) / max_steps
        return lr_init * (lr_final / lr_init) ** t

    imgs_gt = torch.tensor(np.stack(valid_imgs) / 255.0, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    n_iterations = args.iterations
    densify_interval = 100

    # Gradient tracking for densification
    grad_accum = torch.zeros(n_points, 3, device=device)
    count_accum = torch.zeros(n_points, 1, device=device)

    from gsplat import rasterization as gs_rasterization

    print(f"Training: {n_iterations} iterations, {n_views} views")
    print(f"  Logging every {args.log_interval} iters, densify every {densify_interval}")

    pbar = trange(n_iterations, desc="3DGS Training", unit="iter")
    for it in pbar:
        optimizer.zero_grad()

        # Pick one random view per iteration (matching original 3DGS)
        idx = torch.randint(0, n_views, (1,)).item()
        viewmat = viewmats[idx:idx+1]
        K_single = Ks[idx:idx+1]
        gt = imgs_gt[idx:idx+1]

        scales_act = torch.exp(scales)

        # Reshape (N, 48) → (N, 16, 3) for SH degree 3 evaluation
        colors_sh_3d = colors_sh.view(-1, 16, 3)
        renders, alphas, info = gs_rasterization(
            means=means,
            quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-10),
            scales=scales_act,
            opacities=torch.sigmoid(opacities),
            colors=colors_sh_3d,
            viewmats=viewmat,
            Ks=K_single,
            width=img_w,
            height=img_h,
            backgrounds=None,
            sh_degree=3,
        )

        renders = renders.permute(0, 3, 1, 2)  # (1, H, W, 3) → (1, 3, H, W)
        L1 = F.l1_loss(renders, gt)
        ssim_val = ssim(renders, gt)
        loss = (1.0 - 0.2) * L1 + 0.2 * (1.0 - ssim_val)

        loss.backward()

        # Accumulate gradients for densification (every iteration)
        if means.grad is not None:
            with torch.no_grad():
                grad_accum += means.grad.detach() ** 2
                count_accum += 1

        # Update position learning rate before step
        for param_group in optimizer.param_groups:
            if param_group.get("name") == "xyz":
                param_group["lr"] = get_xyz_lr(it)

        optimizer.step()

        # Densification (only up to 15000 iterations, matching original)
        if it > 0 and it % densify_interval == 0 and it < 15000:
            with torch.no_grad():
                new_means, new_scales, new_opacities, grad_accum, count_accum = \
                    densification(means.data, scales_act.data, opacities.data,
                                  grad_accum, count_accum, iteration=it,
                                  max_gaussians=args.max_gaussians)
                
                n_new = len(new_means)
                n_old = len(means)
                if n_new != n_old:
                    n_added = max(0, n_new - n_old)
                    n_trimmed = max(0, n_old - n_new)
                    quats_new = torch.cat([quats.data, quats.data[:1].repeat(n_added, 1)]) if n_added else quats.data[:n_new] if n_trimmed else quats.data
                    colors_sh_new = torch.cat([colors_sh.data, torch.zeros(n_added, colors_sh.shape[1], device=device)]) if n_added else colors_sh.data[:n_new] if n_trimmed else colors_sh.data
                    means = torch.nn.Parameter(new_means)
                    quats = torch.nn.Parameter(quats_new)
                    scales = torch.nn.Parameter(torch.log(new_scales.clamp(min=1e-7)))
                    opacities = torch.nn.Parameter(new_opacities)
                    colors_sh = torch.nn.Parameter(colors_sh_new)
                    
                    params = [
                        {'params': [means], 'lr': 0.00016, 'name': 'xyz'},
                        {'params': [quats], 'lr': 0.001, 'name': 'rotation'},
                        {'params': [scales], 'lr': 0.005, 'name': 'scaling'},
                        {'params': [opacities], 'lr': 0.05, 'name': 'opacity'},
                        {'params': [colors_sh], 'lr': 0.0025, 'name': 'features'},
                    ]
                    optimizer = torch.optim.Adam(params)

                if n_new > 0:
                    pbar.set_postfix({"gaussians": n_new, "loss": f"{loss.item():.6f}"})

        if (it + 1) % args.log_interval == 0:
            print(f"\n  Iter {it+1}: loss={loss.item():.6f}, gaussians={len(means)}")

    print(f"\nTraining complete! Final loss: {loss.item():.6f}")
    print(f"Final Gaussian count: {len(means)}")

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_ply(means.data, quats.data, torch.exp(scales.data),
               torch.sigmoid(opacities.data), colors_sh.data,
               output_dir / "arena_3dgs.ply")

def export_ply(means, quats, scales, opacities, colors_sh, path):
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ]
    dtype_full += [(f'f_dc_{i}', 'f4') for i in range(3)]
    dtype_full += [(f'f_rest_{i}', 'f4') for i in range(45)]
    dtype_full += [('opacity', 'f4')]
    dtype_full += [(f'scale_{i}', 'f4') for i in range(3)]
    dtype_full += [(f'rot_{i}', 'f4') for i in range(4)]

    n = len(means)
    elements = np.zeros(n, dtype=dtype_full)
    elements['x'] = means[:, 0].cpu().numpy()
    elements['y'] = means[:, 1].cpu().numpy()
    elements['z'] = means[:, 2].cpu().numpy()
    elements['f_dc_0'] = colors_sh[:, 0].cpu().numpy()
    elements['f_dc_1'] = colors_sh[:, 1].cpu().numpy()
    elements['f_dc_2'] = colors_sh[:, 2].cpu().numpy()
    rest = colors_sh[:, 3:].cpu().numpy()
    for i in range(45):
        elements[f'f_rest_{i}'] = rest[:, i]
    elements['opacity'] = opacities.cpu().numpy()
    elements['scale_0'] = scales[:, 0].cpu().numpy()
    elements['scale_1'] = scales[:, 1].cpu().numpy()
    elements['scale_2'] = scales[:, 2].cpu().numpy()
    elements['rot_0'] = quats[:, 0].cpu().numpy()
    elements['rot_1'] = quats[:, 1].cpu().numpy()
    elements['rot_2'] = quats[:, 2].cpu().numpy()
    elements['rot_3'] = quats[:, 3].cpu().numpy()

    PlyData([PlyElement.describe(elements, 'vertex')]).write(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Exported: {path} ({len(means)} gaussians, {size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced 3DGS Training")
    parser.add_argument("--input-dir", "-i", default=str(BASE / "gaussian-splatting" / "input"),
                        help="Input directory with images/ and sparse/0/")
    parser.add_argument("--output-dir", "-o", default=str(BASE / "output"),
                        help="Output directory for PLY files")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Number of training iterations (default: 30000)")
    parser.add_argument("--max-gaussians", type=int, default=500000,
                        help="Maximum number of Gaussians (default: 500000)")
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Log every N iterations (default: 1000)")
    parser.add_argument("--max-res", type=int, default=1600,
                        help="Maximum image resolution (longest side, default: 1600)")
    args = parser.parse_args()

    train(args)
