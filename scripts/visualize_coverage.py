#!/usr/bin/env python3
"""
Visualize how well the initial COLMAP point cloud covers each camera view.

Usage (in Colab):
    from scripts.visualize_coverage import check_coverage
    check_coverage(
        sparse_dir="/content/gaussian-splatting/FanTest_3dgs_input/sparse/0",
        ply_path="/content/gaussian-splatting/output/quick_test/arena_3dgs_init.ply",
        images_dir="/content/gaussian-splatting/FanTest_3dgs_input/images",
    )

Output: /content/coverage_views/coverage_*.jpg — one per camera view.
Green dots = COLMAP/initial Gaussians visible in that view.
Views with very few dots have poor initialization coverage.
"""
import os
from pathlib import Path
import numpy as np
import cv2
from plyfile import PlyData


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


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy]
    ])


def check_coverage(sparse_dir, ply_path, images_dir, max_res=800, max_views=None):
    sparse_dir = Path(sparse_dir)
    images_dir = Path(images_dir)

    cams = read_cameras_text(sparse_dir / "cameras.txt")
    imgs_data = read_images_text(sparse_dir / "images.txt")

    ply = PlyData.read(ply_path)
    xs = ply["vertex"]["x"]
    ys = ply["vertex"]["y"]
    zs = ply["vertex"]["z"]
    pts_3d = np.stack([xs, ys, zs], axis=1)
    print(f"Loaded {len(pts_3d)} points from {ply_path}")

    cam = cams[list(cams.keys())[0]]
    img_w0, img_h0 = int(cam["width"]), int(cam["height"])
    p = cam["params"]
    cam_model = cam["model"]
    if cam_model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        fx0, cx0, cy0 = p[0], p[1], p[2]
    elif cam_model == "PINHOLE":
        fx0, cx0, cy0 = p[0], p[2], p[3]
    else:
        fx0, cx0, cy0 = p[0], p[1], p[2]

    scale = min(max_res / max(img_h0, img_w0), 1.0)
    img_w, img_h = round(img_w0 * scale), round(img_h0 * scale)
    fx, cx, cy = fx0 * scale, cx0 * scale, cy0 * scale
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])

    sorted_ids = sorted(imgs_data.keys(), key=lambda x: imgs_data[x]["name"])
    if max_views:
        sorted_ids = sorted_ids[:max_views]

    out_dir = Path("/content/coverage_views")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in sorted_ids:
        info = imgs_data[img_id]
        name = info["name"]
        R = qvec2rotmat(info["qvec"])
        t = info["tvec"]

        pts_cam = (R @ pts_3d.T + t.reshape(3, 1)).T
        in_front = pts_cam[:, 2] > 0.01
        pts_2d = (K @ pts_cam.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
        in_view = (
            in_front
            & (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w)
            & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
        )
        visible = in_front & in_view
        n_visible = visible.sum()
        pct = 100.0 * n_visible / len(pts_3d)

        img_path = images_dir / name
        if not img_path.exists():
            for alt in [images_dir, Path(str(images_dir).replace("_input", ""))]:
                img_path = alt / name
                if img_path.exists():
                    break
        if not img_path.exists():
            print(f"  [{img_id:3d}] {name:30s}  image not found")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [{img_id:3d}] {name:30s}  failed to load")
            continue
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (img_w, img_h))

        overlay = img_bgr.copy()
        pts_proj = pts_2d[visible].astype(int)
        for u, v in pts_proj:
            cv2.circle(overlay, (u, v), 2, (0, 255, 0), -1)
        blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

        cv2.putText(blended, f"View {img_id}: {n_visible}/{len(pts_3d)} pts ({pct:.0f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_path = out_dir / f"coverage_{img_id:03d}.jpg"
        cv2.imwrite(str(out_path), blended)
        print(f"  [{img_id:3d}] {name:30s}  {n_visible:4d}/{len(pts_3d):4d} pts ({pct:3.0f}%)  → {out_path}")

    print(f"\nDone! {len(sorted_ids)} views → {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize COLMAP point coverage per camera view")
    parser.add_argument("--sparse-dir", required=True, help="Path to COLMAP sparse/0 dir (cameras.txt, images.txt)")
    parser.add_argument("--ply", required=True, help="Path to init PLY file (arena_3dgs_init.ply)")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--max-res", type=int, default=800, help="Downscale images to this max side length")
    parser.add_argument("--max-views", type=int, default=None, help="Only process first N views")
    args = parser.parse_args()
    check_coverage(
        sparse_dir=args.sparse_dir,
        ply_path=args.ply,
        images_dir=args.images_dir,
        max_res=args.max_res,
        max_views=args.max_views,
    )
