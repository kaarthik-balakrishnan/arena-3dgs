#!/usr/bin/env python3
"""Visualize COLMAP sparse point cloud + camera poses using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import struct

BASE = Path(__file__).resolve().parent.parent


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
    ])


def read_next_bytes(fid, num_bytes, fmt_str):
    data = fid.read(num_bytes)
    if len(data) < num_bytes:
        return None
    return struct.unpack_from(fmt_str, data)


def read_points3d_binary(path):
    points, colors = [], []
    with open(path, "rb") as fid:
        num = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num):
            pt_id = read_next_bytes(fid, 8, "Q")
            if not pt_id:
                break
            pt = read_next_bytes(fid, 24, "ddd")
            if not pt:
                break
            col = read_next_bytes(fid, 3, "BBB")
            if not col:
                break
            err = read_next_bytes(fid, 8, "d")
            if not err:
                break
            track_len = read_next_bytes(fid, 8, "Q")
            if not track_len:
                break
            # Each track element: image_id (uint32) + point2D_idx (uint32) = 8 bytes
            fid.read(track_len[0] * 8)
            points.append(pt)
            colors.append([c / 255 for c in col])
    return np.array(points), np.array(colors)


def read_points3d_text(path):
    points, colors = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            points.append(list(map(float, parts[1:4])))
            colors.append([int(parts[4]) / 255, int(parts[5]) / 255, int(parts[6]) / 255])
    return np.array(points), np.array(colors)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num):
            cam_id = read_next_bytes(fid, 8, "Q")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            w = read_next_bytes(fid, 8, "Q")[0]
            h = read_next_bytes(fid, 8, "Q")[0]
            model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL"}.get(model_id, f"UNKNOWN_{model_id}")
            num_params = {0: 3, 1: 4, 2: 4, 3: 5}.get(model_id, 0)
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = {"model": model_name, "width": w, "height": h, "params": list(params)}
    return cameras


def read_cameras_text(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cameras[int(parts[0])] = {
                "model": parts[1], "width": int(parts[2]), "height": int(parts[3]),
                "params": list(map(float, parts[4:]))
            }
    return cameras


def read_images_binary(path):
    poses = []
    with open(path, "rb") as fid:
        num = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num):
            img_id = read_next_bytes(fid, 8, "Q")[0]
            qvec = read_next_bytes(fid, 32, "dddd")
            tvec = read_next_bytes(fid, 24, "ddd")
            cam_id = read_next_bytes(fid, 8, "Q")[0]
            name_len = read_next_bytes(fid, 8, "Q")[0]
            name_data = fid.read(name_len)
            name = name_data.decode("utf-8").rstrip("\0")
            num_points = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points * 3 * 8)  # skip point2D data
            R = qvec2rotmat(qvec)
            t = np.array(tvec)
            C = -R.T @ t
            poses.append({"R": R, "t": t, "C": C, "cam_id": cam_id, "name": name, "id": img_id})
    return poses


def read_images_text(path):
    poses = []
    with open(path) as f:
        lines = [l.rstrip() for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].strip().split()
        if len(parts) < 10:
            continue
        qvec = list(map(float, parts[1:5]))
        tvec = list(map(float, parts[5:8]))
        cam_id = int(parts[8])
        name = parts[9]
        R = qvec2rotmat(qvec)
        t = np.array(tvec)
        C = -R.T @ t
        poses.append({"R": R, "t": t, "C": C, "cam_id": cam_id, "name": name})
    return poses


def extract_chamber_color(name):
    """Color-code by chamber."""
    chamber = name.split("_")[0] if name.startswith("Chamber") else "other"
    colors = {
        "Chamber1": "#e74c3c", "Chamber2": "#3498db", "Chamber3": "#2ecc71",
        "Chamber4": "#f39c12", "Chamber5": "#9b59b6", "Chamber6": "#1abc9c",
        "Chamber7": "#e67e22", "Chamber8": "#34495e", "Chamber9": "#7f8c8d",
    }
    return colors.get(chamber, "#95a5a6")


def draw_frustum(ax, R, t, w, h, fx, scale=0.3, color="red", label=None):
    C = -R.T @ t
    # Camera frustum corners at z=1 in camera space, then scaled
    corners_cam = np.array([
        [(0 - w / 2) / fx, (0 - h / 2) / fx, 1],
        [(w - w / 2) / fx, (0 - h / 2) / fx, 1],
        [(w - w / 2) / fx, (h - h / 2) / fx, 1],
        [(0 - w / 2) / fx, (h - h / 2) / fx, 1],
    ]).T * scale
    corners_world = C + (R.T @ corners_cam).T
    apex = C
    for corner in corners_world:
        ax.plot([apex[0], corner[0]], [apex[1], corner[1]], [apex[2], corner[2]],
                color=color, linewidth=0.6, alpha=0.5)
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners_world[i, 0], corners_world[j, 0]],
                [corners_world[i, 1], corners_world[j, 1]],
                [corners_world[i, 2], corners_world[j, 2]],
                color=color, linewidth=0.6, alpha=0.5)
    ax.scatter(C[0], C[1], C[2], c=color, s=12, marker="o", zorder=5)


def main():
    parser = argparse.ArgumentParser(description="Visualize COLMAP sparse reconstruction")
    parser.add_argument("--input", type=str, default="colmap_workspace/sparse_full/round1/0",
                        help="COLMAP model directory (with points3D.bin/images.bin/cameras.bin or .txt)")
    parser.add_argument("--frustum-scale", type=float, default=0.3)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--color-by-chamber", action="store_true",
                        help="Color-code camera frustums by chamber name")
    args = parser.parse_args()

    model_dir = BASE / args.input
    print(f"Reading COLMAP model from {model_dir}...")

    # Try binary first, then text
    if (model_dir / "points3D.bin").exists():
        points, colors = read_points3d_binary(model_dir / "points3D.bin")
        cameras = read_cameras_binary(model_dir / "cameras.bin")
        poses = read_images_binary(model_dir / "images.bin")
    elif (model_dir / "points3D.txt").exists():
        points, colors = read_points3d_text(model_dir / "points3D.txt")
        cameras = read_cameras_text(model_dir / "cameras.txt")
        poses = read_images_text(model_dir / "images.txt")
    else:
        print("No model files found. Try converting with colmap model_converter first.")
        return

    print(f"  {len(points)} points, {len(cameras)} camera(s), {len(poses)} images")

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Determine which points are in which image for visual quality assessment
    points_seen_by = np.zeros(len(points))
    # (simplified - just show point cloud)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=args.point_size, alpha=0.7)

    # Draw frustums
    for pose in poses:
        cam = cameras[pose["cam_id"]]
        fx = cam["params"][0]
        w, h = cam["width"], cam["height"]

        if args.color_by_chamber and pose["name"].startswith("Chamber"):
            color = extract_chamber_color(pose["name"])
        else:
            color = "red"

        draw_frustum(ax, pose["R"], pose["t"], w, h, fx, args.frustum_scale, color=color)
        C = pose["C"]
        # Annotate with image number
        img_num = pose["name"].split("_")[1] if "_" in pose["name"] and len(pose["name"].split("_")) > 1 else ""
        chamber = pose["name"].split("_")[0]
        label = f"{chamber}_{img_num}"
        ax.text(C[0], C[1] + 0.1, C[2], label, fontsize=4, color=color, alpha=0.7)

    # Legend for chambers
    if args.color_by_chamber:
        chamber_colors = {
            "Chamber1": "#e74c3c", "Chamber2": "#3498db", "Chamber3": "#2ecc71",
            "Chamber4": "#f39c12", "Chamber5": "#9b59b6", "Chamber6": "#1abc9c",
            "Chamber7": "#e67e22", "Chamber8": "#34495e", "Chamber9": "#7f8c8d",
        }
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=k) for k, c in chamber_colors.items()
                           if any(p["name"].startswith(k) for p in poses)]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_title(f"COLMAP SfM — {len(points)} points, {len(poses)} registered cameras")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Auto-scale axes to the data
    all_pts = points.copy()
    all_C = np.array([p["C"] for p in poses])
    if len(all_C) > 0:
        all_pts = np.vstack([all_pts, all_C])
    center = all_pts.mean(axis=0)
    max_extent = np.max(np.ptp(all_pts, axis=0)) / 2 + 1
    ax.set_xlim(center[0] - max_extent, center[0] + max_extent)
    ax.set_ylim(center[1] - max_extent, center[1] + max_extent)
    ax.set_zlim(center[2] - max_extent, center[2] + max_extent)

    print("\nClose the window to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
