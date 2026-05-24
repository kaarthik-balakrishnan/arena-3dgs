#!/usr/bin/env python3
"""
Merge Model 1 into Model 0 using shared image poses, then register remaining
91-9=82 unregistered images via image_registrator.

Steps:
  1. Find shared images between Model 0 and Model 1 (by filename)
  2. Create reference file with Model 0 poses for shared images
  3. Align Model 1 to Model 0's coordinate frame via model_aligner
  4. Merge aligned Model 1 into Model 0 via model_merger
  5. Register remaining images via image_registrator
  6. Report results
"""

import subprocess, sys, os, shutil, struct, json, sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE / "colmap_workspace"
EXHAUSTIVE_DIR = WORKSPACE / "sparse_exhaustive"
DATABASE = WORKSPACE / "database_exhaustive.db"
IMAGES_SRC = BASE / "splat-files-processed"

MERGE_OUT = WORKSPACE / "model_merge"
ALIGNED_OUT = WORKSPACE / "model_aligned"
EXTENDED_OUT = WORKSPACE / "model_extended"


def run(cmd, desc, fatal=True):
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"{'─'*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED (code {result.returncode})")
        if fatal:
            sys.exit(1)
    return result.returncode


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
    ])


def read_images_text(path):
    """Read images.txt, return {name: {qvec, tvec, cam_id, image_id, C}}."""
    images = {}
    with open(path) as f:
        lines = [l.rstrip() for l in f if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            cam_id = int(parts[8])
            name = parts[9]
            R = qvec2rotmat(qvec)
            C = -R.T @ tvec
            images[name] = {"qvec": qvec, "tvec": tvec, "cam_id": cam_id, "image_id": image_id, "R": R, "C": C}
    return images


def read_images_text_full(path):
    """Read images.txt including POINTS2D data.
    
    Returns: {name: {qvec, tvec, cam_id, image_id, R, C, points2D}}
    where points2D is list of (x, y, point3D_id)
    """
    images = {}
    with open(path) as f:
        lines = [l.rstrip() for l in f]
    
    # Strip comments and blank lines
    clean = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            clean.append(stripped)
    
    i = 0
    while i < len(clean):
        parts = clean[i].split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            cam_id = int(parts[8])
            name = parts[9]
            R = qvec2rotmat(qvec)
            C = -R.T @ tvec
            
            # Read points2D
            i += 1
            points2D = []
            if i < len(clean):
                pt_parts = clean[i].strip().split()
                if pt_parts and len(pt_parts) >= 3 and not any(c.isalpha() for c in pt_parts[0]):
                    # This is a points2D line
                    num_pts = len(pt_parts) // 3
                    for j in range(num_pts):
                        x = float(pt_parts[3*j])
                        y = float(pt_parts[3*j + 1])
                        pt_id = int(pt_parts[3*j + 2])
                        points2D.append((x, y, pt_id))
                    i += 1
            
            images[name] = {
                "qvec": qvec, "tvec": tvec, "cam_id": cam_id,
                "image_id": image_id, "R": R, "C": C,
                "points2D": points2D
            }
        else:
            i += 1
    
    return images


def write_images_text(path, images):
    """Write images.txt including POINTS2D data.
    images: {name: {qvec, tvec, cam_id, points2D}}
    Re-number image IDs sequentially.
    """
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        
        for img_id, name in enumerate(sorted(images), 1):
            data = images[name]
            q = data["qvec"]
            t = data["tvec"]
            f.write(f"{img_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {data['cam_id']} {name}\n")
            
            # Write points2D
            pts = data.get("points2D", [])
            if pts:
                pt_strs = [f"{x:.6f} {y:.6f} {pt_id}" for (x, y, pt_id) in pts]
                f.write(" ".join(pt_strs) + "\n")
            else:
                f.write("\n")


def write_points3d_text(path, points):
    """Write points3D.txt.
    points: {point3D_id: (xyz, rgb, error)}
    """
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for pt_id in sorted(points):
            xyz, rgb, error = points[pt_id]
            f.write(f"{pt_id} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]} {error:.6f}\n")


def write_cameras_text(path, cameras):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            data = cameras[cam_id]
            params_str = " ".join(str(p) for p in data["params"])
            f.write(f"{cam_id} {data['model']} {data['width']} {data['height']} {params_str}\n")


def compute_similarity_transform(src_pts, dst_pts):
    """Procrustes with N>=3 points."""
    assert len(src_pts) >= 3, f"Need at least 3 correspondences, got {len(src_pts)}"
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    src_var = np.sum(src_centered ** 2)
    s = np.trace(np.diag(S) @ R) / src_var if src_var > 0 else 1.0
    t = dst_mean - s * R @ src_mean
    return s, R, t


def compute_transform_from_cameras(shared_names, images0, images1):
    """Compute (s, R, t) mapping Model1→Model0 from shared image poses.
    
    Uses both camera centers and orientations. With 2+ shared images:
      - s = ||dC0|| / ||dC1||  (ratio of center separations)
      - R = closest rotation to average of R0_i^T @ R1_i
      - t = avg(C0_i - s * R @ C1_i)
    """
    c0_list = [images0[n]["C"] for n in shared_names]
    c1_list = [images1[n]["C"] for n in shared_names]
    r0_list = [images0[n]["R"] for n in shared_names]
    r1_list = [images1[n]["R"] for n in shared_names]
    
    # Scale from distance ratio between camera centers
    if len(shared_names) >= 2:
        d0 = np.linalg.norm(c0_list[1] - c0_list[0])
        d1 = np.linalg.norm(c1_list[1] - c1_list[0])
        s = d0 / d1 if d1 > 1e-10 else 1.0
    else:
        s = 1.0
    
    # Rotation: R = R0_i^T @ R1_i (should be consistent across images)
    R_estimates = []
    for R0, R1 in zip(r0_list, r1_list):
        R_est = R0.T @ R1
        # Ensure determinant 1
        U, _, Vt = np.linalg.svd(R_est)
        R_est = U @ Vt
        if np.linalg.det(R_est) < 0:
            Vt[-1] *= -1
            R_est = U @ Vt
        R_estimates.append(R_est)
    
    R = np.mean(R_estimates, axis=0)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    
    # Translation: average of C0 - s*R@C1
    t_estimates = [c0 - s * R @ c1 for c0, c1 in zip(c0_list, c1_list)]
    t = np.mean(t_estimates, axis=0)
    
    return s, R, t


def rotmat2qvec(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def apply_transform_to_points(points, s, R, t):
    """Transform points: {pt_id: (xyz, rgb, error)}"""
    result = {}
    for pt_id, (xyz, rgb, error) in points.items():
        xyz_new = s * R @ xyz + t
        result[pt_id] = (xyz_new, rgb, error)
    return result


def apply_transform_to_image(data, s, R, t):
    """Transform a single image's pose."""
    R_old = data["R"]
    C_old = data["C"]
    C_new = s * R @ C_old + t
    R_new = R_old @ R.T
    t_new = -R_new @ C_new
    qvec_new = rotmat2qvec(R_new)
    return {
        "qvec": qvec_new,
        "tvec": t_new,
        "cam_id": data["cam_id"],
        "image_id": data["image_id"],
        "R": R_new,
        "C": C_new,
        "points2D": data.get("points2D", []),
    }


# ── Step 1: Identify shared images ─────────────────────────────
print("=" * 60)
print("  MERGE MODEL 1 INTO MODEL 0 + REGISTER REMAINING")
print("=" * 60)

model0_txt = EXHAUSTIVE_DIR / "0_txt"
model1_txt = EXHAUSTIVE_DIR / "1_txt"
model0_bin = EXHAUSTIVE_DIR / "0"
model1_bin = EXHAUSTIVE_DIR / "1"

images0 = read_images_text_full(model0_txt / "images.txt")
images1 = read_images_text_full(model1_txt / "images.txt")

shared = set(images0.keys()) & set(images1.keys())
print(f"\n[1] Shared images between Model 0 and Model 1:")
for name in sorted(shared):
    print(f"    {name}")
print(f"    ({len(shared)} total)")

# ── Step 2: Compute transform ──────────────────────────────────
print(f"\n[2] Computing alignment transform (Model 1 → Model 0)...")
print(f"  Shared images: {len(shared)}")

if len(shared) < 2:
    print("  ⚠ Need at least 2 shared images for alignment. Exiting.")
    sys.exit(1)

s, R, t = compute_transform_from_cameras(list(shared), images0, images1)
print(f"  Scale: {s:.6f}")
print(f"  Rotation trace: {np.trace(R):.4f}")
print(f"  Translation: {t}")

# Validate: check residual error
src_centers = np.array([images1[name]["C"] for name in shared])
dst_centers = np.array([images0[name]["C"] for name in shared])
transformed_src = s * (R @ src_centers.T).T + t
residuals = np.linalg.norm(transformed_src - dst_centers, axis=1)
print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.4f}")
sorted_shared = sorted(shared)
for i, name in enumerate(sorted_shared):
    print(f"    {name}: {residuals[i]:.4f}")

# ── Step 3: Read full model data, transform Model 1, merge ─────
print(f"\n[3] Merging models...")

# Read points3D
def read_points_txt(path):
    points = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                pt_id = int(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
                error = float(parts[7])
                points[pt_id] = (xyz, rgb, error)
    return points

points0 = read_points_txt(model0_txt / "points3D.txt")
points1 = read_points_txt(model1_txt / "points3D.txt")

# Read cameras
def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                cam_id = int(parts[0])
                cameras[cam_id] = {
                    "model": parts[1],
                    "width": int(parts[2]),
                    "height": int(parts[3]),
                    "params": list(map(float, parts[4:])),
                }
    return cameras

cameras0 = read_cameras_txt(model0_txt / "cameras.txt")
cameras1 = read_cameras_txt(model1_txt / "cameras.txt")

# Transform Model 1 points
points1_tf = apply_transform_to_points(points1, s, R, t)

# Transform Model 1 images
images1_tf = {}
for name, data in images1.items():
    images1_tf[name] = apply_transform_to_image(data, s, R, t)

# Offset point IDs for Model 1 to avoid collision
max_pt_id0 = max(points0.keys()) if points0 else 0
pt_id_offset = max_pt_id0 + 1

points_merged = dict(points0)
points_merged.update({pt_id + pt_id_offset: v for pt_id, v in points1_tf.items()})

# Update point IDs in images1_tf's points2D
for name, data in images1_tf.items():
    updated_pts = []
    for (x, y, pt_id) in data.get("points2D", []):
        if pt_id >= 0:
            updated_pts.append((x, y, pt_id + pt_id_offset))
        else:
            updated_pts.append((x, y, pt_id))
    data["points2D"] = updated_pts

# Merge images (Model 1 unique images only)
images_merged = dict(images0)
for name, data in images1_tf.items():
    if name not in images_merged:
        images_merged[name] = data

# Merge cameras (should be same camera model)
cameras_merged = dict(cameras0)

print(f"  Model 0: {len(points0)} points, {len(cameras0)} cameras, {len(images0)} images")
print(f"  Model 1: {len(points1)} points, {len(cameras1)} cameras, {len(images1)} images")
print(f"  Merged:  {len(points_merged)} points, {len(cameras_merged)} cameras, {len(images_merged)} images")

# ── Step 4: Write merged text model ────────────────────────────
print(f"\n[4] Writing merged model...")

if MERGE_OUT.exists():
    shutil.rmtree(MERGE_OUT)
MERGE_OUT.mkdir(parents=True)

write_cameras_text(MERGE_OUT / "cameras.txt", cameras_merged)
write_images_text(MERGE_OUT / "images.txt", images_merged)
write_points3d_text(MERGE_OUT / "points3D.txt", points_merged)

print(f"  Written to {MERGE_OUT}")

# ── Step 5: Convert to binary and run image_registrator ────────
BIN_MERGE = WORKSPACE / "model_merge_bin"
if BIN_MERGE.exists():
    shutil.rmtree(BIN_MERGE)

run(
    f"colmap model_converter "
    f"--input_path {MERGE_OUT} "
    f"--output_path {BIN_MERGE} "
    f"--output_type BIN",
    "Convert merged model to binary"
)

print(f"\n[5] Registering remaining images via image_registrator...")
if EXTENDED_OUT.exists():
    shutil.rmtree(EXTENDED_OUT)

run(
    f"colmap image_registrator "
    f"--database_path {DATABASE} "
    f"--input_path {BIN_MERGE} "
    f"--output_path {EXTENDED_OUT} "
    f"--Mapper.init_min_num_inliers 10 "
    f"--Mapper.init_min_tri_angle 4 "
    f"--Mapper.abs_pose_min_num_inliers 6 "
    f"--Mapper.abs_pose_min_inlier_ratio 0.1 ",
    "Register remaining images with relaxed thresholds",
    fatal=False
)

# ── Step 6: Analyze results ────────────────────────────────────
print(f"\n[6] Analyzing results...")

def get_registered_names_from_txt(txt_dir):
    names = set()
    imgtxt = txt_dir / "images.txt"
    if imgtxt.exists():
        with open(imgtxt) as f:
            lines = [l.rstrip() for l in f if l.strip() and not l.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) >= 10:
                names.add(parts[9])
    return names

# Check what extended/ gives us
extended_txt = EXTENDED_OUT / "txt"
if not extended_txt.exists():
    extended_txt.mkdir(parents=True, exist_ok=True)
    run(
        f"colmap model_converter "
        f"--input_path {EXTENDED_OUT} "
        f"--output_path {extended_txt} "
        f"--output_type TXT",
        "Convert extended model to text",
        fatal=False
    )

extended_names = get_registered_names_from_txt(extended_txt) if EXTENDED_OUT.exists() else set()
merged_names = set(images_merged.keys())

print(f"  Before image_registrator: {len(merged_names)} images")
print(f"  After image_registrator: {len(extended_names)} images (if tool succeeded)")

# Try with the extended output
all_fnames = {f.name for f in IMAGES_SRC.glob("*.jpg")}

if extended_names:
    registered = extended_names
    source_dir = EXTENDED_OUT
else:
    # If image_registrator failed, use our merged model
    print(f"  image_registrator didn't add images, converting merged model directly")
    registered = merged_names
    source_dir = BIN_MERGE

unregistered = all_fnames - registered
print(f"\n  Registered: {len(registered)}/{len(all_fnames)}")
print(f"  Unregistered: {len(unregistered)}")

if unregistered:
    print(f"\n  Unregistered:")
    for name in sorted(unregistered):
        print(f"    {name}")

# By chamber
print(f"\n  By chamber:")
for c in range(1, 10):
    primary = [f for f in all_fnames if f.startswith(f"Chamber{c}_")]
    reg_c = sum(1 for f in primary if f in registered)
    bar = "█" * reg_c + "░" * (len(primary) - reg_c)
    print(f"    Chamber{c}: {bar}  {reg_c}/{len(primary)}")

# Save report
report = {
    "total": len(all_fnames),
    "registered": len(registered),
    "unregistered": len(unregistered),
    "merge_scale": float(s),
    "merge_residual_rms": float(np.sqrt(np.mean(residuals**2))),
    "model0_points": len(points0),
    "model1_points": len(points1),
    "merged_points": len(points_merged),
}
with open(WORKSPACE / "merge_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Determine which model directory to use for visualization
if extended_names and len(extended_names) > len(merged_names):
    final_dir = extended_txt
else:
    # Use our merged text model
    final_dir = MERGE_OUT

print(f"\n  Final model for visualization: {final_dir}")
print(f"  Report: {WORKSPACE}/merge_report.json")
print(f"\n  Done!")
