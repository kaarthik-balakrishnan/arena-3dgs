#!/usr/bin/env python3
"""
Per-chamber COLMAP reconstruction + alignment + merge.

Strategy:
  For each of the 9 chambers, reconstruct independently with ALL images
  that show that chamber (primary images + transition images from adjacent
  chambers). Then align adjacent chamber models using shared image poses
  and merge into one combined model.

Usage:
  python3 colmap_perchamber.py [--fresh]
"""

import subprocess, sys, os, shutil, struct, json, re, argparse, sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path(__file__).resolve().parent
IMAGES_SRC = BASE / "splat-files-processed"
WORKSPACE = BASE / "colmap_workspace"
PER_CHAMBER = WORKSPACE / "per_chamber"     # per-chamber image dirs
PER_CHAMBER_COLMAP = WORKSPACE / "per_chamber_colmap"  # per-chamber COLMAP outputs
COMBINED = WORKSPACE / "combined_model"

# All 91 filenames
ALL_FNAMES = sorted(f.name for f in IMAGES_SRC.glob("*.jpg"))

CAMERA_MODEL = "SIMPLE_RADIAL"
SINGLE_CAMERA = 1
MAX_FEATURES = 8192
PEAK_THRESHOLD = 0.004

MAPPER_PARAMS = {
    "multiple_models": 1,
    "max_num_models": 50,
    "min_model_size": 3,
    "init_min_tri_angle": 4,
    "init_min_num_inliers": 15,
    "abs_pose_min_num_inliers": 8,
    "abs_pose_min_inlier_ratio": 0.25,
    "ba_local_max_num_iterations": 25,
    "ba_global_max_num_iterations": 50,
    "ba_global_frames_freq": 500,
    "ba_global_points_freq": 250000,
}


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


def get_chambers():
    """Return list of chamber numbers [1..9]."""
    chambers = set()
    for fname in ALL_FNAMES:
        m = re.match(r"Chamber(\d+)", fname)
        if m:
            chambers.add(int(m.group(1)))
    return sorted(chambers)


def images_showing_chamber(chamber_num):
    """Return set of filenames that show a given chamber.
    
    An image shows Chamber N if:
      a) Its prefix starts with Chamber{N}_ (it's a primary image)
      b) "_Chamber{N}." or "_Chamber{N}_" appears in the filename
         (taken from another chamber looking into Chamber N)
    """
    result = set()
    for fname in ALL_FNAMES:
        if fname.startswith(f"Chamber{chamber_num}_"):
            result.add(fname)
        elif f"_Chamber{chamber_num}." in fname or f"_Chamber{chamber_num}_" in fname:
            result.add(fname)
    return result


def create_per_chamber_dirs():
    """For each chamber, create a directory with all images showing that chamber.
    
    Transition images (showing two chambers) are hard-linked to both chambers'
    directories so they get reconstructed in both, providing alignment anchors.
    """
    if PER_CHAMBER.exists():
        shutil.rmtree(PER_CHAMBER)
    PER_CHAMBER.mkdir(parents=True)

    per_chamber_files = {}
    for c in get_chambers():
        img_dir = PER_CHAMBER / f"Chamber{c}"
        img_dir.mkdir(exist_ok=True)
        fnames = images_showing_chamber(c)
        per_chamber_files[c] = fnames
        for fname in fnames:
            src = IMAGES_SRC / fname
            dst = img_dir / fname
            if not dst.exists():
                try:
                    os.link(str(src), str(dst))
                except OSError:
                    shutil.copy2(str(src), str(dst))
    return per_chamber_files


def count_images_per_chamber(per_chamber_files):
    for c, fnames in per_chamber_files.items():
        primary = sum(1 for f in fnames if f.startswith(f"Chamber{c}_"))
        aux = len(fnames) - primary
        print(f"    Chamber{c}: {len(fnames):2d} images ({primary} primary, {aux} auxiliary)")


def run_per_chamber_colmap(per_chamber_files):
    """Run COLMAP feature extraction, matching, and reconstruction per chamber."""
    if PER_CHAMBER_COLMAP.exists():
        shutil.rmtree(PER_CHAMBER_COLMAP)
    PER_CHAMBER_COLMAP.mkdir(parents=True)

    results = {}
    for c in get_chambers():
        print(f"\n{'='*60}")
        print(f"  CHAMBER {c} — {len(per_chamber_files[c])} images")
        print(f"{'='*60}")

        chamber_out = PER_CHAMBER_COLMAP / f"Chamber{c}"
        chamber_out.mkdir(exist_ok=True)
        img_dir = PER_CHAMBER / f"Chamber{c}"
        db_path = PER_CHAMBER_COLMAP / f"Chamber{c}.db"

        # Feature extraction
        run(
            f"colmap feature_extractor "
            f"--database_path {db_path} "
            f"--image_path {img_dir} "
            f"--ImageReader.camera_model {CAMERA_MODEL} "
            f"--ImageReader.single_camera {SINGLE_CAMERA} "
            f"--SiftExtraction.max_num_features {MAX_FEATURES} "
            f"--SiftExtraction.peak_threshold {PEAK_THRESHOLD} "
            f"--SiftExtraction.edge_threshold 10",
            f"Chamber {c}: Feature extraction"
        )

        # Sequential matching (chamber images are in alphabetical order = walkthrough)
        run(
            f"colmap sequential_matcher "
            f"--database_path {db_path} "
            f"--FeatureMatching.max_num_matches 32768 "
            f"--SequentialMatching.overlap 15 "
            f"--SequentialMatching.loop_detection 1 "
            f"--SequentialMatching.loop_detection_period 20 ",
            f"Chamber {c}: Sequential matching"
        )

        # Verify matches exist
        conn = sqlite3.connect(str(db_path))
        verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        conn.close()
        print(f"  Verified match pairs: {verified}")

        # Reconstruction
        sparse_out = chamber_out / "sparse"
        sparse_out.mkdir(parents=True, exist_ok=True)
        param_str = " ".join(f"--Mapper.{k} {v}" for k, v in MAPPER_PARAMS.items())
        run(
            f"colmap mapper "
            f"--database_path {db_path} "
            f"--image_path {img_dir} "
            f"--output_path {sparse_out} "
            f"{param_str} ",
            f"Chamber {c}: Reconstruction"
        )

        # Collect best model
        models = []
        for sub in sorted(sparse_out.iterdir()):
            img_bin = sub / "images.bin"
            if img_bin.exists():
                with open(img_bin, "rb") as f:
                    n_imgs = struct.unpack("Q", f.read(8))[0]
                models.append((n_imgs, sub))
        models.sort(key=lambda x: -x[0])

        if models:
            best = models[0][1]
            # Convert to text for easy reading
            txt_dir = best / "txt"
            txt_dir.mkdir(exist_ok=True)
            run(
                f"colmap model_converter "
                f"--input_path {best} "
                f"--output_path {txt_dir} "
                f"--output_type TXT",
                f"Chamber {c}: Convert best model to text",
                fatal=False
            )
            results[c] = {
                "bin_path": best,
                "txt_dir": txt_dir,
                "num_images": models[0][0],
                "best_model_num": models[0][1].name,
            }
            print(f"  Chamber {c} best model: {models[0][0]} images ({models[0][1].name})")
        else:
            print(f"  Chamber {c}: No model produced!")
            results[c] = None

    return results


def read_model_txt(txt_dir):
    """Read a COLMAP text-format model, return points, cameras, images."""
    points = {}      # point3D_id -> (xyz, rgb, error)
    cameras = {}     # camera_id -> {model, width, height, params}
    images = {}      # image_name -> {qvec, tvec, cam_id, R, C}
    
    # Points
    pts_path = txt_dir / "points3D.txt"
    if pts_path.exists():
        with open(pts_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    pt_id = int(parts[0])
                    xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
                    error = float(parts[7])
                    points[pt_id] = (xyz, rgb, error)
    
    # Cameras
    cam_path = txt_dir / "cameras.txt"
    if cam_path.exists():
        with open(cam_path) as f:
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
    
    # Images
    img_path = txt_dir / "images.txt"
    if img_path.exists():
        with open(img_path) as f:
            lines = [l.rstrip() for l in f if l.strip() and not l.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) >= 10:
                name = parts[9]
                qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                cam_id = int(parts[8])
                R = qvec2rotmat(qvec)
                C = -R.T @ tvec
                images[name] = {"qvec": qvec, "tvec": tvec, "cam_id": cam_id, "R": R, "C": C}
    
    return points, cameras, images


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
    ])


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
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


def compute_similarity_transform(src_pts, dst_pts):
    """Compute similarity transform (s, R, t) mapping src_pts to dst_pts.
    
    Uses Umeyama's algorithm (Procrustes analysis with scale).
    src_pts, dst_pts: (N, 3) numpy arrays, N >= 3.
    
    Returns (s, R, t) such that s * R @ src_pt + t ≈ dst_pt.
    """
    assert len(src_pts) >= 3, "Need at least 3 correspondences"
    
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    
    # Cross-covariance matrix
    H = src_centered.T @ dst_centered
    
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    src_var = np.sum(src_centered ** 2)
    s = np.trace(np.diag(S) @ R) / src_var if src_var > 0 else 1.0
    
    t = dst_mean - s * R @ src_mean
    
    return s, R, t


def find_shared_images(chamber_models):
    """Identify shared image names between adjacent chambers.
    
    Returns: list of (cam_a, cam_b, shared_names) for adjacent pairs.
    """
    chambers = get_chambers()
    shared = []
    for i in range(len(chambers) - 1):
        ca, cb = chambers[i], chambers[i + 1]
        if ca not in chamber_models or cb not in chamber_models:
            continue
        pts_a, cam_a, img_a = chamber_models[ca]
        pts_b, cam_b, img_b = chamber_models[cb]
        
        names_a = set(img_a.keys())
        names_b = set(img_b.keys())
        shared_names = names_a & names_b
        
        if shared_names:
            shared.append((ca, cb, sorted(shared_names)))
            print(f"    Chamber{ca} ↔ Chamber{cb}: {len(shared_names)} shared images")
            for n in sorted(shared_names):
                print(f"      - {n}")
        else:
            print(f"    Chamber{ca} ↔ Chamber{cb}: NO shared images!")
    return shared


def accumulate_transforms(chamber_models, shared_list):
    """Compute similarity transforms to align all chambers to Chamber1's frame.
    
    Chamber1 is the reference (identity transform).
    Chamber2 → Chamber1 via their shared images.
    Chamber3 → Chamber2 (then compose with Chamber2's transform) → Chamber1 frame.
    etc.
    """
    chambers = get_chambers()
    transforms = {chambers[0]: (1.0, np.eye(3), np.zeros(3))}  # Chamber1 = reference
    
    for ca, cb, names in shared_list:
        # cb should be ca+1
        # Compute transform from cb's frame to ca's frame
        pts_a, _, img_a = chamber_models[ca]
        pts_b, _, img_b = chamber_models[cb]
        
        src_pts = []  # cb's camera centers
        dst_pts = []  # ca's camera centers
        
        for name in names:
            if name in img_a and name in img_b:
                # Camera center in each model's frame
                src_pts.append(img_b[name]["C"])
                dst_pts.append(img_a[name]["C"])
        
        if len(src_pts) < 3:
            print(f"  ⚠ Only {len(src_pts)} shared cameras between Chamber{ca} and Chamber{cb} — need ≥3")
            continue
        
        src_arr = np.array(src_pts)
        dst_arr = np.array(dst_pts)
        
        s, R, t = compute_similarity_transform(src_arr, dst_arr)
        print(f"  Chamber{cb} → Chamber{ca}: s={s:.4f}, R_trace={np.trace(R):.2f}")
        
        # Compose with ca's transform to get cb → Chamber1 frame
        if ca in transforms:
            s_ca, R_ca, t_ca = transforms[ca]
            # Align cb to Chamber1: combine (cb→ca) then (ca→chamber1)
            s_total = s_ca * s
            R_total = R_ca @ R
            t_total = s_ca * R_ca @ t + t_ca
            transforms[cb] = (s_total, R_total, t_total)
        else:
            print(f"  ⚠ Chamber{ca} has no transform — skipping")
    
    return transforms


def apply_transform_to_model(points, cameras, images, s, R, t):
    """Apply similarity transform to a COLMAP model (points, cameras, images)."""
    transformed_points = {}
    for pt_id, (xyz, rgb, error) in points.items():
        xyz_new = s * R @ xyz + t
        transformed_points[pt_id] = (xyz_new, rgb, error)
    
    transformed_images = {}
    for name, data in images.items():
        R_old = data["R"]
        t_old = data["tvec"]
        C_old = data["C"]
        
        # New camera center
        C_new = s * R @ C_old + t
        
        # New rotation and translation
        R_new = R_old @ R.T
        t_new = -R_new @ C_new
        
        qvec_new = rotmat2qvec(R_new)
        
        transformed_images[name] = {
            "qvec": qvec_new,
            "tvec": t_new,
            "cam_id": data["cam_id"],
            "R": R_new,
            "C": C_new,
        }
    
    return transformed_points, cameras, transformed_images


def merge_models(all_chamber_models, transforms, per_chamber_files):
    """Merge all aligned chamber models into one combined model."""
    combined_points = {}    # point3D_id -> (xyz, rgb, error)
    combined_cameras = {}   # camera_id -> {model, width, height, params}
    combined_images = {}    # image_name -> {qvec, tvec, cam_id, R, C}
    
    # We need unique point IDs to avoid collisions
    point_id_offset = 0
    camera_id_offset = 0
    
    chambers = get_chambers()
    
    for c in chambers:
        if c not in all_chamber_models or c not in transforms:
            print(f"  Skipping Chamber {c} — no model or transform")
            continue
        
        points, cameras, images = all_chamber_models[c]
        s, R, t = transforms[c]
        
        # Transform
        pts_tf, cams_tf, imgs_tf = apply_transform_to_model(points, cameras, images, s, R, t)
        
        # Add to combined model with offset IDs
        for pt_id, (xyz, rgb, error) in pts_tf.items():
            combined_points[pt_id + point_id_offset] = (xyz, rgb, error)
        point_id_offset += max(pts_tf.keys()) + 1 if pts_tf else 0
        
        for cam_id, data in cams_tf.items():
            if cam_id not in combined_cameras:
                combined_cameras[cam_id] = data
        
        for name, data in imgs_tf.items():
            if name not in combined_images:
                combined_images[name] = data
    
    print(f"\n  Combined model: {len(combined_points)} points, "
          f"{len(combined_cameras)} cameras, {len(combined_images)} images")
    
    return combined_points, combined_cameras, combined_images


def write_model_txt(out_dir, points, cameras, images):
    """Write a COLMAP text-format model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Cameras
    with open(out_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            data = cameras[cam_id]
            params_str = " ".join(str(p) for p in data["params"])
            f.write(f"{cam_id} {data['model']} {data['width']} {data['height']} {params_str}\n")
    
    # Images
    with open(out_dir / "images.txt", "w") as f:
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
            f.write("\n")  # empty points2D
    
    # Points
    with open(out_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for pt_id in sorted(points):
            xyz, rgb, error = points[pt_id]
            f.write(f"{pt_id} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]} {error:.6f}\n")
    
    print(f"  Written to {out_dir}")
    print(f"    cameras.txt: {len(cameras)} cameras")
    print(f"    images.txt: {len(images)} images")
    print(f"    points3D.txt: {len(points)} points")


def write_registration_report(per_chamber_files, chamber_results, chamber_models, transforms, combined_images):
    """Write a report of what registered where."""
    report = {
        "total_images": len(ALL_FNAMES),
        "per_chamber": {},
        "transforms": {},
        "combined": len(combined_images),
    }
    
    for c in get_chambers():
        report["per_chamber"][f"Chamber{c}"] = {
            "total": len(per_chamber_files[c]),
        }
        if c in chamber_results and chamber_results[c]:
            report["per_chamber"][f"Chamber{c}"]["model_images"] = chamber_results[c]["num_images"]
        else:
            report["per_chamber"][f"Chamber{c}"]["model_images"] = 0
        
        if c in chamber_models:
            pts, cams, imgs = chamber_models[c]
            report["per_chamber"][f"Chamber{c}"]["registered"] = len(imgs)
            report["per_chamber"][f"Chamber{c}"]["points"] = len(pts)
        else:
            report["per_chamber"][f"Chamber{c}"]["registered"] = 0
            report["per_chamber"][f"Chamber{c}"]["points"] = 0
    
    for (ca, cb, names) in find_shared_images(chamber_models):
        report["transforms"][f"Chamber{ca}_Chamber{cb}"] = {
            "shared_images": len(names),
            "shared_names": names,
        }
        if ca in transforms and cb in transforms:
            s, R, t = transforms[cb]
            report["transforms"][f"Chamber{ca}_Chamber{cb}"]["scale"] = float(s)
            report["transforms"][f"Chamber{ca}_Chamber{cb}"]["rotation"] = R.tolist()
            report["transforms"][f"Chamber{ca}_Chamber{cb}"]["translation"] = t.tolist()
    
    report["combined_register"] = len(combined_images)
    
    report_path = WORKSPACE / "per_chamber_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")
    
    return report


def print_summary(per_chamber_files, chamber_results, combined_images):
    total = len(ALL_FNAMES)
    registered = len(combined_images)
    
    print(f"\n{'='*60}")
    print(f"  PER-CHAMBER REGISTRATION SUMMARY")
    print(f"{'='*60}")
    
    for c in get_chambers():
        total_c = len(per_chamber_files[c])
        if c in chamber_results and chamber_results[c]:
            reg_c = chamber_results[c]["num_images"]
        else:
            reg_c = 0
        # Where did those images end up?
        # Combined: count images from this chamber in combined model
        primary_of_c = [f for f in ALL_FNAMES if f.startswith(f"Chamber{c}_")]
        in_combined = sum(1 for f in primary_of_c if f in combined_images)
        bar = "█" * in_combined + "░" * (len(primary_of_c) - in_combined)
        print(f"  Chamber{c}: {bar}  {in_combined}/{len(primary_of_c)}")
    
    print(f"\n  Combined model: {registered}/{total} registered ({100 * registered // total}%)")
    
    unregistered = sorted(set(ALL_FNAMES) - set(combined_images.keys()))
    if unregistered:
        print(f"\n  Unregistered ({len(unregistered)}):")
        for name in unregistered:
            print(f"    {name}")


def main():
    parser = argparse.ArgumentParser(description="Per-chamber COLMAP + alignment")
    parser.add_argument("--fresh", action="store_true", help="Delete all per-chamber outputs and recompute")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip per-chamber COLMAP, only do alignment")
    args = parser.parse_args()

    print("=" * 60)
    print("  PER-CHAMBER COLMAP RECONSTRUCTION + ALIGNMENT")
    print("=" * 60)
    
    # Step 1: Create per-chamber image directories
    print("\n[1/5] Creating per-chamber image sets...")
    per_chamber_files = create_per_chamber_dirs()
    count_images_per_chamber(per_chamber_files)
    
    # Step 2: Run per-chamber COLMAP
    chamber_results = {}
    if not args.skip_colmap:
        print("\n[2/5] Running per-chamber COLMAP...")
        chamber_results = run_per_chamber_colmap(per_chamber_files)
    else:
        print("\n[2/5] Skipping COLMAP (--skip-colmap)")
        # Load existing results from disk
        for c in get_chambers():
            models_dir = PER_CHAMBER_COLMAP / f"Chamber{c}" / "sparse"
            if models_dir.exists():
                model_subdirs = sorted(models_dir.iterdir())
                for sub in model_subdirs:
                    img_bin = sub / "images.bin"
                    if img_bin.exists():
                        with open(img_bin, "rb") as f:
                            n = struct.unpack("Q", f.read(8))[0]
                        txt_dir = sub / "txt"
                        if txt_dir.exists():
                            chamber_results[c] = {
                                "bin_path": sub,
                                "txt_dir": txt_dir,
                                "num_images": n,
                                "best_model_num": sub.name,
                            }
                            print(f"  Chamber{c}: loaded model {sub.name} ({n} images)")
                        break
    
    # Step 3: Read all chamber models
    print("\n[3/5] Reading chamber models...")
    chamber_models = {}
    for c in get_chambers():
        if c in chamber_results and chamber_results[c]:
            pts, cams, imgs = read_model_txt(chamber_results[c]["txt_dir"])
            chamber_models[c] = (pts, cams, imgs)
            print(f"  Chamber{c}: {len(pts)} points, {len(cams)} cameras, {len(imgs)} images")
        else:
            print(f"  Chamber{c}: No model available")
    
    # Step 4: Find shared images and compute transforms
    print("\n[4/5] Computing alignment transforms...")
    shared_list = find_shared_images(chamber_models)
    transforms = accumulate_transforms(chamber_models, shared_list)
    
    for c in get_chambers():
        if c in transforms:
            s, R, t = transforms[c]
            print(f"  Chamber{c}: s={s:.4f}")
        else:
            print(f"  Chamber{c}: (reference / no transform)")
    
    # Step 5: Merge into combined model
    print("\n[5/5] Merging into combined model...")
    combined_points, combined_cameras, combined_images = merge_models(
        chamber_models, transforms, per_chamber_files
    )
    
    # Write combined model
    write_model_txt(COMBINED, combined_points, combined_cameras, combined_images)
    
    # Also write as binary if possible (model_converter can do this)
    bin_combined = WORKSPACE / "combined_model_bin"
    run(
        f"colmap model_converter "
        f"--input_path {COMBINED} "
        f"--output_path {bin_combined} "
        f"--output_type BIN",
        "Convert combined model to binary",
        fatal=False
    )
    
    # Report
    write_registration_report(per_chamber_files, chamber_results, chamber_models, transforms, combined_images)
    print_summary(per_chamber_files, chamber_results, combined_images)
    
    print(f"\n  Done! Combined model: {COMBINED}")
    
    return 0


if __name__ == "__main__":
    main()
