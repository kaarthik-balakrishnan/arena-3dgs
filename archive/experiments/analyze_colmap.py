import os
import re
import statistics
from collections import defaultdict, Counter

BASE = "/Users/kaarthikabhinav/Documents/GaussianSplat/colmap_workspace"

MODELS = {
    "Global (91 img, exhaustive+global)": "sparse_global/0/txt",
    "Exhaustive Model 0 (76 img)": "sparse_exhaustive/0_txt",
    "Exhaustive Model 1 (8 img)": "sparse_exhaustive/1_txt",
}


def read_points(path):
    points = []
    n_comments = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                n_comments += 1
                continue
            parts = line.split()
            pid = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error = float(parts[7])
            track_parts = parts[8:]
            track = [(int(track_parts[i]), int(track_parts[i + 1])) for i in range(0, len(track_parts), 2)]
            points.append({
                "id": pid,
                "xyz": (x, y, z),
                "rgb": (r, g, b),
                "error": error,
                "track": track,
                "track_len": len(track),
            })
    return points


def read_images(path):
    images = []
    n_comments = 0
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            i += 1
            continue
        # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        img_id = int(parts[0])
        name = parts[9]
        camera_id = int(parts[8])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

        # Line 2: POINTS2D[] as (X, Y, POINT3D_ID) triples
        i += 1
        pts_line = lines[i]
        pt_parts = pts_line.split()
        points2d = []
        for j in range(0, len(pt_parts), 3):
            x_2d = float(pt_parts[j])
            y_2d = float(pt_parts[j + 1])
            p3d_id = int(pt_parts[j + 2])
            points2d.append((x_2d, y_2d, p3d_id))

        n_obs = sum(1 for _, _, pid in points2d if pid != -1)
        n_total = len(points2d)

        images.append({
            "id": img_id,
            "name": name,
            "camera_id": camera_id,
            "pose": (qw, qx, qy, qz, tx, ty, tz),
            "points2d": points2d,
            "n_observations": n_obs,
            "n_features": n_total,
        })
        i += 1
    return images


def extract_chamber(name):
    m = re.match(r"(Chamber\d+)", name)
    return m.group(1) if m else "Other"


def analyze_model(label, dirpath):
    pt_path = os.path.join(BASE, dirpath, "points3D.txt")
    img_path = os.path.join(BASE, dirpath, "images.txt")
    cam_path = os.path.join(BASE, dirpath, "cameras.txt")

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")

    # Cameras
    cams = []
    with open(cam_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split(",") if "," in line else line.split()
                cams.append(parts)
    print(f"\n  Cameras: {len(cams)}")
    for c in cams:
        print(f"    ID={c[0]}, Model={c[1]}, W={c[2]}, H={c[3]}")

    # Images
    images = read_images(img_path)
    print(f"\n  Images: {len(images)}")

    # Chamber distribution
    chamber_counts = Counter()
    for img in images:
        ch = extract_chamber(img["name"])
        chamber_counts[ch] += 1
    print(f"  Chamber distribution:")
    for ch in sorted(chamber_counts):
        print(f"    {ch}: {chamber_counts[ch]} images")

    # Per-image stats
    n_obs_list = [img["n_observations"] for img in images]
    n_feat_list = [img["n_features"] for img in images]
    print(f"\n  Observations per image:")
    print(f"    Mean: {statistics.mean(n_obs_list):.1f}")
    print(f"    Median: {statistics.median(n_obs_list):.1f}")
    print(f"    Min: {min(n_obs_list)}, Max: {max(n_obs_list)}")
    print(f"  Total 2D features per image:")
    print(f"    Mean: {statistics.mean(n_feat_list):.1f}")
    print(f"    Median: {statistics.median(n_feat_list):.1f}")
    print(f"    Min: {min(n_feat_list)}, Max: {max(n_feat_list)}")

    # Points
    points = read_points(pt_path)
    print(f"\n  3D Points: {len(points)}")
    track_lens = [p["track_len"] for p in points]
    print(f"  Track length:")
    print(f"    Mean: {statistics.mean(track_lens):.2f}")
    print(f"    Median: {statistics.median(track_lens):.1f}")
    print(f"    Min: {min(track_lens)}, Max: {max(track_lens)}")
    # Track length distribution
    tl_dist = Counter(track_lens)
    print(f"  Track length distribution (top 10):")
    for tl, cnt in sorted(tl_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {tl}: {cnt} points ({100 * cnt / len(points):.1f}%)")

    # Reprojection error stats
    errors = [p["error"] for p in points]
    print(f"  Reprojection error:")
    print(f"    Mean: {statistics.mean(errors):.4f}")
    print(f"    Median: {statistics.median(errors):.4f}")
    print(f"    Min: {min(errors):.6f}, Max: {max(errors):.4f}")

    return {
        "label": label,
        "n_cameras": len(cams),
        "n_images": len(images),
        "n_points": len(points),
        "track_lens": track_lens,
        "errors": errors,
        "n_obs_list": n_obs_list,
        "n_feat_list": n_feat_list,
        "chamber_counts": chamber_counts,
    }


results = []
for label, dirpath in MODELS.items():
    r = analyze_model(label, dirpath)
    results.append(r)

# Comparison summary
print(f"\n\n{'=' * 80}")
print(f"  COMPARISON SUMMARY")
print(f"{'=' * 80}")
print(f"\n{'Model':45s} {'Images':>7s} {'Points':>8s} {'Mean Track':>10s} {'Median Track':>12s} {'Mean Obs/Img':>12s} {'Mean Feat/Img':>13s} {'Mean Err':>10s}")
print(f"{'-' * 45} {'-' * 7} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 13} {'-' * 10}")
for r in results:
    mt = statistics.mean(r["track_lens"])
    medt = statistics.median(r["track_lens"])
    mo = statistics.mean(r["n_obs_list"])
    mf = statistics.mean(r["n_feat_list"])
    me = statistics.mean(r["errors"])
    print(f"{r['label']:45s} {r['n_images']:>7d} {r['n_points']:>8d} {mt:>10.2f} {medt:>12.1f} {mo:>12.1f} {mf:>13.1f} {me:>10.4f}")

print(f"\n\n  Chamber distribution across models:")
all_chambers = sorted(set(c for r in results for c in r["chamber_counts"]))
header = f"{'Chamber':20s}"
for r in results:
    short = r["label"].split("(")[0].strip()
    header += f" {short:>18s}"
print(f"\n  {header}")
print(f"  {'-' * 20} {'-' * 18 * len(results)}")
for ch in all_chambers:
    row = f"  {ch:20s}"
    for r in results:
        cnt = r["chamber_counts"].get(ch, 0)
        row += f" {cnt:>18d}"
    print(row)
