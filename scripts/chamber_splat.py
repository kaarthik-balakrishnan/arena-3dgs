#!/usr/bin/env python3
"""
Chamber-based Gaussian Splatting with cross-chamber alignment.

Key idea: use the naming convention (ChamberX_NN_ChamberY.jpg) to:
1. Run per-chamber COLMAP independently (better for inconsistent lighting)
2. Find 3D-3D correspondences between adjacent chambers via transition images
3. Compute rigid transforms and merge into a unified model
4. Train a single 3DGS on the merged data (or stitch per-chamber 3DGS)

Naming convention:
  ChamberX_NN.jpg              — interior of Chamber X, no transition visible
  ChamberX_NN_ChamberY.jpg     — taken FROM Chamber X, Chamber Y visible in frame
  ChamberX_NN_topview.jpg      — top-down angled view showing walls of Chamber X
  ChamberX_NN_outside.jpg      — exterior shot near Chamber X (optional)
"""
import os, sys, re, struct, time, json, sqlite3, shutil, subprocess, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════
#  1. DATA PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_image_name(filename):
    stem = Path(filename).stem
    m = re.match(
        r'Chamber(\d+)_(\d+)(?:_(Chamber(\d+)))?(?:_(topview|outside))?$',
        stem, re.IGNORECASE
    )
    if not m:
        return None
    return {
        'source': int(m.group(1)),
        'seq': int(m.group(2)),
        'has_target': m.group(4) is not None,
        'target': int(m.group(4)) if m.group(4) else None,
        'view': m.group(5) if m.group(5) else 'interior',
        'stem': stem,
        'filename': filename,
    }


def categorize_images(image_dir):
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob('*.jpg'))
    by_chamber = defaultdict(list)
    all_parsed = {}
    for p in images:
        info = parse_image_name(p.name)
        if info is None:
            print(f"  WARNING: Could not parse {p.name}")
            continue
        all_parsed[p.name] = info
        by_chamber[info['source']].append(info)

    # Also track transition images between each pair
    transitions = defaultdict(list)
    for info in all_parsed.values():
        if info['target'] is not None:
            pair = tuple(sorted([info['source'], info['target']]))
            transitions[pair].append(info)

    return dict(by_chamber), dict(transitions), all_parsed


def print_dataset_summary(by_chamber, transitions):
    print("Dataset summary:")
    print(f"  Chambers: {sorted(by_chamber.keys())}")
    print()
    for cid in sorted(by_chamber.keys()):
        imgs = by_chamber[cid]
        interior = [i for i in imgs if i['view'] == 'interior' and i['target'] is None]
        trans = [i for i in imgs if i['target'] is not None]
        topview = [i for i in imgs if i['view'] == 'topview']
        outside = [i for i in imgs if i['view'] == 'outside']
        print(f"  Chamber {cid}: {len(imgs)} photos"
              f" = {len(interior)} interior + {len(trans)} transitions"
              f" + {len(topview)} topview + {len(outside)} outside")
    print()
    print("  Chamber adjacencies with transition photos:")
    for pair in sorted(transitions.keys()):
        fwd = [t for t in transitions[pair] if t['source'] == pair[0]]
        rev = [t for t in transitions[pair] if t['source'] == pair[1]]
        print(f"    {pair[0]} ↔ {pair[1]}: {len(fwd)}→{len(rev)}← ({len(fwd)+len(rev)} total)")


def filter_bad_images(by_chamber, quality_scores=None, brightness_range=(20, 240), min_sharpness=0.3):
    filtered = {}
    removed = []
    for cid, imgs in by_chamber.items():
        good = []
        for img in imgs:
            name = img['filename']
            skip = False
            if quality_scores and name in quality_scores:
                q = quality_scores[name]
                if q.get('brightness', 128) < brightness_range[0]:
                    removed.append((name, f"too dark ({q['brightness']:.0f})"))
                    skip = True
                elif q.get('brightness', 128) > brightness_range[1]:
                    removed.append((name, f"overexposed ({q['brightness']:.0f})"))
                    skip = True
                if q.get('sharpness', 999) < min_sharpness:
                    removed.append((name, f"too blurry ({q['sharpness']:.2f})"))
                    skip = True
            if not skip:
                good.append(img)
        filtered[cid] = good
    return filtered, removed


# ═══════════════════════════════════════════════════════════════════════
#  2. COLMAP MODEL I/O
# ═══════════════════════════════════════════════════════════════════════

def read_cameras_text(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]],
            }
    return cameras


def read_images_text(path):
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and l[0] != '#']
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id = int(parts[0])
        images[img_id] = {
            'id': img_id,
            'qvec': np.array([float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])]),
            'tvec': np.array([float(parts[5]), float(parts[6]),
                              float(parts[7])]),
            'camera_id': int(parts[8]),
            'name': parts[9],
            'points2D': [],
        }
        if i + 1 < len(lines) and lines[i + 1].strip():
            pts_parts = lines[i + 1].strip().split()
            for j in range(0, len(pts_parts), 3):
                x = float(pts_parts[j])
                y = float(pts_parts[j + 1])
                p3d_id = int(pts_parts[j + 2])
                images[img_id]['points2D'].append({
                    'x': x, 'y': y, 'point3D_id': p3d_id,
                })
    return images


def read_points3d_text(path):
    points = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and l[0] != '#']
    for line in lines:
        parts = line.split()
        pid = int(parts[0])
        points[pid] = {
            'id': pid,
            'xyz': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
            'rgb': np.array([int(parts[4]), int(parts[5]), int(parts[6])]),
            'error': float(parts[7]),
            'track': [],
        }
        for k in range(8, len(parts), 2):
            points[pid]['track'].append({
                'image_id': int(parts[k]),
                'point2D_idx': int(parts[k + 1]),
            })
    return points


def read_cameras_binary(path):
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('Q', f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack('Q', f.read(8))[0]
            model_id = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            param_counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9,
                            7: 9, 8: 5, 9: 4, 10: 5, 11: 8, 12: 12, 13: 7}
            n_params = param_counts.get(model_id, 4)
            remaining = len(f.read()) - f.tell()
            actual_params = remaining // 8
            if actual_params != n_params:
                if actual_params > 0 and actual_params <= 20:
                    n_params = actual_params
            cameras[cam_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': [0.0] * n_params,
            }
    print(f"  WARNING: binary camera parsing may be unreliable "
          f"(COLMAP version-dependent). Prefer text format.")
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, 'rb') as f:
        num = struct.unpack('Q', f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack('Q', f.read(8))[0]
            qvec = struct.unpack('dddd', f.read(32))
            tvec = struct.unpack('ddd', f.read(24))
            cam_id = struct.unpack('Q', f.read(8))[0]
            name_len = struct.unpack('I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            num_pts = struct.unpack('Q', f.read(8))[0]
            pts2d = []
            for _ in range(num_pts):
                x, y = struct.unpack('dd', f.read(16))
                p3d_id = struct.unpack('Q', f.read(8))[0]
                pts2d.append({'x': x, 'y': y, 'point3D_id': p3d_id})
            images[img_id] = {
                'id': img_id,
                'qvec': np.array(qvec),
                'tvec': np.array(tvec),
                'camera_id': cam_id,
                'name': name,
                'points2D': pts2d,
            }
    return images


def read_points3d_binary(path):
    points = {}
    with open(path, 'rb') as f:
        num = struct.unpack('Q', f.read(8))[0]
        for _ in range(num):
            pid = struct.unpack('Q', f.read(8))[0]
            xyz = struct.unpack('ddd', f.read(24))
            rgb = struct.unpack('BBB', f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            desc = np.frombuffer(f.read(128), dtype=np.uint8).copy()
            track_len = struct.unpack('Q', f.read(8))[0]
            track = []
            for _ in range(track_len):
                img_id = struct.unpack('I', f.read(4))[0]
                pt2d_idx = struct.unpack('I', f.read(4))[0]
                track.append({'image_id': img_id, 'point2D_idx': pt2d_idx})
            points[pid] = {
                'id': pid,
                'xyz': np.array(xyz, dtype=np.float64),
                'rgb': np.array(rgb, dtype=np.uint8),
                'error': error,
                'descriptor': desc,
                'track': track,
            }
    return points


def load_colmap_model(sparse_dir, db_path=None):
    sparse_dir = Path(sparse_dir)
    txt_dir = sparse_dir if (sparse_dir / 'points3D.txt').exists() else None
    bin_dir = sparse_dir if (sparse_dir / 'points3D.bin').exists() else None

    # Prefer text format (version-independent, has all metadata except descriptors)
    if txt_dir:
        cameras = read_cameras_text(txt_dir / 'cameras.txt')
        images = read_images_text(txt_dir / 'images.txt')
        points3d = read_points3d_text(txt_dir / 'points3D.txt')
        model = {'cameras': cameras, 'images': images,
                 'points3d': points3d, 'format': 'text'}
        print(f"  Loaded text model: {len(cameras)} cameras, "
              f"{len(images)} images, {len(points3d)} points")

        if db_path:
            model['_db_path'] = str(db_path)

        # Try to augment with binary descriptors if available
        if bin_dir:
            try:
                bin_points = read_points3d_binary(bin_dir / 'points3D.bin')
                n_with_desc = sum(1 for v in bin_points.values()
                                   if 'descriptor' in v)
                if n_with_desc > 0:
                    for pid, pt in bin_points.items():
                        if pid in points3d and 'descriptor' in pt:
                            points3d[pid]['descriptor'] = pt['descriptor']
                    print(f"  Augmented {n_with_desc} points with descriptors "
                          f"from binary")
                    model['format'] = 'text+descriptors'
            except Exception as e:
                print(f"  Could not read binary descriptors: {e}")

        return model

    if bin_dir:
        try:
            cameras = read_cameras_binary(bin_dir / 'cameras.bin')
            images = read_images_binary(bin_dir / 'images.bin')
            points3d = read_points3d_binary(bin_dir / 'points3D.bin')
            model = {'cameras': cameras, 'images': images,
                     'points3d': points3d, 'format': 'binary'}
            if db_path:
                model['_db_path'] = str(db_path)
            print(f"  Loaded binary model: {len(cameras)} cameras, "
                  f"{len(images)} images, {len(points3d)} points")
            return model
        except Exception as e:
            print(f"  Binary model load failed: {e}")
            print(f"  Convert to text format: colmap model_converter "
                  f"--input_path {bin_dir} --output_path {bin_dir / 'txt'} "
                  f"--output_type TXT")

    raise FileNotFoundError(f"No COLMAP data found in {sparse_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  3. PER-CHAMBER COLMAP EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def fix_CHamber1_08(image_dir):
    """Fix the capital-H naming anomaly."""
    src = Path(image_dir) / 'CHamber1_08.jpg'
    dst = Path(image_dir) / 'Chamber1_08.jpg'
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
        print(f"  Fixed: CHamber1_08.jpg → Chamber1_08.jpg")
        return True
    return False


def run_colmap_feature_extraction(database_path, image_path, options=None):
    opts = options or {}
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", str(opts.get('max_features', 32768)),
        "--SiftExtraction.first_octave", "-1",
        "--SiftExtraction.peak_threshold", str(opts.get('peak_threshold', 0.02)),
        "--SiftExtraction.use_gpu", "0",
    ]
    print(f"  Running: colmap feature_extractor (CPU SIFT) ...")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Feature extraction failed (code {ret.returncode})")


def run_colmap_exhaustive_matching(database_path):
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "0",
    ]
    print(f"  Running: colmap exhaustive_matcher (CPU) ...")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Exhaustive matching failed (code {ret.returncode})")


def run_colmap_mapper(database_path, image_path, output_path, options=None):
    opts = options or {}
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--Mapper.ba_local_max_num_iterations",
        str(opts.get('ba_local_iters', 25)),
        "--Mapper.ba_global_max_num_iterations",
        str(opts.get('ba_global_iters', 50)),
        "--Mapper.multiple_models", str(opts.get('multiple_models', "1")),
        "--Mapper.max_num_models", "50",
        "--Mapper.min_model_size", str(opts.get('min_model_size', 3)),
    ]
    print(f"  Running: colmap mapper ...")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"  WARNING: mapper returned code {ret.returncode}")
    return find_best_model(output_path)


def find_best_model(sparse_path):
    sparse_path = Path(sparse_path)
    best_num = 0
    best_dir = None
    for sub in sorted(sparse_path.iterdir()):
        if not sub.is_dir():
            continue
        img_bin = sub / 'images.bin'
        if not img_bin.exists():
            continue
        with open(img_bin, 'rb') as f:
            num = struct.unpack('Q', f.read(8))[0]
        if num > best_num:
            best_num = num
            best_dir = sub
    if best_dir:
        print(f"  Best model: {best_dir.name} ({best_num} images)")
    return best_dir


def run_chamber_colmap(chamber_id, image_paths, workspace,
                       image_source_dir, options=None):
    opts = options or {}
    ws = Path(workspace)
    chamber_ws = ws / f'chamber_{chamber_id}'
    images_dir = chamber_ws / 'images'
    db_path = chamber_ws / 'database.db'
    sparse_dir = chamber_ws / 'sparse'
    model_dir = chamber_ws / 'model'

    # Skip if already done
    if model_dir.exists() and (model_dir / 'points3D.bin').exists():
        with open(model_dir / 'images.bin', 'rb') as f:
            num = struct.unpack('Q', f.read(8))[0]
        print(f"  Chamber {chamber_id}: already done ({num} images)")
        return model_dir

    # Copy images
    chamber_ws.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    for img_info in image_paths:
        src = Path(image_source_dir) / img_info['filename']
        dst = images_dir / img_info['filename']
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    n_copied = len(list(images_dir.glob('*.jpg')))
    print(f"  Chamber {chamber_id}: copied {n_copied} images")

    # Run COLMAP
    try:
        run_colmap_feature_extraction(db_path, images_dir, opts)
        run_colmap_exhaustive_matching(db_path)
        best = run_colmap_mapper(db_path, images_dir, sparse_dir, opts)

        if best is None:
            print(f"  Chamber {chamber_id}: COLMAP produced no model!")
            return None

        # Copy best model to model_dir
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(best, model_dir)

        # Also convert to text format for portable access
        txt_dir = model_dir / 'txt'
        txt_dir.mkdir(exist_ok=True)
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ])

        # Count registered images
        with open(model_dir / 'images.bin', 'rb') as f:
            num = struct.unpack('Q', f.read(8))[0]
        print(f"  Chamber {chamber_id}: registered {num}/{n_copied} images")
        return model_dir

    except Exception as e:
        print(f"  Chamber {chamber_id}: FAILED - {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  4. CROSS-CHAMBER ALIGNMENT (CAMERA-POSE + BRIDGE-POINT DESCRIPTORS)
# ═══════════════════════════════════════════════════════════════════════

def quat_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy],
    ])

def get_bridge_point_ids(model, current_chamber_id, target_chamber_id):
    """Find 3D point IDs observed by transition images showing target_chamber.

    A transition image FROM current_chamber that SHOWS target_chamber
    captures features in the transition/overlap region between the two.
    These 3D points serve as bridge points for cross-chamber alignment.
    Includes topview and angle variants for parallax-rich geometry.
    """
    bridge_pts = set()
    prefix = f'Chamber{current_chamber_id}_'
    marker = f'_Chamber{target_chamber_id}'
    for img in model['images'].values():
        name = img['name']
        # Image taken FROM current_chamber, with target_chamber in view
        if name.startswith(prefix) and marker in name:
            for pt2d in img['points2D']:
                pid = pt2d['point3D_id']
                if pid > 0 and pid < 2**63:
                    bridge_pts.add(pid)
    return sorted(bridge_pts)


def get_bridge_positions(model, bridge_point_ids):
    """Get 3D positions of bridge points in model's coordinate frame."""
    pts = []
    valid_ids = []
    for pid in bridge_point_ids:
        if pid in model['points3d']:
            pts.append(model['points3d'][pid]['xyz'])
            valid_ids.append(pid)
    return np.array(pts, dtype=np.float64), valid_ids


def get_bridge_descriptors(database_path, bridge_point_ids,
                           model_images, model_points3d):
    """Read SIFT descriptors for bridge points from COLMAP SQLite database.

    For each 3D point, averages the descriptors from all its 2D observations
    to get a robust mean descriptor (128-dim uint8).
    """
    db_path = Path(database_path)
    if not db_path.exists():
        return {}

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    descriptors = {}
    # Build image_id → name mapping from database
    cur.execute("SELECT image_id, name FROM images")
    db_images = {row[0]: row[1] for row in cur.fetchall()}

    for pid in bridge_point_ids:
        if pid not in model_points3d:
            continue
        pt = model_points3d[pid]
        track = pt.get('track', [])

        desc_list = []
        for te in track:
            db_img_id = te['image_id']
            pt2d_idx = te['point2D_idx']

            # Map model image_id → database image_id via image name
            img_name = model_images.get(db_img_id, {}).get('name', '')
            if not img_name:
                continue

            db_id = None
            for db_iid, db_name in db_images.items():
                if db_name == img_name:
                    db_id = db_iid
                    break
            if db_id is None:
                continue

            cur.execute(
                "SELECT data FROM descriptors WHERE image_id=?", (db_id,)
            )
            row = cur.fetchone()
            if row is None:
                continue

            blob = row[0]
            cur.execute(
                "SELECT rows, cols FROM descriptors WHERE image_id=?",
                (db_id,)
            )
            dims = cur.fetchone()
            if dims is None:
                continue
            rows, cols = dims
            if pt2d_idx < rows:
                desc_data = np.frombuffer(blob, dtype=np.uint8)
                start_byte = pt2d_idx * cols
                end_byte = start_byte + cols
                desc = desc_data[start_byte:end_byte]
                if len(desc) == cols:
                    desc_list.append(desc)

        if desc_list:
            descriptors[pid] = np.mean(desc_list, axis=0).astype(np.uint8)

    conn.close()
    return descriptors


def match_bridge_points(desc_X, pos_X, desc_Y, pos_Y):
    """Match bridge points between chambers using brute-force + Lowe's test.

    Returns list of (idx_X, idx_Y).
    """
    if len(desc_X) == 0 or len(desc_Y) == 0:
        return []

    keys_X = np.array(list(desc_X.keys()))
    keys_Y = np.array(list(desc_Y.keys()))
    arr_X = np.array([desc_X[k] for k in keys_X]).astype(np.float32)
    arr_Y = np.array([desc_Y[k] for k in keys_Y]).astype(np.float32)

    import cv2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(arr_X, arr_Y, k=2)

    good = []
    for m_pair in matches:
        if len(m_pair) >= 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        elif len(m_pair) == 1:
            good.append(m_pair[0])

    matched_pairs = []
    for m in good:
        idx_X = int(keys_X[m.queryIdx])
        idx_Y = int(keys_Y[m.trainIdx])
        matched_pairs.append((idx_X, idx_Y))

    return matched_pairs


def compute_rigid_transform_ransac(matched_pairs, pos_X, pos_Y, max_iters=2000):
    """RANSAC on matched 3D-3D pairs to find rigid transform (R, t).

    Returns (R, t), inlier_pairs, n_inliers.
    Returns (None, [], 0) if no valid transform found.
    """
    if len(matched_pairs) < 4:
        print(f"  Too few matches ({len(matched_pairs)}) for RANSAC")
        return None, [], 0

    pts_X = np.array([pos_X[i] for i, _ in matched_pairs])
    pts_Y = np.array([pos_Y[j] for _, j in matched_pairs])

    best_inliers = []
    best_n = 0
    best_R = None
    best_t = None
    n_pts = len(matched_pairs)

    for _ in range(max_iters):
        idx = np.random.choice(n_pts, 3, replace=False)
        pX = pts_X[idx]
        pY = pts_Y[idx]

        # Compute centroid
        cX = pX.mean(axis=0)
        cY = pY.mean(axis=0)

        H = (pX - cX).T @ (pY - cY)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cX - R @ cY

        # Count inliers
        err = np.linalg.norm(pts_X - (R @ pts_Y.T).T - t, axis=1)
        inlier_mask = err < 0.1  # 10 cm threshold
        n_inliers = inlier_mask.sum()

        if n_inliers > best_n:
            best_n = n_inliers
            best_inliers = inlier_mask
            best_R = R
            best_t = t

    if best_R is None or best_n < 4:
        return None, [], 0

    # Refine with all inliers
    inlier_idx = np.where(best_inliers)[0]
    inlier_pairs = [matched_pairs[i] for i in inlier_idx]
    pX_in = pts_X[inlier_idx]
    pY_in = pts_Y[inlier_idx]

    cX = pX_in.mean(axis=0)
    cY = pY_in.mean(axis=0)
    H = (pX_in - cX).T @ (pY_in - cY)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cX - R @ cY

    n_inliers = len(inlier_idx)
    return (R, t), inlier_pairs, n_inliers


def get_shared_transition_images(model_X, model_Y, chamber_X_id, chamber_Y_id):
    """Find transition images between chambers X and Y that registered in BOTH models."""
    names_X = {}
    for img_id, img in model_X['images'].items():
        n = img['name']
        if f'_Chamber{chamber_Y_id}' in n and n.startswith(f'Chamber{chamber_X_id}_'):
            names_X[n] = img
        elif f'_Chamber{chamber_X_id}' in n and n.startswith(f'Chamber{chamber_Y_id}_'):
            names_X[n] = img

    names_Y = {}
    for img_id, img in model_Y['images'].items():
        n = img['name']
        if f'_Chamber{chamber_Y_id}' in n and n.startswith(f'Chamber{chamber_X_id}_'):
            names_Y[n] = img
        elif f'_Chamber{chamber_X_id}' in n and n.startswith(f'Chamber{chamber_Y_id}_'):
            names_Y[n] = img

    shared = sorted(set(names_X) & set(names_Y))
    only_X = sorted(set(names_X) - set(names_Y))
    only_Y = sorted(set(names_Y) - set(names_X))
    return shared, only_X, only_Y


def compute_chamber_transform(model_X, model_Y,
                              chamber_X_id, chamber_Y_id,
                              db_path_X=None, db_path_Y=None,
                              images_source=None):
    """Hybrid alignment: camera-pose (shared images) + bridge-point descriptors.

    Aligns chamber Y → chamber X's coordinate frame using:
    1. Camera-pose of transition images that registered in BOTH models
    2. 3D bridge points triangulated from transition images showing the other
       chamber (topview/parallax variants give the best geometry)

    Returns ((R, t), matched_names) or (None, []) on failure.
    """
    pair = (min(chamber_X_id, chamber_Y_id), max(chamber_X_id, chamber_Y_id))
    shared, only_X, only_Y = get_shared_transition_images(
        model_X, model_Y, chamber_X_id, chamber_Y_id
    )

    print(f"  Transition images shared: {len(shared)}, "
          f"only in X: {len(only_X)}, only in Y: {len(only_Y)}")

    # ── Part 1: Camera-pose alignment from shared images ──
    camera_pose_transforms = []
    for name in shared:
        img_X = next(img for img in model_X['images'].values() if img['name'] == name)
        img_Y = next(img for img in model_Y['images'].values() if img['name'] == name)

        R_X = quat_to_rotmat(img_X['qvec'])
        R_Y = quat_to_rotmat(img_Y['qvec'])
        t_X = img_X['tvec']
        t_Y = img_Y['tvec']

        R = R_X @ R_Y.T
        t = t_X - R @ t_Y
        camera_pose_transforms.append((R, t))

    if camera_pose_transforms:
        Rs = np.array([T[0] for T in camera_pose_transforms])
        ts = np.array([T[1] for T in camera_pose_transforms])
        cam_R = np.median(Rs.reshape(len(Rs), -1), axis=0).reshape(3, 3)
        U, _, Vt = np.linalg.svd(cam_R)
        cam_R = U @ Vt
        cam_t = np.median(ts, axis=0)
        print(f"  Camera-pose alignment: {len(camera_pose_transforms)} shared images")
    else:
        cam_R, cam_t = None, None

    # ── Part 2: Bridge-point descriptor matching ──
    bridge_R, bridge_t = None, None
    bridge_inliers = 0
    bridge_X_ids, bridge_X_desc = None, None
    bridge_Y_ids, bridge_Y_desc = None, None
    bridge_matched_names = []

    for db_path, model, only_set, cid_from, cid_to, label in [
        (db_path_X, model_X, only_X + shared, chamber_X_id, chamber_Y_id, 'X→Y'),
        (db_path_Y, model_Y, only_Y + shared, chamber_Y_id, chamber_X_id, 'Y→X'),
    ]:
        if not only_set or db_path is None:
            continue
        if not Path(db_path).exists():
            print(f"  Database not found: {db_path}, skipping bridge points for {label}")
            continue

        print(f"  Extracting bridge points ({label})...")
        b_ids = get_bridge_point_ids(model, cid_from, cid_to)
        if len(b_ids) < 5:
            print(f"    Too few bridge points: {len(b_ids)}")
            continue

        descs = get_bridge_descriptors(db_path, b_ids, model['images'], model['points3d'])
        if len(descs) < 5:
            print(f"    Too few with descriptors: {len(descs)}")
            continue

        if label == 'X→Y':
            bridge_X_ids, bridge_X_desc = b_ids, descs
        else:
            bridge_Y_ids, bridge_Y_desc = b_ids, descs

    # Match bridge points across chambers
    if (bridge_X_desc is not None and bridge_Y_desc is not None
            and len(bridge_X_desc) >= 5 and len(bridge_Y_desc) >= 5):
        pos_X, _ = get_bridge_positions(model_X, bridge_X_ids)
        pos_Y, _ = get_bridge_positions(model_Y, bridge_Y_ids)

        # Build position lookup by point ID
        pos_X_dict = {bridge_X_ids[i]: pos_X[i] for i in range(len(bridge_X_ids))}
        pos_Y_dict = {bridge_Y_ids[i]: pos_Y[i] for i in range(len(bridge_Y_ids))}

        matched_pairs = match_bridge_points(
            bridge_X_desc, pos_X_dict,
            bridge_Y_desc, pos_Y_dict
        )
        print(f"  Descriptor matches: {len(matched_pairs)}")

        if len(matched_pairs) >= 4:
            result = compute_rigid_transform_ransac(matched_pairs, pos_X_dict, pos_Y_dict)
            if result[0] is not None:
                (bridge_R, bridge_t), inlier_pairs, bridge_inliers = result
                print(f"  Bridge-point RANSAC: {bridge_inliers} inliers")
                # Extract names of matched images for reporting
                bridge_matched_names = []
                for pid_X, pid_Y in inlier_pairs:
                    # Find which images observe these points
                    for img in model_X['images'].values():
                        for pt2d in img['points2D']:
                            if pt2d['point3D_id'] == pid_X:
                                bridge_matched_names.append(img['name'])
                                break
                        else:
                            continue
                        break
                bridge_matched_names = sorted(set(bridge_matched_names))
            else:
                print(f"  Bridge-point RANSAC: failed")
                bridge_matched_names = []
        else:
            bridge_matched_names = []
    else:
        bridge_matched_names = []

    # ── Part 3: Combine both estimates ──
    all_transforms = []
    all_names = []

    if cam_R is not None:
        all_transforms.append((cam_R, cam_t))
        all_names.extend(shared)

    if bridge_R is not None:
        all_transforms.append((bridge_R, bridge_t))
        all_names.extend(bridge_matched_names)

    if not all_transforms:
        # Fall back: try using only unshared transition images via PnP
        print("  No transforms from either method. Trying SIFT+PnP on unshared images...")
        if images_source is not None and (only_X or only_Y):
            result = _align_via_pnp(
                model_X, model_Y, only_X, only_Y,
                chamber_X_id, chamber_Y_id, images_source
            )
            if result[0] is not None:
                all_transforms.append(result[0])
                all_names.extend(result[1])

    if not all_transforms:
        print(f"  ❌ No transform found between chambers {pair}")
        return None, []

    # Robust combination via median
    Rs = np.array([T[0] for T in all_transforms])
    ts = np.array([T[1] for T in all_transforms])
    mean_R = np.median(Rs.reshape(len(Rs), -1), axis=0).reshape(3, 3)
    U, _, Vt = np.linalg.svd(mean_R)
    R_final = U @ Vt
    t_final = np.median(ts, axis=0)

    print(f"  ✅ Final: camera-pose={len(camera_pose_transforms)}, "
          f"bridge={bridge_inliers}, total={len(all_transforms)} estimates")
    return (R_final, t_final), list(set(all_names))


def _align_via_pnp(model_X, model_Y, only_X, only_Y,
                   chamber_X_id, chamber_Y_id, images_source):
    """Fallback: use SIFT feature matching + PnP on unshared transition images."""
    import cv2
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    transforms = []

    for name in only_X:
        img_path = Path(images_source) / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        kp, desc = sift.detectAndCompute(img, None)
        if desc is None or len(kp) < 10:
            continue

        # Try to match against Y's registered images that have 3D points
        pts_2d = []
        pts_3d = []
        for img_data in model_Y['images'].values():
            y_name = img_data['name']
            y_path = Path(images_source) / y_name
            if not y_path.exists():
                continue
            # Check if this Y-image has 3D points
            y_pts3d = [p for p in img_data['points2D']
                       if p['point3D_id'] > 0 and p['point3D_id'] < 2**63]
            if len(y_pts3d) < 10:
                continue

            img_y = cv2.imread(str(y_path))
            if img_y is None:
                continue
            kp_y, desc_y = sift.detectAndCompute(img_y, None)
            if desc_y is None or len(kp_y) < 10:
                continue

            matches = bf.match(desc, desc_y)
            for m in matches:
                y_kp = kp_y[m.trainIdx]
                yx, yy = y_kp.pt
                best_dist = 5.0
                best_pt = None
                for pt2d in img_data['points2D']:
                    if pt2d['point3D_id'] > 0 and pt2d['point3D_id'] < 2**63:
                        d = np.hypot(pt2d['x'] - yx, pt2d['y'] - yy)
                        if d < best_dist:
                            best_dist = d
                            best_pt = pt2d
                if best_pt is not None:
                    pts_2d.append(kp[m.queryIdx].pt)
                    pts_3d.append(
                        model_Y['points3d'][best_pt['point3D_id']]['xyz']
                    )

        if len(pts_2d) >= 8:
            pts_2d = np.array(pts_2d, dtype=np.float32)
            pts_3d = np.array(pts_3d, dtype=np.float32)
            # Get camera intrinsics from X's model
            cam = list(model_X['cameras'].values())[0]
            fx = cam['params'][0]
            cx = cam['params'][1]
            cy = cam['params'][2]
            K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
            _, rvec, tvec, _ = cv2.solvePnPRansac(
                pts_3d, pts_2d, K, None, iterationsCount=2000,
                reprojectionError=8.0, confidence=0.99
            )
            if rvec is not None:
                R_Y, _ = cv2.Rodrigues(rvec)
                # This is the camera pose of the X-image in Y's frame
                # Now get its pose in X's frame
                img_X = next((img for img in model_X['images'].values()
                              if img['name'] == name), None)
                if img_X is not None:
                    R_X = quat_to_rotmat(img_X['qvec'])
                    t_X = img_X['tvec']
                    # R_Y→X = R_X @ R_Y.T, t = t_X - R_Y→X @ tvec
                    R = R_X @ R_Y.T
                    t = t_X - R @ tvec.flatten()
                    transforms.append((R, t))

    if not transforms:
        return (None, [])

    Rs = np.array([T[0] for T in transforms])
    ts = np.array([T[1] for T in transforms])
    R = np.median(Rs.reshape(len(Rs), -1), axis=0).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = np.median(ts, axis=0)
    return (R, t), []


def format_transform(R, t):
    return (f"Rotation:\n{np.array2string(R, precision=6, suppress_small=True)}\n"
            f"Translation:\n{np.array2string(t, precision=6, suppress_small=True)}")


# ═══════════════════════════════════════════════════════════════════════
#  5. UNIFIED MODEL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def apply_transform_to_model(model, R, t):
    model = model.copy()
    points3d = {}
    for pid, pt in model['points3d'].items():
        pt = dict(pt)
        pt['xyz'] = R @ pt['xyz'] + t
        points3d[pid] = pt
    model['points3d'] = points3d

    images = {}
    for img_id, img in model['images'].items():
        img = dict(img)
        R_img = quat_to_rotmat(img['qvec'])
        t_img = img['tvec']

        # Apply transform: new_R = R @ R_img, new_t = R @ t_img + t
        new_R = R @ R_img
        new_t = R @ t_img + t

        # Convert back to quaternion
        trace = new_R[0,0] + new_R[1,1] + new_R[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            img['qvec'] = np.array([0.25 / s, (new_R[2,1] - new_R[1,2]) * s,
                                     (new_R[0,2] - new_R[2,0]) * s,
                                     (new_R[1,0] - new_R[0,1]) * s])
        elif new_R[0,0] > new_R[1,1] and new_R[0,0] > new_R[2,2]:
            s = 2.0 * np.sqrt(1.0 + new_R[0,0] - new_R[1,1] - new_R[2,2])
            img['qvec'] = np.array([(new_R[2,1] - new_R[1,2]) / s,
                                     0.25 * s,
                                     (new_R[0,1] + new_R[1,0]) / s,
                                     (new_R[0,2] + new_R[2,0]) / s])
        elif new_R[1,1] > new_R[2,2]:
            s = 2.0 * np.sqrt(1.0 + new_R[1,1] - new_R[0,0] - new_R[2,2])
            img['qvec'] = np.array([(new_R[0,2] - new_R[2,0]) / s,
                                     (new_R[0,1] + new_R[1,0]) / s,
                                     0.25 * s,
                                     (new_R[1,2] + new_R[2,1]) / s])
        else:
            s = 2.0 * np.sqrt(1.0 + new_R[2,2] - new_R[0,0] - new_R[1,1])
            img['qvec'] = np.array([(new_R[1,0] - new_R[0,1]) / s,
                                     (new_R[0,2] + new_R[2,0]) / s,
                                     (new_R[1,2] + new_R[2,1]) / s,
                                     0.25 * s])
        img['tvec'] = new_t
        images[img_id] = img
    model['images'] = images

    return model


def merge_models_into_unified(chamber_models, transforms, base_chamber=1):
    unified = {
        'cameras': {},
        'images': {},
        'points3d': {},
    }
    image_offset = 0
    point_offset = 0
    next_cam_id = 1
    next_img_id = 1
    next_pt_id = 1
    id_map = {}

    sorted_chambers = sorted(chamber_models.keys())
    for cid in sorted_chambers:
        model = chamber_models[cid]
        if cid == base_chamber:
            R = np.eye(3)
            t = np.zeros(3)
        else:
            if cid not in transforms:
                print(f"  ERROR: No transform for Chamber {cid}")
                continue
            R, t = transforms[cid]

        # Remap IDs to avoid collisions
        for cam_id, cam in model['cameras'].items():
            unified['cameras'][next_cam_id] = dict(cam)
            if 'id' in unified['cameras'][next_cam_id]:
                unified['cameras'][next_cam_id]['id'] = next_cam_id
            id_map.setdefault('cam', {})[cam_id] = next_cam_id
            next_cam_id += 1

        for img_id, img in model['images'].items():
            img = dict(img)
            img['id'] = next_img_id
            img['camera_id'] = id_map['cam'].get(img['camera_id'],
                                                  img['camera_id'])
            # Transform camera pose
            R_img = quat_to_rotmat(img['qvec'])
            new_R = R @ R_img
            new_t = R @ img['tvec'] + t
            trace = new_R[0,0] + new_R[1,1] + new_R[2,2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                img['qvec'] = np.array([0.25 / s, (new_R[2,1] - new_R[1,2]) * s,
                                         (new_R[0,2] - new_R[2,0]) * s,
                                         (new_R[1,0] - new_R[0,1]) * s])
            elif new_R[0,0] > new_R[1,1] and new_R[0,0] > new_R[2,2]:
                s = 2.0 * np.sqrt(1.0 + new_R[0,0] - new_R[1,1] - new_R[2,2])
                img['qvec'] = np.array([(new_R[2,1] - new_R[1,2]) / s, 0.25 * s,
                                          (new_R[0,1] + new_R[1,0]) / s,
                                          (new_R[0,2] + new_R[2,0]) / s])
            elif new_R[1,1] > new_R[2,2]:
                s = 2.0 * np.sqrt(1.0 + new_R[1,1] - new_R[0,0] - new_R[2,2])
                img['qvec'] = np.array([(new_R[0,2] - new_R[2,0]) / s,
                                          (new_R[0,1] + new_R[1,0]) / s, 0.25 * s,
                                          (new_R[1,2] + new_R[2,1]) / s])
            else:
                s = 2.0 * np.sqrt(1.0 + new_R[2,2] - new_R[0,0] - new_R[1,1])
                img['qvec'] = np.array([(new_R[1,0] - new_R[0,1]) / s,
                                          (new_R[0,2] + new_R[2,0]) / s,
                                          (new_R[1,2] + new_R[2,1]) / s, 0.25 * s])
            img['tvec'] = new_t
            img['name'] = f"Chamber{cid}_" + img['name'].split('_', 1)[-1] \
                          if '_' in img['name'] else img['name']

            # Update point2D references
            for pt2d in img['points2D']:
                old_p3d = pt2d['point3D_id']
                if old_p3d > 0 and old_p3d < 2**63:
                    new_p3d = id_map.get('pt', {}).get(old_p3d, old_p3d + point_offset)
                    pt2d['point3D_id'] = new_p3d

            unified['images'][next_img_id] = img
            id_map.setdefault('img', {})[img_id] = next_img_id
            next_img_id += 1

        for pid, pt in model['points3d'].items():
            pt = dict(pt)
            pt['xyz'] = R @ pt['xyz'] + t
            pt['id'] = next_pt_id

            new_track = []
            for te in pt.get('track', []):
                old_img_id = te['image_id']
                new_img_id = id_map.get('img', {}).get(old_img_id,
                                                        old_img_id + image_offset)
                new_track.append({
                    'image_id': new_img_id,
                    'point2D_idx': te['point2D_idx'],
                })
            pt['track'] = new_track
            unified['points3d'][next_pt_id] = pt
            id_map.setdefault('pt', {})[pid] = next_pt_id
            next_pt_id += 1

    print(f"\nUnified model: {len(unified['cameras'])} cameras, "
          f"{len(unified['images'])} images, "
          f"{len(unified['points3d'])} points")
    return unified


def export_colmap_model_text(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cameras
    with open(output_dir / 'cameras.txt', 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(model['cameras'])}\n")
        for cam_id, cam in model['cameras'].items():
            params = ' '.join(f'{p:.6f}' for p in cam['params'])
            f.write(f"{cam_id} {cam.get('model', 'SIMPLE_RADIAL')} "
                    f"{cam['width']} {cam['height']} {params}\n")

    # Images
    with open(output_dir / 'images.txt', 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(model['images'])}, "
                f"mean observations per image: ...\n")
        for img_id in sorted(model['images'].keys()):
            img = model['images'][img_id]
            q = img['qvec']
            t = img['tvec']
            f.write(f"{img_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} "
                    f"{q[3]:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} "
                    f"{img['camera_id']} {img['name']}\n")
            pts_str = ' '.join(
                f"{pt['x']:.4f} {pt['y']:.4f} {pt['point3D_id']}"
                for pt in img['points2D']
            )
            f.write(f" {pts_str}\n")

    # Points3D
    with open(output_dir / 'points3D.txt', 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(model['points3d'])}, "
                f"mean track length: ...\n")
        for pid in sorted(model['points3d'].keys()):
            pt = model['points3d'][pid]
            track_str = ' '.join(
                f"{te['image_id']} {te['point2D_idx']}"
                for te in pt.get('track', [])
            )
            f.write(f"{pid} {pt['xyz'][0]:.6f} {pt['xyz'][1]:.6f} "
                    f"{pt['xyz'][2]:.6f} {pt['rgb'][0]} {pt['rgb'][1]} "
                    f"{pt['rgb'][2]} {pt.get('error', 1.0):.6f} "
                    f"{track_str}\n")

    print(f"  Exported COLMAP model to {output_dir}")
    return output_dir


def export_colmap_model_binary(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cameras binary
    with open(output_dir / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('Q', len(model['cameras'])))
        for cam_id, cam in model['cameras'].items():
            f.write(struct.pack('Q', cam_id))
            model_id = {'SIMPLE_PINHOLE': 0, 'PINHOLE': 1,
                        'SIMPLE_RADIAL': 2, 'RADIAL': 3}.get(
                            cam.get('model', 'SIMPLE_RADIAL'), 2)
            f.write(struct.pack('I', model_id))
            f.write(struct.pack('Q', cam['width']))
            f.write(struct.pack('Q', cam['height']))
            for p in cam['params']:
                f.write(struct.pack('d', p))

    # Images binary
    with open(output_dir / 'images.bin', 'wb') as f:
        f.write(struct.pack('Q', len(model['images'])))
        for img_id in sorted(model['images'].keys()):
            img = model['images'][img_id]
            f.write(struct.pack('Q', img_id))
            for v in img['qvec']:
                f.write(struct.pack('d', v))
            for v in img['tvec']:
                f.write(struct.pack('d', v))
            f.write(struct.pack('Q', img['camera_id']))
            name_bytes = img['name'].encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('Q', len(img['points2D'])))
            for pt in img['points2D']:
                f.write(struct.pack('d', pt['x']))
                f.write(struct.pack('d', pt['y']))
                f.write(struct.pack('Q', pt['point3D_id']))

    # Points3D binary
    with open(output_dir / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('Q', len(model['points3d'])))
        for pid in sorted(model['points3d'].keys()):
            pt = model['points3d'][pid]
            f.write(struct.pack('Q', pid))
            for v in pt['xyz']:
                f.write(struct.pack('d', v))
            f.write(struct.pack('BBB', *pt['rgb']))
            f.write(struct.pack('d', pt.get('error', 1.0)))
            if 'descriptor' in pt and pt['descriptor'] is not None:
                f.write(pt['descriptor'].tobytes())
            else:
                f.write(b'\x00' * 128)
            f.write(struct.pack('Q', len(pt.get('track', []))))
            for te in pt.get('track', []):
                f.write(struct.pack('I', te['image_id']))
                f.write(struct.pack('I', te['point2D_idx']))

    print(f"  Exported COLMAP binary model to {output_dir}")
    return output_dir


# ═══════════════════════════════════════════════════════════════════════
#  6. 3DGS TRAINING WRAPPER
# ═══════════════════════════════════════════════════════════════════════

def setup_3dgs_input(colmap_model, images_source, output_dir):
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    sparse_dir = output_dir / 'sparse' / '0'
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Export COLMAP model as text
    export_colmap_model_text(colmap_model, sparse_dir)

    # Copy images
    images_source = Path(images_source)
    copied = 0
    for img_id, img in colmap_model['images'].items():
        src = images_source / img['name']
        dst = images_dir / img['name']
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1

    print(f"  Setup 3DGS input: {copied} images, {sparse_dir}")
    return output_dir


def train_unified_3dgs(input_dir, output_dir, iterations=30000,
                       max_gaussians=500000):
    """Train a single 3DGS model on the unified data."""
    sys.path.insert(0, str(BASE / 'scripts'))
    from train_3dgs_enhanced import train

    import argparse
    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        iterations=iterations,
        max_gaussians=max_gaussians,
        log_interval=1000,
    )
    train(args)


# ═══════════════════════════════════════════════════════════════════════
#  7. CHECKPOINT / SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(state_path, data):
    with open(state_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Checkpoint saved: {state_path}")


def load_checkpoint(state_path):
    state_path = Path(state_path)
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


if __name__ == '__main__':
    # Quick test: analyze the dataset
    by_chamber, transitions, all_parsed = categorize_images(
        BASE / 'splat-files-processed'
    )
    print_dataset_summary(by_chamber, transitions)
