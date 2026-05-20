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
import os, sys, re, struct, time, json, sqlite3, shutil, warnings
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
        'has_target': m.group(2) is not None,
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


def extract_descriptors_from_db(database_path, bridge_point_ids,
                                 model_images, model_points3d):
    """Extract SIFT descriptors from COLMAP SQLite database for bridge points.

    For each 3D point, averages the descriptors from all its observations
    to get a robust mean descriptor.
    """
    db_path = Path(database_path)
    if not db_path.exists():
        print(f"  Database not found: {db_path}")
        return {}

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    descriptors = {}
    # Build image_id → name mapping from database
    cur.execute("SELECT image_id, name FROM images")
    db_images = {row[0]: row[1] for row in cur.fetchall()}

    # Build name → image_id mapping from model
    name_to_model_id = {img['name']: img_id
                         for img_id, img in model_images.items()}

    for pid in bridge_point_ids:
        if pid not in model_points3d:
            continue
        pt = model_points3d[pid]
        track = pt.get('track', [])

        desc_list = []
        for te in track:
            db_img_id = te['image_id']
            pt2d_idx = te['point2D_idx']

            # Get database image_id from model image name
            img_name = model_images.get(db_img_id, {}).get('name', '')
            if not img_name:
                continue

            # Find corresponding image in database
            db_id = None
            for db_iid, db_name in db_images.items():
                if db_name == img_name:
                    db_id = db_iid
                    break
            if db_id is None:
                continue

            # Read descriptor blob
            cur.execute(
                "SELECT data FROM descriptors WHERE image_id=?",
                (db_id,)
            )
            row = cur.fetchone()
            if row is None:
                continue

            blob = row[0]
            # Descriptors are stored as rows×128 uint8 array
            desc_data = np.frombuffer(blob, dtype=np.uint8)
            # We need rows and cols
            cur.execute(
                "SELECT rows, cols FROM descriptors WHERE image_id=?",
                (db_id,)
            )
            dims = cur.fetchone()
            if dims is None:
                continue
            rows, cols = dims
            if pt2d_idx < rows:
                start = pt2d_idx * cols
                desc = desc_data[start:start + cols]
                if len(desc) == cols:
                    desc_list.append(desc)

        if desc_list:
            descriptors[pid] = np.mean(desc_list, axis=0).astype(np.uint8)

    conn.close()
    return descriptors


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
    ]
    print(f"  Running: colmap feature_extractor ...")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Feature extraction failed (code {ret.returncode})")


def run_colmap_exhaustive_matching(database_path):
    cmd = ["colmap", "exhaustive_matcher", "--database_path", str(database_path)]
    print(f"  Running: colmap exhaustive_matcher ...")
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
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "50",
        "--Mapper.min_model_size", "3",
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
        os.system(
            f"colmap model_converter "
            f"--input_path {model_dir} "
            f"--output_path {txt_dir} "
            f"--output_type TXT"
        )

        # Count registered images
        with open(model_dir / 'images.bin', 'rb') as f:
            num = struct.unpack('Q', f.read(8))[0]
        print(f"  Chamber {chamber_id}: registered {num}/{n_copied} images")
        return model_dir

    except Exception as e:
        print(f"  Chamber {chamber_id}: FAILED - {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  4. CROSS-CHAMBER ALIGNMENT (DESCRIPTOR MATCHING + RANSAC)
# ═══════════════════════════════════════════════════════════════════════

def get_bridge_point_ids(model, transition_image_names):
    images = model['images']
    points3d = model['points3d']

    # Find image IDs for transition images
    trans_img_ids = set()
    for img_id, img in images.items():
        if img['name'] in transition_image_names:
            trans_img_ids.add(img_id)

    if not trans_img_ids:
        print("  WARNING: No transition images found in model")
        return set(), {}

    # Collect 3D point IDs visible in these transition images
    bridge_point_ids = set()
    point_to_images = defaultdict(list)
    for img_id in trans_img_ids:
        for pt in images[img_id]['points2D']:
            p3d_id = pt['point3D_id']
            if p3d_id >= 2**63:  # INVALID sentinel (very large number)
                continue
            if p3d_id > 0:
                bridge_point_ids.add(p3d_id)
                point_to_images[p3d_id].append(img_id)

    print(f"  Bridge points: {len(bridge_point_ids)} "
          f"from {len(trans_img_ids)} transition images")
    return bridge_point_ids, point_to_images


def get_bridge_descriptors(model, bridge_point_ids):
    points3d = model['points3d']
    descriptors = {}
    for pid in bridge_point_ids:
        if pid in points3d:
            pt = points3d[pid]
            if 'descriptor' in pt and pt['descriptor'] is not None:
                descriptors[pid] = pt['descriptor']
    return descriptors


def get_bridge_positions(model, bridge_point_ids):
    points3d = model['points3d']
    positions = {}
    for pid in bridge_point_ids:
        if pid in points3d:
            positions[pid] = points3d[pid]['xyz']
    return positions


def match_bridge_points(desc_X, pos_X, desc_Y, pos_Y,
                        ratio_threshold=0.75, min_matches=6):
    desc_X_ids = np.array(list(desc_X.keys()))
    desc_Y_ids = np.array(list(desc_Y.keys()))

    if len(desc_X_ids) == 0 or len(desc_Y_ids) == 0:
        print("  No descriptors to match")
        return []

    mat_X = np.array([desc_X[pid] for pid in desc_X_ids], dtype=np.float32)
    mat_Y = np.array([desc_Y[pid] for pid in desc_Y_ids], dtype=np.float32)

    # L2 distance matrix (brute force)
    dists = np.zeros((len(mat_X), len(mat_Y)), dtype=np.float32)
    for i in range(len(mat_X)):
        diff = mat_X[i:i+1] - mat_Y
        dists[i] = np.sqrt((diff * diff).sum(axis=1))

    matches = []
    for i in range(len(mat_X)):
        idx = np.argsort(dists[i])
        best = dists[i, idx[0]]
        second = dists[i, idx[1]] if len(idx) > 1 else float('inf')
        if best < ratio_threshold * second:  # Lowe's ratio test (L2 distance)
            x_id = desc_X_ids[i]
            y_id = desc_Y_ids[idx[0]]
            matches.append((x_id, y_id, float(best)))

    matches.sort(key=lambda m: m[2])  # Sort by ascending distance
    print(f"  Found {len(matches)} candidate matches (ratio test, L2)")
    return matches


def compute_rigid_transform_ransac(matches, pos_X, pos_Y,
                                   max_iters=5000, inlier_threshold=0.05,
                                   min_inliers=6):
    if len(matches) < min_inliers:
        print(f"  Too few matches ({len(matches)}), need {min_inliers}")
        return None, [], []

    pts_X = np.array([pos_X[m[0]] for m in matches])  # (N, 3)
    pts_Y = np.array([pos_Y[m[1]] for m in matches])  # (N, 3)

    N = len(pts_X)
    best_inliers = []
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_error = float('inf')

    if N < 3:
        return None, [], []

    rng = np.random.RandomState(42)

    for it in range(max_iters):
        # Sample 3 random correspondences
        if N > 3:
            sample = rng.choice(N, 3, replace=False)
        else:
            sample = np.arange(N)

        p_X = pts_X[sample]
        p_Y = pts_Y[sample]

        c_X = p_X.mean(axis=0)
        c_Y = p_Y.mean(axis=0)

        # Kabsch: H = X_c.T @ Y_c gives R_right where X ≈ Y @ R_right + t
        H = (p_X - c_X).T @ (p_Y - c_Y)
        U, _, Vt = np.linalg.svd(H)
        R_right = Vt.T @ U.T
        if np.linalg.det(R_right) < 0:
            U[:, -1] *= -1
            R_right = Vt.T @ U.T
        # Convert to left-multiply: X ≈ R @ Y + t
        R = R_right.T
        t = c_X - R @ c_Y

        # Evaluate all matches
        diff = pts_X - (R @ pts_Y.T).T - t
        errors = np.linalg.norm(diff, axis=1)
        inliers = np.where(errors < inlier_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_t = t
            best_error = errors[inliers].mean() if len(inliers) > 0 else float('inf')

        if len(inliers) >= N * 0.8:
            break

    if len(best_inliers) < min_inliers:
        print(f"  RANSAC failed: {len(best_inliers)} inliers (need {min_inliers})")
        return None, [], []

    # Refine with all inliers
    inlier_pts_X = pts_X[best_inliers]
    inlier_pts_Y = pts_Y[best_inliers]
    c_X = inlier_pts_X.mean(axis=0)
    c_Y = inlier_pts_Y.mean(axis=0)
    H = (inlier_pts_X - c_X).T @ (inlier_pts_Y - c_Y)
    U, _, Vt = np.linalg.svd(H)
    R_right = Vt.T @ U.T
    if np.linalg.det(R_right) < 0:
        U[:, -1] *= -1
        R_right = Vt.T @ U.T
    R_refined = R_right.T
    t_refined = c_X - R_refined @ c_Y

    matched_pairs = [(matches[i][0], matches[i][1]) for i in best_inliers]
    print(f"  RANSAC: {len(best_inliers)}/{N} inliers, "
          f"mean error={best_error:.4f}")
    return (R_refined, t_refined), matched_pairs, best_inliers


def compute_chamber_transform(model_X, model_Y,
                              chamber_X_id, chamber_Y_id):
    pair = (min(chamber_X_id, chamber_Y_id), max(chamber_X_id, chamber_Y_id))

    # Determine transition image names
    if chamber_X_id < chamber_Y_id:
        fwd_name_pat = f'Chamber{chamber_X_id}_'
        fwd_name_suffix = f'_Chamber{chamber_Y_id}.jpg'
        rev_name_pat = f'Chamber{chamber_Y_id}_'
        rev_name_suffix = f'_Chamber{chamber_X_id}.jpg'
    else:
        fwd_name_pat = f'Chamber{chamber_X_id}_'
        fwd_name_suffix = f'_Chamber{chamber_Y_id}.jpg'
        rev_name_pat = f'Chamber{chamber_Y_id}_'
        rev_name_suffix = f'_Chamber{chamber_X_id}.jpg'

    # Images going FROM X to Y (showing Y's space, in X's model)
    fwd_names = set()
    for img_id, img in model_X['images'].items():
        if img['name'].endswith(fwd_name_suffix) and \
           fwd_name_pat in img['name']:
            fwd_names.add(img['name'])

    # Images going FROM Y to X (showing X's space, in Y's model)
    rev_names = set()
    for img_id, img in model_Y['images'].items():
        if img['name'].endswith(rev_name_suffix) and \
           rev_name_pat in img['name']:
            rev_names.add(img['name'])

    if not fwd_names:
        print(f"  No transition images from Chamber{chamber_X_id}→Chamber{chamber_Y_id}")
        fwd_names = set()
    if not rev_names:
        print(f"  No transition images from Chamber{chamber_Y_id}→Chamber{chamber_X_id}")
        rev_names = set()

    if not fwd_names and not rev_names:
        print(f"  No transition images between {pair}")
        return None, []

    # Extract bridge points
    bridge_X, _ = get_bridge_point_ids(model_X, fwd_names)
    bridge_Y, _ = get_bridge_point_ids(model_Y, rev_names)

    # Get descriptors and positions
    desc_X = get_bridge_descriptors(model_X, bridge_X)
    desc_Y = get_bridge_descriptors(model_Y, bridge_Y)
    pos_X = get_bridge_positions(model_X, bridge_X)
    pos_Y = get_bridge_positions(model_Y, bridge_Y)

    # Print dataset format info
    fmt_X = model_X.get('format', 'unknown')
    fmt_Y = model_Y.get('format', 'unknown')
    has_desc_X = len(desc_X) > 0
    has_desc_Y = len(desc_Y) > 0
    print(f"  Model X format: {fmt_X}, has descriptors: {has_desc_X} "
          f"({len(bridge_X)} bridge points, {len(desc_X)} with desc)")
    print(f"  Model Y format: {fmt_Y}, has descriptors: {has_desc_Y} "
          f"({len(bridge_Y)} bridge points, {len(desc_Y)} with desc)")

    if not has_desc_X or not has_desc_Y:
        print("  Descriptors not in model — trying COLMAP SQLite database...")
        for model, bridge, label in [(model_X, bridge_X, "X"),
                                      (model_Y, bridge_Y, "Y")]:
            model_db_path = model.get('_db_path', '')
            if model_db_path:
                db_path = Path(model_db_path)
            else:
                db_path = Path('database.db')
            if db_path.exists():
                db_desc = extract_descriptors_from_db(
                    db_path, bridge, model['images'], model['points3d']
                )
                if label == "X":
                    desc_X.update(db_desc)
                    print(f"  Database gave X: {len(db_desc)} descriptors")
                else:
                    desc_Y.update(db_desc)
                    print(f"  Database gave Y: {len(db_desc)} descriptors")
            else:
                print(f"  Database not found: {db_path}")

    if len(desc_X) < 3 or len(desc_Y) < 3:
        print(f"  Cannot match: too few descriptors ({len(desc_X)} vs {len(desc_Y)})")
        return None, []

    # Match descriptors
    matches = match_bridge_points(desc_X, pos_X, desc_Y, pos_Y)
    if len(matches) < 6:
        print(f"  Too few matches ({len(matches)}) for alignment")
        return None, []

    # RANSAC
    result = compute_rigid_transform_ransac(matches, pos_X, pos_Y)
    if result[0] is None:
        return None, []

    (R, t), matched_pairs, inliers = result
    return (R, t), matched_pairs


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
        qw, qx, qy, qz = img['qvec']
        R_img = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
            [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
            [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy],
        ])
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
    camera_offset = 0
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
            qw, qx, qy, qz = img['qvec']
            R_img = np.array([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
                [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
                [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy],
            ])
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
