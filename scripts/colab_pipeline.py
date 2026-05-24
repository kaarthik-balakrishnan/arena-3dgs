import os, json, datetime, struct, sqlite3, glob, shutil, subprocess, sys, atexit, urllib.request
from pathlib import Path

INPUT_DIR = "/content/gaussian-splatting/input"
COLMAP_DIR = "/content/gaussian-splatting/sparse"
MERGED_DIR = "/content/gaussian-splatting/sparse_merged"
SPARSE_DIR = os.path.join(INPUT_DIR, "sparse", "0")
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
SCRIPTS_DIR = "/content/scripts"
DB_PATH = os.path.join(COLMAP_DIR, "database.db")

REPO_URL = "https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main"
GITHUB_REPO = "kaarthik-balakrishnan/arena-3dgs"
EXPECTED_IMAGES = 91


def ensure_virtual_display():
    DISPLAY_NUM = 99
    os.environ["DISPLAY"] = f":{DISPLAY_NUM}"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    LOCK_FILE = f"/tmp/.X{DISPLAY_NUM}-lock"
    if not os.path.exists(LOCK_FILE):
        xvfb_proc = subprocess.Popen(
            ["Xvfb", f":{DISPLAY_NUM}", "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        atexit.register(lambda: xvfb_proc.terminate())


def colmap_available():
    return bool(os.popen("which colmap 2>/dev/null").read().strip())


def check_colmap():
    if not colmap_available():
        raise RuntimeError("colmap not found. Run dependencies cell first.")


def install_dependencies(session):
    if session.is_step_done("deps_installed") and colmap_available():
        print("Dependencies already installed (session state). Skipping.")
        ensure_virtual_display()
        return

    if session.is_step_done("deps_installed"):
        print("Session says deps installed but colmap not found. Re-installing.")

    DRV = session.checkpoint_path("deps_installed")
    colmap_missing = not colmap_available()
    if not os.path.exists(DRV) or colmap_missing:
        print("[1/4] Installing COLMAP + display deps...")
        os.system("apt-get update -qq && apt-get install -y -qq colmap xvfb libgl1-mesa-glx libglib2.0-0")
        os.system("colmap version 2>&1 | head -1")

        print("[2/4] Installing PyTorch...")
        os.system("pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 -q")
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        else:
            print(f"  PyTorch {torch.__version__}, CUDA: False")

        print("[3/4] Installing Python packages...")
        os.system("pip install plyfile numpy pillow opencv-python-headless tqdm gsplat scipy -q")

        print("[4/4] Verifying GPU...")
        Path(DRV).touch()
        print("\nSystem deps installed!")
    else:
        print("System deps already installed (Drive marker found).")

    ENHANCED_PY = os.path.join(SCRIPTS_DIR, "train_3dgs_enhanced.py")
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    if not os.path.exists(ENHANCED_PY):
        print("Downloading enhanced training script from GitHub...")
        url = f"{REPO_URL}/scripts/train_3dgs_enhanced.py"
        urllib.request.urlretrieve(url, ENHANCED_PY)
        print("  Done.")
    else:
        print("Enhanced training script already downloaded.")
    sys.path.insert(0, SCRIPTS_DIR)

    ensure_virtual_display()
    session.mark_step("deps_installed")
    print("\nAll dependencies ready!")


def download_images(session):
    import requests

    if session.is_step_done("images_downloaded"):
        print("Images already downloaded. Verifying files are present...")

    os.makedirs(INPUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(existing) >= EXPECTED_IMAGES:
        print(f"{len(existing)} images already present.")
        session.set_param("num_images", len(existing))
        session.mark_step("images_downloaded")
        return

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/splat-files-processed"
    resp = requests.get(api_url)
    if resp.status_code == 200:
        files_list = resp.json()
        for item in files_list:
            if item["name"].lower().endswith((".jpg", ".jpeg", ".png")):
                img_resp = requests.get(item["download_url"])
                with open(os.path.join(INPUT_DIR, item["name"]), "wb") as f:
                    f.write(img_resp.content)
        imgs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Downloaded {len(imgs)} images from GitHub")
    else:
        print(f"GitHub API error ({resp.status_code}). Upload images manually to {INPUT_DIR}")

    imgs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Total: {len(imgs)} images")
    session.set_param("num_images", len(imgs))
    session.mark_step("images_downloaded")


def _restore_colmap_data(session):
    if session.checkpoint_exists("database.db"):
        session.restore_from_drive("database.db", DB_PATH)
        return True
    return False


def _restore_sparse_model(session):
    MODEL_PATH = os.path.join(COLMAP_DIR, "0")
    if session.checkpoint_exists("sparse_model"):
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        session.restore_from_drive("sparse_model", MODEL_PATH)
        return True
    return False


def run_colmap_features(session):
    check_colmap()
    MODEL_PATH = os.path.join(COLMAP_DIR, "0")
    os.makedirs(COLMAP_DIR, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_features"):
        print("Feature extraction already done. Restoring data from Drive...")
        _restore_colmap_data(session)
        _restore_sparse_model(session)
    elif os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 1000000:
        print("Features already extracted locally. Skipping.")
    else:
        if session.checkpoint_exists("database.db"):
            print("Database checkpoint found on Drive. Restoring...")
            session.restore_from_drive("database.db", DB_PATH)

        if not (os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 1000000):
            subprocess.run(
                f"colmap feature_extractor "
                f"--database_path {DB_PATH} "
                f"--image_path {INPUT_DIR} "
                f"--ImageReader.camera_model SIMPLE_RADIAL "
                f"--ImageReader.single_camera 1 "
                f"--SiftExtraction.use_gpu 0 "
                f"--SiftExtraction.max_num_features 8192 "
                f"--SiftExtraction.first_octave -1 "
                f"--SiftExtraction.peak_threshold 0.01",
                shell=True, check=True,
            )
            print("\nSaving checkpoint to Drive...")
            session.save_to_drive(DB_PATH, "database.db")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
    print(f"  Keypoints tables: {rows}")
    conn.close()
    session.mark_step("colmap_features")


def run_colmap_matching(session):
    check_colmap()
    DB_PATH_LOCAL = DB_PATH
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_matching"):
        print("Feature matching already done. Restoring DB from Drive...")
        _restore_colmap_data(session)
        return

    conn = sqlite3.connect(DB_PATH_LOCAL)
    match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()

    if match_count > 1000:
        print(f"{match_count} match pairs already exist. Skipping.")
        session.mark_step("colmap_matching")
        return

    print("\n=== Sequential Matching ===")
    subprocess.run(
        f"colmap sequential_matcher "
        f"--database_path {DB_PATH_LOCAL} "
        f"--SiftMatching.use_gpu 0 "
        f"--SequentialMatching.overlap 20",
        shell=True, check=True,
    )
    session.save_to_drive(DB_PATH_LOCAL, "database.db")

    print("\n=== Exhaustive Matching ===")
    subprocess.run(
        f"colmap exhaustive_matcher "
        f"--database_path {DB_PATH_LOCAL} "
        f"--SiftMatching.use_gpu 0",
        shell=True, check=True,
    )
    session.save_to_drive(DB_PATH_LOCAL, "database.db")

    conn = sqlite3.connect(DB_PATH_LOCAL)
    verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    print(f"  Verified: {verified} image pairs")
    conn.close()
    session.mark_step("colmap_matching")


def run_colmap_reconstruction(session):
    check_colmap()
    MODEL_PATH = os.path.join(COLMAP_DIR, "0")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_reconstruction"):
        print("COLMAP reconstruction already done. Restoring model from Drive...")
        _restore_colmap_data(session)
        _restore_sparse_model(session)
    elif os.path.exists(os.path.join(MODEL_PATH, "images.bin")):
        with open(os.path.join(MODEL_PATH, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        print(f"Model already exists ({n} images). Skipping.")
    elif session.checkpoint_exists("sparse_model"):
        print("Restoring sparse model from Drive...")
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        session.restore_from_drive("sparse_model", MODEL_PATH)
    else:
        subprocess.run(
            f"colmap mapper "
            f"--database_path {DB_PATH} "
            f"--image_path {INPUT_DIR} "
            f"--output_path {COLMAP_DIR} "
            f"--Mapper.multiple_models 1 "
            f"--Mapper.max_num_models 50 "
            f"--Mapper.init_min_tri_angle 4 "
            f"--Mapper.init_min_num_inliers 15 "
            f"--Mapper.abs_pose_min_num_inliers 8 "
            f"--Mapper.ba_local_max_num_iterations 25 "
            f"--Mapper.ba_global_max_num_iterations 50",
            shell=True, check=True,
        )

    best_model = None
    best_n = 0
    for sub in sorted(os.listdir(COLMAP_DIR)):
        img_path = os.path.join(COLMAP_DIR, sub, "images.bin")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
            if n > best_n:
                best_n = n
                best_model = sub

    if best_model:
        print(f"\nBest model: sub={best_model}, images={best_n}")
        src_m = os.path.join(COLMAP_DIR, best_model)
        session.save_to_drive(src_m, "sparse_model")
        session.set_param("colmap_registered_images", best_n)
    else:
        print("No reconstruction produced.")

    session.mark_step("colmap_reconstruction")

    _copy_sparse_to_input(best_model)


def run_colmap_merge(session):
    check_colmap()
    os.makedirs(MERGED_DIR, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_merged"):
        print("Model merging already done. Restoring data from Drive...")
        _restore_colmap_data(session)
        _restore_sparse_model(session)
        if os.path.exists(os.path.join(MERGED_DIR, "images.bin")):
            with open(os.path.join(MERGED_DIR, "images.bin"), "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
            print(f"Merged model already exists ({n} images).")
    elif os.path.exists(os.path.join(MERGED_DIR, "images.bin")):
        with open(os.path.join(MERGED_DIR, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        print(f"Merged model already exists ({n} images). Skipping.")
    else:
        models = []
        for sub in sorted(os.listdir(COLMAP_DIR)):
            img_path = os.path.join(COLMAP_DIR, sub, "images.bin")
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    n = struct.unpack("Q", f.read(8))[0]
                if n >= 5:
                    models.append((sub, n))
                    print(f"  Found model {sub}: {n} images")

        if len(models) >= 2:
            print("\nAttempting to merge models...")
            models.sort(key=lambda x: -x[1])
            current = os.path.join(COLMAP_DIR, models[0][0])
            for i in range(1, min(3, len(models))):
                other = os.path.join(COLMAP_DIR, models[i][0])
                merge_out = f"/content/gaussian-splatting/merge_{i}"
                os.makedirs(merge_out, exist_ok=True)
                result = subprocess.run(
                    ["colmap", "model_merger",
                     "--input_path1", current,
                     "--input_path2", other,
                     "--output_path", merge_out],
                    capture_output=True, text=True,
                )
                if os.path.exists(os.path.join(merge_out, "images.bin")):
                    with open(os.path.join(merge_out, "images.bin"), "rb") as f:
                        n = struct.unpack("Q", f.read(8))[0]
                    print(f"  Merge with {models[i][0]} successful: {n} images")
                    current = merge_out
                else:
                    print(f"  Merge with {models[i][0]} failed")
                    shutil.rmtree(merge_out, ignore_errors=True)
            shutil.copytree(current, MERGED_DIR, dirs_exist_ok=True)
        else:
            print("Not enough models to merge.")

        best_src = MERGED_DIR if os.path.exists(os.path.join(MERGED_DIR, "images.bin")) else (
            os.path.join(COLMAP_DIR, models[0][0]) if models else "0"
        )
        txt_dir = os.path.join(best_src, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        subprocess.run(
            f"colmap model_converter --input_path {best_src} --output_path {txt_dir} --output_type TXT 2>/dev/null",
            shell=True,
        )

    final = MERGED_DIR if os.path.exists(os.path.join(MERGED_DIR, "images.bin")) else os.path.join(COLMAP_DIR, "0")
    if os.path.exists(os.path.join(final, "images.bin")):
        with open(os.path.join(final, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        with open(os.path.join(final, "points3D.bin"), "rb") as f:
            pts = struct.unpack("Q", f.read(8))[0]
        print(f"\nOptimized COLMAP result: {n} images, {pts} 3D points")
        session.set_param("merged_images", n)
        session.set_param("merged_points3d", pts)

    session.mark_step("colmap_merged")
    _copy_sparse_to_input("0")


def _copy_sparse_to_input(best_model):
    os.makedirs(SPARSE_DIR, exist_ok=True)
    final_model = MERGED_DIR if os.path.exists(os.path.join(MERGED_DIR, "images.bin")) else (
        os.path.join(COLMAP_DIR, best_model) if best_model else os.path.join(COLMAP_DIR, "0")
    )
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        txt_src = os.path.join(os.path.join(final_model, "txt"), fname)
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, os.path.join(SPARSE_DIR, fname))
        elif os.path.exists(os.path.join(final_model, fname)):
            shutil.copy2(os.path.join(final_model, fname), os.path.join(SPARSE_DIR, fname))
        else:
            subprocess.run(
                f"colmap model_converter --input_path {final_model} --output_path {os.path.join(final_model, 'txt')} --output_type TXT 2>/dev/null",
                shell=True,
            )
            txt_src = os.path.join(os.path.join(final_model, "txt"), fname)
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, os.path.join(SPARSE_DIR, fname))


COLMAP_MODELS = {
    "30-image original": {
        "dir": "colmap_data",
        "expected_imgs": 30,
        "desc": "Original COLMAP model (30 registered images)",
    },
    "34-image merged": {
        "dir": "colmap_data_optimized",
        "expected_imgs": 34,
        "desc": "Merged/optimized COLMAP model (34 registered images)",
    },
    "84-image full": {
        "dir": "colmap_data_84",
        "expected_imgs": 84,
        "desc": "Full arena model with 84 registered images (recommended)",
    },
}


def download_precomputed_colmap(session, model_choice=None):
    import requests

    if session.is_step_done("colmap_downloaded"):
        print("Pre-computed COLMAP data already downloaded. Verifying files...")
    os.makedirs(SPARSE_DIR, exist_ok=True)

    if os.path.exists(os.path.join(SPARSE_DIR, "images.txt")):
        with open(os.path.join(SPARSE_DIR, "images.txt")) as f:
            n = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
        print(f"COLMAP data already present ({n} images).")
        session.set_param("expected_images", n)
        session.mark_step("colmap_downloaded")
        return

    if model_choice is None or model_choice not in COLMAP_MODELS:
        model_choice = "84-image full"

    model_info = COLMAP_MODELS[model_choice]

    local_names = ["cameras.txt", "images.txt", "points3D.txt"]
    for local in local_names:
        url = f"{REPO_URL}/{model_info['dir']}/{local}"
        print(f"Downloading {local} from {model_info['dir']}...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(os.path.join(SPARSE_DIR, local), "w") as f:
                f.write(r.text)
        else:
            print(f"  FAILED (status {r.status_code})")

    session.set_param("colmap_download_choice", model_choice)
    session.set_param("expected_images", model_info["expected_imgs"])

    with open(os.path.join(SPARSE_DIR, "images.txt")) as f:
        img_lines = [l for l in f if l.strip() and not l.startswith("#")]
        num_images = len(img_lines) // 2
    print(f"\nCOLMAP data: {num_images} registered images (SIMPLE_RADIAL)")
    session.mark_step("colmap_downloaded")


def _organize_images():
    if os.path.exists(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) >= 30:
        return
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in glob.glob(os.path.join(INPUT_DIR, ext)):
            dst = os.path.join(IMAGES_DIR, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(dst):
                shutil.move(f, dst)


def convert_to_3dgs_format(session):
    check_colmap()

    if session.is_step_done("data_converted"):
        print("Data already converted. Verifying files exist...")
        required = ["cameras.txt", "images.txt", "points3D.txt"]
        missing = [f for f in required if not os.path.exists(os.path.join(SPARSE_DIR, f))]
        if not missing:
            print("  All required files present.")
            return
        else:
            print(f"  Missing: {missing}. Will convert again.")

    _organize_images()

    required = ["cameras.txt", "images.txt", "points3D.txt"]
    missing = [f for f in required if not os.path.exists(os.path.join(SPARSE_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing COLMAP data: {missing}. Run COLMAP or download pre-computed data first.")

    cam_path = os.path.join(SPARSE_DIR, "cameras.txt")
    with open(cam_path) as f:
        lines = f.readlines()
    modified = False
    with open(cam_path, "w") as f:
        for line in lines:
            if line.startswith("#") or not line.strip():
                f.write(line)
            else:
                parts = line.strip().split()
                if parts[1] == "SIMPLE_RADIAL":
                    f.write(f"{parts[0]} PINHOLE {parts[2]} {parts[3]} {parts[4]} {parts[4]} {parts[5]} {parts[6]}\n")
                    modified = True
                else:
                    f.write(line)
    if modified:
        print("  Converted camera model: SIMPLE_RADIAL -> PINHOLE")

    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        p = os.path.join(SPARSE_DIR, fn)
        if os.path.exists(p):
            os.remove(p)
    bin_dir = "/content/gaussian-splatting/sparse_bin"
    os.makedirs(bin_dir, exist_ok=True)
    result = subprocess.run(
        ["colmap", "model_converter",
         "--input_path", SPARSE_DIR,
         "--output_path", bin_dir,
         "--output_type", "BIN"],
        capture_output=True, text=True,
    )
    if os.path.exists(os.path.join(bin_dir, "images.bin")):
        for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
            shutil.copy2(os.path.join(bin_dir, fn), os.path.join(SPARSE_DIR, fn))
        shutil.rmtree(bin_dir, ignore_errors=True)
        print("  Built binary files (cameras.bin, images.bin, points3D.bin)")
    else:
        print(f"WARNING: Binary conversion failed: {result.stderr}")

    with open(os.path.join(SPARSE_DIR, "images.txt")) as f:
        num_images = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
    print(f"\nReady for training: {num_images} images, PINHOLE model")
    session.set_param("training_images", num_images)
    session.mark_step("data_converted")


def train_3dgs(session, iterations=30000, max_gaussians=500000, log_interval=1000, max_res=1600, output_name="arena_3dgs"):
    step_name = f"training_{iterations // 1000}k"

    if session.is_step_done(step_name):
        print(f"Training ({iterations} iters) already done (session state). Skipping.")
        return

    if "scripts.train_3dgs_enhanced" in sys.modules:
        del sys.modules["scripts.train_3dgs_enhanced"]
    from scripts.train_3dgs_enhanced import train as train_fn
    from argparse import Namespace

    OUTPUT_DIR = f"/content/gaussian-splatting/output/{output_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args = Namespace(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        iterations=iterations,
        max_gaussians=max_gaussians,
        log_interval=log_interval,
        max_res=max_res,
    )
    try:
        train_fn(args)
        print(f"\nTraining ({iterations} iters) complete!")
        session.mark_step(step_name)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()


def export_pointcloud(session):
    if session.is_step_done("exported_ply"):
        print("PLY already exported (session state). Skipping.")
        return

    PLY_DST = "/content/arena_3dgs_pointcloud.ply"
    candidates = [
        "/content/gaussian-splatting/output/arena_3dgs/arena_3dgs.ply",
        "/content/gaussian-splatting/output/quick_test/arena_3dgs.ply",
    ]

    found_ply = None
    for p in candidates:
        if os.path.exists(p):
            found_ply = p
            print(f"Found: {p}")
            break

    if found_ply:
        shutil.copy2(found_ply, PLY_DST)
        size_mb = os.path.getsize(PLY_DST) / (1024 * 1024)
        print(f"\nPoint cloud: {PLY_DST} ({size_mb:.1f} MB)")
        session.save_to_drive(PLY_DST, "arena_3dgs_pointcloud.ply")
        from google.colab import files
        files.download(PLY_DST)
    else:
        print("No trained model found. Run training cells first.")

    session.mark_step("exported_ply")


def validate_pointcloud(ply_path="/content/arena_3dgs_pointcloud.ply"):
    if not os.path.exists(ply_path):
        print("No PLY found. Export first.")
        return

    from plyfile import PlyData
    import numpy as np

    ply = PlyData.read(ply_path)
    data = ply["vertex"].data
    n = len(data)
    print(f"Gaussians: {n:,}")

    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [p for p in required if p not in data.dtype.names]
    if missing:
        print(f"\nWARNING: Unity requires: {missing}")
    else:
        print("\nUnity format: OK")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    print(f"Bounds: X[{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}] "
          f"Y[{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}]")

    opacities = data["opacity"]
    visible = (np.array(opacities) > 0.01).sum()
    print(f"Visible Gaussians (>0.01 opacity): {visible:,}")
    print(f"File size: {os.path.getsize(ply_path) / 1024**2:.1f} MB")
