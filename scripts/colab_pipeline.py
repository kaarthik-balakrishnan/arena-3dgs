import os, json, datetime, struct, sqlite3, glob, shutil, subprocess, sys, atexit, urllib.request, tempfile
from pathlib import Path

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


def ensure_virtual_display(display_num=99):
    os.environ["DISPLAY"] = f":{display_num}"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    lock = f"/tmp/.X{display_num}-lock"
    if not os.path.exists(lock):
        proc = subprocess.Popen(
            ["Xvfb", f":{display_num}", "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        atexit.register(lambda: proc.terminate())


def colmap_available():
    return bool(os.popen("which colmap 2>/dev/null").read().strip())


def check_colmap():
    if not colmap_available():
        raise RuntimeError("colmap not found. Run dependencies cell first.")


def install_dependencies(
    session,
    *,
    scripts_dir="/content/scripts",
    repo_url="https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main",
):
    if session.is_step_done("deps_installed") and colmap_available():
        print("Dependencies already installed (session state). Skipping.")
        ensure_virtual_display()
        return
    if session.is_step_done("deps_installed"):
        print("Session says deps installed but colmap not found. Re-installing.")

    drv = session.checkpoint_path("deps_installed")
    colmap_missing = not colmap_available()
    if not os.path.exists(drv) or colmap_missing:
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
        Path(drv).touch()
        print("\nSystem deps installed!")
    else:
        print("System deps already installed (Drive marker found).")

    enhanced = os.path.join(scripts_dir, "train_3dgs_enhanced.py")
    os.makedirs(scripts_dir, exist_ok=True)
    print("Downloading latest enhanced training script from GitHub...")
    urllib.request.urlretrieve(f"{repo_url}/scripts/train_3dgs_enhanced.py", enhanced)
    print("  Done.")
    sys.path.insert(0, scripts_dir)

    ensure_virtual_display()
    session.mark_step("deps_installed")
    print("\nAll dependencies ready!")


def load_images(
    session,
    source_path,
    *,
    input_dir="/content/gaussian-splatting/input",
    min_images=1,
):
    """Copy images from source_path on Drive into the training input directory."""
    if session.is_step_done("images_loaded"):
        print("Images already loaded. Skipping.")
        return

    if not os.path.isdir(source_path):
        raise NotADirectoryError(f"Source path not found: {source_path}")

    os.makedirs(input_dir, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(source_path)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(os.path.join(source_path, fname), os.path.join(input_dir, fname))
            count += 1

    if count < min_images:
        raise RuntimeError(f"Only {count} images found in {source_path} (need at least {min_images})")

    print(f"Copied {count} images from {source_path}")
    session.set_param("num_images", count)
    session.mark_step("images_loaded")


def download_precomputed_colmap(
    session,
    model_url=None,
    *,
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
):
    import requests, shutil

    if session.is_step_done("colmap_downloaded"):
        print("Pre-computed COLMAP data already downloaded. Verifying files...")
    os.makedirs(sparse_dir, exist_ok=True)

    if os.path.exists(os.path.join(sparse_dir, "images.txt")):
        with open(os.path.join(sparse_dir, "images.txt")) as f:
            n = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
        print(f"COLMAP data already present ({n} images).")
        session.set_param("expected_images", n)
        session.mark_step("colmap_downloaded")
        return

    if model_url is None:
        model_url = "https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main/colmap_data_84"

    is_url = model_url.startswith(("http://", "https://"))
    print(f"Source: {'URL' if is_url else 'Local path'} -> {model_url}")

    for name in ["cameras.txt", "images.txt", "points3D.txt"]:
        dest = os.path.join(sparse_dir, name)
        if is_url:
            url = f"{model_url}/{name}"
            print(f"  Downloading {name}...")
            r = requests.get(url)
            if r.status_code == 200:
                with open(dest, "w") as f:
                    f.write(r.text)
            else:
                print(f"  FAILED (status {r.status_code})")
        else:
            src = os.path.join(model_url, name)
            if os.path.exists(src):
                shutil.copy2(src, dest)
                print(f"  Copied {name}")
            else:
                print(f"  NOT FOUND: {src}")

    session.set_param("colmap_model_url", model_url)

    with open(os.path.join(sparse_dir, "images.txt")) as f:
        img_lines = [l for l in f if l.strip() and not l.startswith("#")]
        num_images = len(img_lines) // 2
    print(f"\nCOLMAP data: {num_images} registered images")
    session.mark_step("colmap_downloaded")


def _organize_images(*, input_dir="/content/gaussian-splatting/input", images_dir="/content/gaussian-splatting/input/images"):
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) >= 30:
        return
    os.makedirs(images_dir, exist_ok=True)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in glob.glob(os.path.join(input_dir, ext)):
            dst = os.path.join(images_dir, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(dst):
                shutil.move(f, dst)


def _restore_colmap_data(session, *, db_path="/content/gaussian-splatting/sparse/database.db"):
    if session.checkpoint_exists("database.db"):
        session.restore_from_drive("database.db", db_path)
        return True
    return False


def _restore_sparse_model(session, *, colmap_dir="/content/gaussian-splatting/sparse"):
    model_path = os.path.join(colmap_dir, "0")
    if session.checkpoint_exists("sparse_model"):
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        session.restore_from_drive("sparse_model", model_path)
        return True
    return False


def _copy_sparse_to_input(
    best_model,
    *,
    colmap_dir="/content/gaussian-splatting/sparse",
    merged_dir="/content/gaussian-splatting/sparse_merged",
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
):
    os.makedirs(sparse_dir, exist_ok=True)
    final_model = merged_dir if os.path.exists(os.path.join(merged_dir, "images.bin")) else (
        os.path.join(colmap_dir, best_model) if best_model else os.path.join(colmap_dir, "0")
    )
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        txt_src = os.path.join(os.path.join(final_model, "txt"), fname)
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, os.path.join(sparse_dir, fname))
        elif os.path.exists(os.path.join(final_model, fname)):
            shutil.copy2(os.path.join(final_model, fname), os.path.join(sparse_dir, fname))
        else:
            subprocess.run(
                f"colmap model_converter --input_path {final_model} --output_path {os.path.join(final_model, 'txt')} --output_type TXT 2>/dev/null",
                shell=True,
            )
            txt_src = os.path.join(os.path.join(final_model, "txt"), fname)
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, os.path.join(sparse_dir, fname))


def run_colmap_features(
    session,
    *,
    input_dir="/content/gaussian-splatting/input",
    colmap_dir="/content/gaussian-splatting/sparse",
    db_path="/content/gaussian-splatting/sparse/database.db",
    camera_model="SIMPLE_RADIAL",
    single_camera=True,
    max_num_features=8192,
    first_octave=-1,
    peak_threshold=0.01,
):
    check_colmap()
    os.makedirs(colmap_dir, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_features"):
        print("Feature extraction already done. Restoring data from Drive...")
        _restore_colmap_data(session, db_path=db_path)
        _restore_sparse_model(session, colmap_dir=colmap_dir)
    elif os.path.exists(db_path) and os.path.getsize(db_path) > 1000000:
        print("Features already extracted locally. Skipping.")
    else:
        if session.checkpoint_exists("database.db"):
            print("Database checkpoint found on Drive. Restoring...")
            session.restore_from_drive("database.db", db_path)

        if not (os.path.exists(db_path) and os.path.getsize(db_path) > 1000000):
            subprocess.run(
                f"colmap feature_extractor "
                f"--database_path {db_path} "
                f"--image_path {input_dir} "
                f"--ImageReader.camera_model {camera_model} "
                f"--ImageReader.single_camera {1 if single_camera else 0} "
                f"--SiftExtraction.use_gpu 0 "
                f"--SiftExtraction.max_num_features {max_num_features} "
                f"--SiftExtraction.first_octave {first_octave} "
                f"--SiftExtraction.peak_threshold {peak_threshold}",
                shell=True, check=True,
            )
            print("\nSaving checkpoint to Drive...")
            session.save_to_drive(db_path, "database.db")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
    print(f"  Keypoints tables: {rows}")
    conn.close()
    session.mark_step("colmap_features")


def run_colmap_matching(
    session,
    *,
    db_path="/content/gaussian-splatting/sparse/database.db",
    matching_mode="sequential + exhaustive",
    sequential_overlap=20,
):
    check_colmap()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_matching"):
        print("Feature matching already done. Restoring DB from Drive...")
        _restore_colmap_data(session, db_path=db_path)
        return

    conn = sqlite3.connect(db_path)
    match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()

    if match_count > 1000:
        print(f"{match_count} match pairs already exist. Skipping.")
        session.mark_step("colmap_matching")
        return

    mode = matching_mode.lower()

    if "sequential" in mode:
        print("\n=== Sequential Matching ===")
        subprocess.run(
            f"colmap sequential_matcher "
            f"--database_path {db_path} "
            f"--SiftMatching.use_gpu 0 "
            f"--SequentialMatching.overlap {sequential_overlap}",
            shell=True, check=True,
        )
        session.save_to_drive(db_path, "database.db")

    if "exhaustive" in mode:
        print("\n=== Exhaustive Matching ===")
        subprocess.run(
            f"colmap exhaustive_matcher "
            f"--database_path {db_path} "
            f"--SiftMatching.use_gpu 0",
            shell=True, check=True,
        )
        session.save_to_drive(db_path, "database.db")

    conn = sqlite3.connect(db_path)
    verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    print(f"  Verified: {verified} image pairs")
    conn.close()
    session.mark_step("colmap_matching")


def run_colmap_reconstruction(
    session,
    *,
    input_dir="/content/gaussian-splatting/input",
    colmap_dir="/content/gaussian-splatting/sparse",
    db_path="/content/gaussian-splatting/sparse/database.db",
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
    multiple_models=True,
    max_num_models=50,
    init_min_tri_angle=4,
    init_min_num_inliers=15,
    abs_pose_min_num_inliers=8,
    ba_local_max_num_iterations=25,
    ba_global_max_num_iterations=50,
):
    import struct

    check_colmap()
    model_path = os.path.join(colmap_dir, "0")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_reconstruction"):
        print("COLMAP reconstruction already done. Restoring model from Drive...")
        _restore_colmap_data(session, db_path=db_path)
        _restore_sparse_model(session, colmap_dir=colmap_dir)
    elif os.path.exists(os.path.join(model_path, "images.bin")):
        with open(os.path.join(model_path, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        print(f"Model already exists ({n} images). Skipping.")
    elif session.checkpoint_exists("sparse_model"):
        print("Restoring sparse model from Drive...")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        session.restore_from_drive("sparse_model", model_path)
    else:
        subprocess.run(
            f"colmap mapper "
            f"--database_path {db_path} "
            f"--image_path {input_dir} "
            f"--output_path {colmap_dir} "
            f"--Mapper.multiple_models {1 if multiple_models else 0} "
            f"--Mapper.max_num_models {max_num_models} "
            f"--Mapper.init_min_tri_angle {init_min_tri_angle} "
            f"--Mapper.init_min_num_inliers {init_min_num_inliers} "
            f"--Mapper.abs_pose_min_num_inliers {abs_pose_min_num_inliers} "
            f"--Mapper.ba_local_max_num_iterations {ba_local_max_num_iterations} "
            f"--Mapper.ba_global_max_num_iterations {ba_global_max_num_iterations}",
            shell=True, check=True,
        )

    best_model = None
    best_n = 0
    for sub in sorted(os.listdir(colmap_dir)):
        img_path = os.path.join(colmap_dir, sub, "images.bin")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
            if n > best_n:
                best_n = n
                best_model = sub

    if best_model:
        print(f"\nBest model: sub={best_model}, images={best_n}")
        src_m = os.path.join(colmap_dir, best_model)
        session.save_to_drive(src_m, "sparse_model")
        session.set_param("colmap_registered_images", best_n)
    else:
        print("No reconstruction produced.")

    session.mark_step("colmap_reconstruction")

    _copy_sparse_to_input(best_model, colmap_dir=colmap_dir, sparse_dir=sparse_dir)


def run_colmap_merge(
    session,
    *,
    colmap_dir="/content/gaussian-splatting/sparse",
    merged_dir="/content/gaussian-splatting/sparse_merged",
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
    min_images_for_merge=5,
):
    import struct

    check_colmap()
    os.makedirs(merged_dir, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if session.is_step_done("colmap_merged"):
        print("Model merging already done. Restoring data from Drive...")
        _restore_colmap_data(session, db_path=os.path.join(colmap_dir, "database.db"))
        _restore_sparse_model(session, colmap_dir=colmap_dir)
        if os.path.exists(os.path.join(merged_dir, "images.bin")):
            with open(os.path.join(merged_dir, "images.bin"), "rb") as f:
                n = struct.unpack("Q", f.read(8))[0]
            print(f"Merged model already exists ({n} images).")
    elif os.path.exists(os.path.join(merged_dir, "images.bin")):
        with open(os.path.join(merged_dir, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        print(f"Merged model already exists ({n} images). Skipping.")
    else:
        models = []
        for sub in sorted(os.listdir(colmap_dir)):
            img_path = os.path.join(colmap_dir, sub, "images.bin")
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    n = struct.unpack("Q", f.read(8))[0]
                if n >= min_images_for_merge:
                    models.append((sub, n))
                    print(f"  Found model {sub}: {n} images")

        if len(models) >= 2:
            print("\nAttempting to merge models...")
            models.sort(key=lambda x: -x[1])
            current = os.path.join(colmap_dir, models[0][0])
            for i in range(1, min(3, len(models))):
                other = os.path.join(colmap_dir, models[i][0])
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
            shutil.copytree(current, merged_dir, dirs_exist_ok=True)
        else:
            print("Not enough models to merge.")

        best_src = merged_dir if os.path.exists(os.path.join(merged_dir, "images.bin")) else (
            os.path.join(colmap_dir, models[0][0]) if models else "0"
        )
        txt_dir = os.path.join(best_src, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        subprocess.run(
            f"colmap model_converter --input_path {best_src} --output_path {txt_dir} --output_type TXT 2>/dev/null",
            shell=True,
        )

    final = merged_dir if os.path.exists(os.path.join(merged_dir, "images.bin")) else os.path.join(colmap_dir, "0")
    if os.path.exists(os.path.join(final, "images.bin")):
        with open(os.path.join(final, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        with open(os.path.join(final, "points3D.bin"), "rb") as f:
            pts = struct.unpack("Q", f.read(8))[0]
        print(f"\nOptimized COLMAP result: {n} images, {pts} 3D points")
        session.set_param("merged_images", n)
        session.set_param("merged_points3d", pts)

    session.mark_step("colmap_merged")
    _copy_sparse_to_input("0", colmap_dir=colmap_dir, merged_dir=merged_dir, sparse_dir=sparse_dir)


def convert_to_3dgs_format(
    session,
    *,
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
    input_dir="/content/gaussian-splatting/input",
    images_dir="/content/gaussian-splatting/input/images",
):
    check_colmap()

    if session.is_step_done("data_converted"):
        print("Data already converted. Verifying files exist...")
        required = ["cameras.txt", "images.txt", "points3D.txt"]
        missing = [f for f in required if not os.path.exists(os.path.join(sparse_dir, f))]
        if not missing:
            print("  All required files present.")
            return
        else:
            print(f"  Missing: {missing}. Will convert again.")

    _organize_images(input_dir=input_dir, images_dir=images_dir)

    required = ["cameras.txt", "images.txt", "points3D.txt"]
    missing = [f for f in required if not os.path.exists(os.path.join(sparse_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing COLMAP data: {missing}. Run COLMAP or download pre-computed data first.")

    with open(os.path.join(sparse_dir, "images.txt")) as f:
        num_images = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
    print(f"\nReady for training: {num_images} images, SIMPLE_RADIAL model")
    session.set_param("training_images", num_images)
    session.mark_step("data_converted")


def train_3dgs(
    session,
    iterations=30000,
    max_gaussians=0,
    log_interval=1000,
    max_res=1600,
    output_name="arena_3dgs",
    *,
    input_dir="/content/gaussian-splatting/input",
    output_base="/content/gaussian-splatting/output",
    random_background=False,
    opacity_reset_interval=3000,
    densify_until_iter=15000,
    densify_from_iter=500,
    densify_interval=100,
    sh_degree=3,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    lambda_dssim=0.2,
):
    step_name = f"training_{iterations // 1000}k"

    if session.is_step_done(step_name):
        print(f"Training ({iterations} iters) already done (session state). Skipping.")
        return

    if "scripts.train_3dgs_enhanced" in sys.modules:
        del sys.modules["scripts.train_3dgs_enhanced"]
    from scripts.train_3dgs_enhanced import train as train_fn
    from argparse import Namespace

    output_dir = os.path.join(output_base, output_name)
    os.makedirs(output_dir, exist_ok=True)
    args = Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        iterations=iterations,
        max_gaussians=max_gaussians,
        log_interval=log_interval,
        max_res=max_res,
        random_background=random_background,
        max_init_points=50000,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=0.0002,
        prune_opacity_threshold=0.005,
        opacity_reset_interval=opacity_reset_interval,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_max_steps=30000,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
        lambda_dssim=lambda_dssim,
        sh_degree=sh_degree,
    )
    try:
        train_fn(args)
        print(f"\nTraining ({iterations} iters) complete!")
        session.mark_step(step_name)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()


def export_pointcloud(
    session,
    *,
    output_dirs=None,
    dst="/content/arena_3dgs_pointcloud.ply",
):
    if session.is_step_done("exported_ply"):
        print("PLY already exported (session state). Skipping.")
        return

    if output_dirs is None:
        output_dirs = [
            "/content/gaussian-splatting/output/arena_3dgs/arena_3dgs.ply",
            "/content/gaussian-splatting/output/quick_test/arena_3dgs.ply",
        ]

    found_ply = None
    for p in output_dirs:
        if os.path.exists(p):
            found_ply = p
            print(f"Found: {p}")
            break

    if found_ply:
        shutil.copy2(found_ply, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"\nPoint cloud: {dst} ({size_mb:.1f} MB)")
        session.save_to_drive(dst, "arena_3dgs_pointcloud.ply")
        from google.colab import files
        files.download(dst)
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
