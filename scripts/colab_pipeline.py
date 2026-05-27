import os, struct, sqlite3, glob, shutil, subprocess, sys, atexit, urllib.request
from pathlib import Path


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
    *,
    drive_path=None,
    scripts_dir="/content/scripts",
    repo_url="https://raw.githubusercontent.com/kaarthik-balakrishnan/arena-3dgs/main",
):
    deps_marker = os.path.join(drive_path, ".deps_installed") if drive_path else None

    if deps_marker and os.path.exists(deps_marker) and colmap_available():
        print("Dependencies already installed (Drive marker found).")
    else:
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
        if deps_marker:
            Path(deps_marker).touch()
        print("\nSystem deps installed!")

    # Always download latest training scripts, regardless of marker
    os.makedirs(scripts_dir, exist_ok=True)
    for script_name in ["train_3dgs_enhanced.py", "visualize_coverage.py"]:
        print(f"Downloading {script_name} from GitHub...")
        urllib.request.urlretrieve(f"{repo_url}/scripts/{script_name}",
                                   os.path.join(scripts_dir, script_name))
        print("  Done.")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    ensure_virtual_display()
    print("\nAll dependencies ready!")


def load_images(
    source_path,
    *,
    input_dir="/content/gaussian-splatting/input",
    min_images=1,
):
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


def verify_sparse_model(*, sparse_dir="/content/gaussian-splatting/input/sparse/0"):
    required = ["cameras.txt", "images.txt", "points3D.txt"]
    missing = [f for f in required if not os.path.exists(os.path.join(sparse_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing COLMAP data in {sparse_dir}: {missing}")

    with open(os.path.join(sparse_dir, "images.txt")) as f:
        n = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
    print(f"COLMAP model verified: {n} registered images in {sparse_dir}")
    return n


def _ensure_images_dir(*, input_dir="/content/gaussian-splatting/input", images_dir="/content/gaussian-splatting/input/images"):
    """Ensure images subdirectory exists with copies of input images for 3DGS training."""
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) >= 1:
        return
    os.makedirs(images_dir, exist_ok=True)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in glob.glob(os.path.join(input_dir, ext)):
            dst = os.path.join(images_dir, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(dst):
                shutil.copy2(f, dst)


def run_colmap_features(
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

    if os.path.exists(db_path) and os.path.getsize(db_path) > 1000000:
        print("Features already extracted. Skipping.")
        return

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

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
    print(f"  Keypoints tables: {rows}")
    conn.close()


def run_colmap_matching(
    *,
    db_path="/content/gaussian-splatting/sparse/database.db",
    matching_mode="sequential + exhaustive",
    sequential_overlap=20,
):
    check_colmap()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    conn = sqlite3.connect(db_path)
    match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()

    if match_count > 1000:
        print(f"{match_count} match pairs already exist. Skipping.")
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

    if "exhaustive" in mode:
        print("\n=== Exhaustive Matching ===")
        subprocess.run(
            f"colmap exhaustive_matcher "
            f"--database_path {db_path} "
            f"--SiftMatching.use_gpu 0",
            shell=True, check=True,
        )

    conn = sqlite3.connect(db_path)
    verified = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    print(f"  Verified: {verified} image pairs")
    conn.close()


def run_colmap_reconstruction(
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
    check_colmap()
    model_path = os.path.join(colmap_dir, "0")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if os.path.exists(os.path.join(model_path, "images.bin")):
        with open(os.path.join(model_path, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        print(f"Model already exists ({n} images). Skipping.")
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

    # Find best model
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
        # Export to text format and copy to sparse_dir
        src_m = os.path.join(colmap_dir, best_model)
        txt_dir = os.path.join(src_m, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        subprocess.run(
            f"colmap model_converter --input_path {src_m} --output_path {txt_dir} --output_type TXT 2>/dev/null",
            shell=True,
        )
        os.makedirs(sparse_dir, exist_ok=True)
        for name in ["cameras.txt", "images.txt", "points3D.txt"]:
            txt_src = os.path.join(txt_dir, name)
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, os.path.join(sparse_dir, name))
    else:
        print("No reconstruction produced.")


def run_colmap_merge(
    *,
    colmap_dir="/content/gaussian-splatting/sparse",
    merged_dir="/content/gaussian-splatting/sparse_merged",
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
    min_images_for_merge=5,
):
    check_colmap()
    os.makedirs(merged_dir, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if os.path.exists(os.path.join(merged_dir, "images.bin")):
        with open(os.path.join(merged_dir, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        print(f"Merged model already exists ({n} images). Skipping.")
        return

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

    # Copy final merged model to sparse_dir as text
    final = merged_dir if os.path.exists(os.path.join(merged_dir, "images.bin")) else (
        os.path.join(colmap_dir, models[0][0]) if models else None
    )
    if final:
        txt_dir = os.path.join(final, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        subprocess.run(
            f"colmap model_converter --input_path {final} --output_path {txt_dir} --output_type TXT 2>/dev/null",
            shell=True,
        )
        os.makedirs(sparse_dir, exist_ok=True)
        for name in ["cameras.txt", "images.txt", "points3D.txt"]:
            txt_src = os.path.join(txt_dir, name)
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, os.path.join(sparse_dir, name))

        with open(os.path.join(final, "images.bin"), "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
        with open(os.path.join(final, "points3D.bin"), "rb") as f:
            pts = struct.unpack("Q", f.read(8))[0]
        print(f"\nOptimized COLMAP result: {n} images, {pts} 3D points")


def convert_to_3dgs_format(
    *,
    sparse_dir="/content/gaussian-splatting/input/sparse/0",
    input_dir="/content/gaussian-splatting/input",
    images_dir="/content/gaussian-splatting/input/images",
):
    check_colmap()

    required = ["cameras.txt", "images.txt", "points3D.txt"]
    missing = [f for f in required if not os.path.exists(os.path.join(sparse_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing COLMAP data: {missing}. Run COLMAP or load pre-computed data first.")

    # Ensure images subdirectory exists for 3DGS training
    _ensure_images_dir(input_dir=input_dir, images_dir=images_dir)

    with open(os.path.join(sparse_dir, "images.txt")) as f:
        num_images = sum(1 for l in f if l.strip() and not l.startswith("#")) // 2
    print(f"\nReady for training: {num_images} images")


def train_3dgs(
    iterations=30000,
    max_gaussians=0,
    log_interval=1000,
    max_res=800,
    output_name="arena_3dgs",
    *,
    input_dir="/content/gaussian-splatting/input",
    sparse_dir=None,
    output_base="/content/gaussian-splatting/output",
    random_background=False,
    opacity_reset_interval=3000,
    densify_until_iter=10000,
    densify_from_iter=500,
    densify_interval=100,
    sh_degree=3,
    sh_degree_interval=1000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    feature_lr=0.0025,
    opacity_lr=0.025,
    percent_dense=0.01,
    force_split_scale=0.02,
    scaling_lr=0.005,
    rotation_lr=0.001,
    lambda_dssim=0.2,
):
    if "scripts.train_3dgs_enhanced" in sys.modules:
        del sys.modules["scripts.train_3dgs_enhanced"]
    from scripts.train_3dgs_enhanced import train as train_fn
    from argparse import Namespace

    output_dir = os.path.join(output_base, output_name)
    os.makedirs(output_dir, exist_ok=True)
    args = Namespace(
        input_dir=input_dir,
        sparse_dir=sparse_dir,
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
        prune_opacity_threshold=0.01,
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
        sh_degree_interval=sh_degree_interval,
        percent_dense=percent_dense,
        force_split_scale=force_split_scale,
    )
    try:
        train_fn(args)
        print(f"\nTraining ({iterations} iters) complete!")
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()


def export_pointcloud(
    *,
    output_dirs=None,
    dst="/content/arena_3dgs_pointcloud.ply",
):
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
        from google.colab import files
        files.download(dst)
    else:
        print("No trained model found. Run training cells first.")


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
