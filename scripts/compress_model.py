#!/usr/bin/env python3
"""
Post-training compression for 3D Gaussian Splatting.

Implements the full Unity GaussianSplatting plugin pipeline:
  - Gaussian pruning (opacity / screen-size threshold)
  - SH coefficient vector quantization (mini-batch k-means)
  - Attribute quantization (per-chunk or global)
  - Morton (Z-order) reordering + chunking
  - Per-chunk normalization to [0,1] for lower bit depths
  - Custom binary .gsplat export

Usage:
  python scripts/compress_model.py output/model.ply --quality medium
  python scripts/compress_model.py output/model.ply --quality low -o output/compressed
  python scripts/compress_model.py --list-qualities
  python scripts/compress_model.py output/model.ply --prune-opacity 0.01 --prune-screen-size 20

Outputs:
  <name>_compressed.gsplat   —  custom binary with per-chunk quantization
  <name>_compressed.ply      —  standard-format PLY (decompressed, for compat)
  <name>_standard.splat      —  32-byte/splat SuperSplat-compatible format
"""
import os, sys, math, struct, time, functools
from pathlib import Path
import warnings
import numpy as np

print = functools.partial(print, flush=True)

BASE = Path(__file__).resolve().parent.parent

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData, PlyElement


# ═══════════════════════════════════════════════════════════════════════════════
#  Morton (Z-order) encoding — 3D → 64-bit code
# ═══════════════════════════════════════════════════════════════════════════════

def _spread(v: np.ndarray) -> np.ndarray:
    """Spread 21 bits of each of 3 int32s so they interleave into 63 bits."""
    v = v.astype(np.uint64)
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    return v


def morton_code(points: np.ndarray) -> np.ndarray:
    """Compute 63-bit Z-order code for an (N, 3) float array.

    Quantizes positions to 21 bits per axis within the bounding box.
    """
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    extent[extent < 1e-10] = 1.0
    norm = ((points - pmin) / extent * ((1 << 21) - 1)).astype(np.int32)
    x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]
    return _spread(x) | (_spread(y) << 1) | (_spread(z) << 2)


def morton_reorder(points: np.ndarray, tensors: dict) -> dict:
    """Reorder all tensors by 3D Morton code of positions."""
    codes = morton_code(points)
    order = np.argsort(codes)
    reordered = {}
    for key, arr in tensors.items():
        reordered[key] = arr[order]
    return reordered


# ═══════════════════════════════════════════════════════════════════════════════
#  Chunk utilities
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 256


def split_chunks(tensors: dict, chunk_size: int = CHUNK_SIZE) -> list:
    """Split a dict of (N, D) tensors into a list of per-chunk dicts."""
    N = len(next(iter(tensors.values())))
    chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = {k: v[start:end] for k, v in tensors.items()}
        chunks.append(chunk)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-chunk normalizer
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_chunk(chunk: dict, fields: list[str], bit_widths: dict[str, int]):
    """Normalize specified fields to [0, 1] within the chunk, then quantize."""
    meta = {}
    for field in fields:
        raw = chunk[field].astype(np.float64)
        cmin = raw.min(axis=0)
        cmax = raw.max(axis=0)
        span = np.maximum(cmax - cmin, 1e-10)
        norm = (raw - cmin) / span
        bits = bit_widths.get(field, 8)
        max_val = (1 << bits) - 1
        q = np.round(np.clip(norm, 0.0, 1.0) * max_val).astype(_int_type(bits))
        chunk[f'{field}_q'] = q
        meta[field] = (cmin.astype(np.float32), cmax.astype(np.float32))
    return meta


def _int_type(bits: int):
    if bits <= 8:
        return np.uint8
    elif bits <= 16:
        return np.uint16
    elif bits <= 32:
        return np.uint32
    return np.uint64


# ═══════════════════════════════════════════════════════════════════════════════
#  SH Clustering — Mini-batch k-means (numpy-based, chunked)
# ═══════════════════════════════════════════════════════════════════════════════

class SHClusterCompressor:
    """Cluster SH rest coefficients (Nx45) into a palette of K centroids."""
    def __init__(self, n_clusters=16384, n_iters=15, batch_size=100000,
                 cluster_chunk=2048, random_seed=42):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.cluster_chunk = min(cluster_chunk, n_clusters)
        self.random_seed = random_seed
        self.centroids = None
        self.inertia_ = None

    def _nearest_chunked(self, points, centroids):
        N = points.shape[0]
        K = centroids.shape[0]
        chunk = min(self.cluster_chunk, K)
        nearest = np.empty(N, dtype=np.int32)
        min_dist = np.full(N, np.inf, dtype=np.float32)
        a2 = (points * points).sum(axis=1, keepdims=True)
        for k_start in range(0, K, chunk):
            k_end = min(k_start + chunk, K)
            c = centroids[k_start:k_end]
            ab = points @ c.T
            b2 = (c * c).sum(axis=1, keepdims=True).T
            dists = a2 + b2 - 2 * ab
            c_nearest = dists.argmin(axis=1).astype(np.int32)
            c_min = dists[np.arange(N), c_nearest]
            update = c_min < min_dist
            if update.any():
                min_dist[update] = c_min[update]
                nearest[update] = c_nearest[update] + k_start
        return nearest

    def fit(self, sh_rest):
        N, D = sh_rest.shape
        print(f"  Fitting {self.n_clusters} clusters on {N:,} x {D} SH data...")
        t0 = time.time()
        rng = np.random.RandomState(self.random_seed)
        perm = rng.permutation(N)
        centroids = sh_rest[perm[:self.n_clusters]].astype(np.float32).copy()
        counts = np.ones(self.n_clusters) * 1e-6
        n_splits = max(1, N // self.batch_size)
        perm_train = rng.permutation(N)

        for it in range(self.n_iters):
            rng.shuffle(perm_train)
            for split in range(n_splits):
                start = split * self.batch_size
                end = min(start + self.batch_size, N)
                batch = sh_rest[perm_train[start:end]]
                nearest = self._nearest_chunked(batch, centroids)
                sum_per_c = np.zeros((self.n_clusters, D), dtype=np.float32)
                cnt_per_c = np.zeros(self.n_clusters, dtype=np.float32)
                np.add.at(sum_per_c, nearest, batch)
                np.add.at(cnt_per_c, nearest, 1)
                mask = cnt_per_c > 0
                if mask.any():
                    lr = cnt_per_c[mask] / (counts[mask] + cnt_per_c[mask])
                    centroids[mask] = (1 - lr[:, None]) * centroids[mask] \
                                      + lr[:, None] * (sum_per_c[mask] / cnt_per_c[mask, None])
                    counts[mask] += cnt_per_c[mask]

            if (it + 1) % 10 == 0 or it == 0:
                sample = sh_rest[perm[:min(50000, N)]]
                nearest = self._nearest_chunked(sample, centroids)
                a2_s = (sample * sample).sum(axis=1)
                inertia = (a2_s - 2 * (sample * centroids[nearest]).sum(axis=1)
                           + (centroids[nearest] * centroids[nearest]).sum(axis=1)).mean()
                print(f"    Iter {it+1}/{self.n_iters}, inertia={inertia:.4f}, "
                      f"time={time.time()-t0:.1f}s")

        self.centroids = centroids
        self.inertia_ = inertia if self.inertia_ is None else inertia
        print(f"  Clustering done in {time.time()-t0:.1f}s, "
              f"final inertia={self.inertia_:.4f}")
        return self

    def compress(self, sh_rest):
        N = sh_rest.shape[0]
        chunk = 200000
        indices = np.empty(N, dtype=np.int32)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            indices[start:end] = self._nearest_chunked(
                sh_rest[start:end], self.centroids)
        return indices


# ═══════════════════════════════════════════════════════════════════════════════
#  PLY I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_ply(ply_path):
    """Load a 3DGS PLY file and return dict of numpy arrays."""
    ply_path = Path(ply_path)
    print(f"\nLoading: {ply_path}")
    t0 = time.time()
    ply = PlyData.read(str(ply_path))
    vertex = ply['vertex']
    data = vertex.data
    N = len(data)
    dtype_names = list(data.dtype.names)
    print(f"  {N:,} gaussians, properties: {dtype_names}")

    required = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    missing = [p for p in required if p not in dtype_names]
    if missing:
        print(f"  ERROR: Missing required fields: {missing}")
        return None

    has_full_sh = all(f'f_rest_{i}' in dtype_names for i in range(45))

    tensors = {
        'positions': np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32),
        'colors_dc': np.column_stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']]).astype(np.float32),
        'opacity': np.asarray(data['opacity'], dtype=np.float32),
        'scales': np.column_stack([data['scale_0'], data['scale_1'], data['scale_2']]).astype(np.float32),
        'quats': np.column_stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']]).astype(np.float32),
    }

    if has_full_sh:
        rest = np.column_stack([data[f'f_rest_{i}'] for i in range(45)]).astype(np.float32)
        tensors['sh_rest'] = rest
        print(f"  SH rest: 45 coefficients per splat")
    else:
        tensors['sh_rest'] = np.zeros((N, 45), dtype=np.float32)
        print(f"  SH rest: none (filling zeros)")

    size_mb = ply_path.stat().st_size / (1024 * 1024)
    print(f"  Loaded in {time.time()-t0:.1f}s, file size: {size_mb:.1f} MB")
    return tensors


def export_ply(tensors, path):
    """Export a standard-format PLY from a dict of numpy arrays."""
    N = len(tensors['positions'])
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ]
    dtype_full += [(f'f_dc_{i}', 'f4') for i in range(3)]
    dtype_full += [(f'f_rest_{i}', 'f4') for i in range(45)]
    dtype_full += [('opacity', 'f4')]
    dtype_full += [(f'scale_{i}', 'f4') for i in range(3)]
    dtype_full += [(f'rot_{i}', 'f4') for i in range(4)]

    elements = np.zeros(N, dtype=dtype_full)
    elements['x'] = tensors['positions'][:, 0]
    elements['y'] = tensors['positions'][:, 1]
    elements['z'] = tensors['positions'][:, 2]
    elements['f_dc_0'] = tensors['colors_dc'][:, 0]
    elements['f_dc_1'] = tensors['colors_dc'][:, 1]
    elements['f_dc_2'] = tensors['colors_dc'][:, 2]

    rest = tensors.get('sh_rest', np.zeros((N, 45), dtype=np.float32))
    for i in range(45):
        elements[f'f_rest_{i}'] = rest[:, i]

    elements['opacity'] = tensors['opacity']
    elements['scale_0'] = tensors['scales'][:, 0]
    elements['scale_1'] = tensors['scales'][:, 1]
    elements['scale_2'] = tensors['scales'][:, 2]
    elements['rot_0'] = tensors['quats'][:, 0]
    elements['rot_1'] = tensors['quats'][:, 1]
    elements['rot_2'] = tensors['quats'][:, 2]
    elements['rot_3'] = tensors['quats'][:, 3]

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Exported: {path} ({size_mb:.1f} MB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Quality Presets
# ═══════════════════════════════════════════════════════════════════════════════

QUALITY_PRESETS = {
    'very_high': {
        'desc': 'Minimal loss, ~2x',
        'sh_mode': 'fp16',
        'n_sh_clusters': 0,
        'chunk_pos_bits': 16,
        'chunk_scale_bits': 16,
        'chunk_color_bits': 16,
        'chunk_opacity_bits': 16,
    },
    'high': {
        'desc': 'Low loss, ~3x',
        'sh_mode': 'norm11',
        'n_sh_clusters': 0,
        'chunk_pos_bits': 11,
        'chunk_scale_bits': 11,
        'chunk_color_bits': 10,
        'chunk_opacity_bits': 10,
    },
    'medium': {
        'desc': 'Balanced, ~5x',
        'sh_mode': 'norm565',
        'n_sh_clusters': 0,
        'chunk_pos_bits': 11,
        'chunk_scale_bits': 8,
        'chunk_color_bits': 8,
        'chunk_opacity_bits': 8,
    },
    'low': {
        'desc': 'SH cluster 16K, ~15x',
        'sh_mode': 'cluster',
        'n_sh_clusters': 16384,
        'chunk_pos_bits': 8,
        'chunk_scale_bits': 6,
        'chunk_color_bits': 6,
        'chunk_opacity_bits': 6,
    },
    'very_low': {
        'desc': 'SH cluster 4K, ~18x',
        'sh_mode': 'cluster',
        'n_sh_clusters': 4096,
        'chunk_pos_bits': 6,
        'chunk_scale_bits': 5,
        'chunk_color_bits': 5,
        'chunk_opacity_bits': 4,
    },
}

# Which fields to normalize per-chunk (by quality tier)
CHUNK_FIELDS = ['positions', 'scales', 'colors_dc', 'opacity']


def _sh_bytes_per_splat(mode, n_clusters):
    if mode == 'fp16':
        return 90
    elif mode == 'norm11':
        return 45 * 2
    elif mode == 'norm565':
        return 15 * 2
    elif mode == 'cluster':
        bits = max(8, (n_clusters - 1).bit_length())
        return (bits + 7) // 8
    return 90


# ═══════════════════════════════════════════════════════════════════════════════
#  Compress SH
# ═══════════════════════════════════════════════════════════════════════════════

def compress_sh(sh_rest, mode, n_clusters=0, cluster_compressor=None):
    """Compress SH rest coefficients. Returns (compressed, compressor, meta)."""
    if mode == 'fp16':
        return sh_rest.astype(np.float16), None, {'mode': 'fp16'}

    elif mode == 'norm11':
        max_val = (2 ** 10) - 1
        q = np.round(np.clip(sh_rest, -1, 1) * max_val).astype(np.int16)
        return q, None, {'mode': 'norm11'}

    elif mode == 'norm565':
        N, D = sh_rest.shape
        sh3 = sh_rest.reshape(N, 15, 3)
        r = (np.clip(sh3[:, :, 0], -1, 1) * 15).round().astype(np.int32) & 0x1F
        g = (np.clip(sh3[:, :, 1], -1, 1) * 31).round().astype(np.int32) & 0x3F
        b = (np.clip(sh3[:, :, 2], -1, 1) * 15).round().astype(np.int32) & 0x1F
        packed = ((r << 11) | (g << 5) | b).astype(np.uint16)
        return packed, None, {'mode': 'norm565'}

    elif mode == 'cluster':
        if cluster_compressor is None:
            cc = SHClusterCompressor(n_clusters=n_clusters)
            cc.fit(sh_rest)
            cluster_compressor = cc
        indices = cluster_compressor.compress(sh_rest).astype(np.uint32)
        return indices, cluster_compressor, {
            'mode': 'cluster', 'n_clusters': n_clusters,
            'palette_bytes': cluster_compressor.n_clusters * 45 * 2,
        }

    raise ValueError(f"Unknown SH mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Standard .splat Export (32-byte per Gaussian, SuperSplat-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

def export_standard_splat(ply_path, output_path=None):
    """Export PLY to standard 32-byte-per-splat format (SuperSplat/PlayCanvas)."""
    ply_path = Path(ply_path)
    if output_path is None:
        output_path = ply_path.with_name(ply_path.stem + "_standard.splat")
    output_path = Path(output_path)

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    N = len(vertex)

    pos = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    rots = np.column_stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]]).astype(np.float32)
    scales = np.column_stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]]).astype(np.float32)
    opacities = vertex["opacity"].astype(np.float32)
    dc = np.column_stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]]).astype(np.float32)

    norms = np.sqrt(np.sum(rots ** 2, axis=1, keepdims=True))
    rots_norm = rots / (norms + 1e-10)
    rots_packed = np.clip(np.round((rots_norm + 1.0) * 127.5), 0, 255).astype(np.uint8)
    scales_packed = np.clip(np.round(scales * 16.0 + 128.0), 0, 255).astype(np.uint8)
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities))
    opacity_packed = np.clip(np.round(opacity_sigmoid * 255.0), 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for i in range(N):
            f.write(pos[i].tobytes())
            f.write(rots_packed[i].tobytes())
            f.write(scales_packed[i].tobytes())
            f.write(opacity_packed[i])
            f.write(dc[i].tobytes())

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Standard .splat: {output_path} ({size_mb:.1f} MB, {N:,} gaussians)")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Binary .gsplat export (Unity-inspired format)
# ═══════════════════════════════════════════════════════════════════════════════

def export_gsplat_binary(
    chunk_data: list,
    sh_compressed: np.ndarray,
    sh_meta: dict,
    palette: np.ndarray | None,
    quality: str,
    config: dict,
    path: Path,
):
    """Export compressed binary with per-chunk quantization.

    Format:
      [Header 164B]
      [SH palette: n_clusters * 45 * float16]
      [Chunk headers: n_chunks * chunk_header_size]
      [Per-Gaussian data: packed per-chunk]
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = len(chunk_data)
    n_gaussians = sum(len(ch['positions_q']) for ch in chunk_data)
    chunk_size = CHUNK_SIZE
    sh_mode = sh_meta['mode']
    n_sh_clusters = sh_meta.get('n_clusters', 0)

    # ── SH palette ──
    palette_bytes = palette.astype(np.float16).tobytes() if palette is not None else b''

    # ── Per-chunk header size ──
    # Each chunk header: 8 fields × (3 or 1) × float32 min/max
    chunk_header_bytes = sum(
        (3 if f in ('positions', 'scales', 'colors_dc') else 1) * 2 * 4
        for f in CHUNK_FIELDS
    )  # = (3+3+3+1)*2*4 = 80 bytes

    # ── Per-Gaussian byte count (fixed stride for simplicity) ──
    pos_bytes = (config['chunk_pos_bits'] * 3 + 7) // 8
    scale_bytes = (config['chunk_scale_bits'] * 3 + 7) // 8
    color_bytes = (config['chunk_color_bits'] * 3 + 7) // 8
    opacity_bytes = (config['chunk_opacity_bits'] + 7) // 8
    rot_bytes = 4   # uint8 x 4
    sh_bytes = _sh_bytes_per_splat(sh_mode, n_sh_clusters)
    per_gaussian_bytes = pos_bytes + scale_bytes + color_bytes + opacity_bytes + rot_bytes + sh_bytes

    # ── Write ──
    with open(path, 'wb') as f:
        # Header
        f.write(b'GSCP')                         # magic
        f.write(struct.pack('<I', 1))             # version
        f.write(struct.pack('<I', n_gaussians))
        f.write(struct.pack('<I', n_chunks))
        f.write(struct.pack('<I', chunk_size))
        f.write(struct.pack('<I', chunk_header_bytes))
        f.write(struct.pack('<I', per_gaussian_bytes))
        f.write(struct.pack('<I', n_sh_clusters))
        name = quality.encode('ascii', errors='replace')
        f.write(name.ljust(32, b'\x00'))          # quality name, padded
        f.write(bytes(96))                        # reserved

        # SH palette
        f.write(palette_bytes)

        # Chunk headers: for each chunk, write min/max for each field
        for ch in chunk_data:
            meta = ch['_meta']
            for field in CHUNK_FIELDS:
                cmin, cmax = meta[field]
                f.write(np.ascontiguousarray(cmin).tobytes())
                f.write(np.ascontiguousarray(cmax).tobytes())

        # Per-Gaussian data: contiguous per-attribute arrays per chunk
        global_idx = 0
        for ch in chunk_data:
            n = len(ch['positions_q'])
            f.write(np.ascontiguousarray(ch['positions_q']).tobytes())
            f.write(np.ascontiguousarray(ch['scales_q']).tobytes())
            f.write(np.ascontiguousarray(ch['colors_dc_q']).tobytes())
            f.write(np.ascontiguousarray(ch['opacity_q']).tobytes())
            # Rotation: map [0,1] → uint8
            rot_q = np.round(np.clip(ch['quats'], 0, 1) * 255).astype(np.uint8)
            f.write(np.ascontiguousarray(rot_q).tobytes())
            # SH index (cluster mode) or zero padding
            if sh_mode == 'cluster':
                sh_idx = sh_compressed[global_idx:global_idx + n].astype(
                    np.uint16 if sh_bytes >= 2 else np.uint8)
                f.write(np.ascontiguousarray(sh_idx).tobytes())
            else:
                # norm565 / norm11 / fp16: store raw compressed SH
                if sh_compressed.ndim == 2:
                    f.write(np.ascontiguousarray(
                        sh_compressed[global_idx:global_idx + n]).tobytes())
                else:
                    f.write(np.ascontiguousarray(
                        sh_compressed[global_idx:global_idx + n, np.newaxis]).tobytes())
            global_idx += n


def _pack(buf, offset, arr):
    """Write a numpy array (1D) into a bytearray at offset, byte-aligned."""
    nbytes = arr.dtype.itemsize * arr.size
    buf[offset:offset + nbytes] = np.ascontiguousarray(arr).tobytes()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main compression pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def compress(ply_path, quality='medium', output_dir=None,
             prune_opacity=None, prune_screen_size=None, no_reorder=False):
    """Run the full compression pipeline."""
    ply_path = Path(ply_path)
    if not ply_path.exists():
        print(f"ERROR: {ply_path} not found")
        return None

    if output_dir is None:
        output_dir = ply_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = QUALITY_PRESETS.get(quality)
    if preset is None:
        print(f"ERROR: Unknown quality '{quality}'. Options: {list(QUALITY_PRESETS.keys())}")
        return None

    base_name = ply_path.stem
    gsplat_path = output_dir / f"{base_name}_compressed.gsplat"
    compat_ply_path = output_dir / f"{base_name}_compressed.ply"

    original_mb = ply_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*65}")
    print(f"  Model compression: {ply_path.name}")
    print(f"  Quality: {quality} ({preset['desc']})")
    print(f"  Original size: {original_mb:.1f} MB")
    print(f"{'='*65}")

    # ── 1. Load ──
    tensors = load_ply(ply_path)
    if tensors is None:
        return None

    N = len(tensors['positions'])
    t_start = time.time()

    # ── 2. Prune Gaussians ──
    if prune_opacity is not None or prune_screen_size is not None:
        print(f"\n  Pruning:")
        keep = np.ones(N, dtype=bool)
        if prune_opacity is not None:
            opacity = 1.0 / (1.0 + np.exp(-tensors['opacity']))
            op_mask = opacity > prune_opacity
            print(f"    Opacity > {prune_opacity}: {op_mask.sum():,}/{N:,}")
            keep &= op_mask
        if prune_screen_size is not None:
            # approximate: screen size ~ exp(scale_mean) in pixels at typical focal length
            scale_mean = tensors['scales'].mean(axis=1)
            ss_mask = np.exp(scale_mean) < prune_screen_size * 5  # rough heuristic
            print(f"    Screen-size estimate < {prune_screen_size}: {ss_mask.sum():,}/{N:,}")
            keep &= ss_mask
        for k in tensors:
            tensors[k] = tensors[k][keep]
        print(f"    Kept: {keep.sum():,} / {N:,} ({100*keep.sum()/N:.1f}%)")
        N = keep.sum()

    # ── 3. SH compression ──
    sh_mode = preset['sh_mode']
    n_clusters = preset['n_sh_clusters']
    cluster_comp = None

    print(f"\n  SH compression mode: {sh_mode}", end='')
    if sh_mode == 'cluster':
        print(f" ({n_clusters} clusters)")
    else:
        print()

    sh_compressed, cluster_comp, sh_meta = compress_sh(
        tensors['sh_rest'], sh_mode, n_clusters=n_clusters
    )

    palette = None
    if cluster_comp is not None and cluster_comp.centroids is not None:
        palette = cluster_comp.centroids.astype(np.float32)
        # Decode SH for compat PLY
        tensors['sh_rest'] = cluster_comp.decompress(
            sh_compressed).astype(np.float32)

    # ── 4. Morton reorder ──
    if not no_reorder:
        print(f"\n  Morton reordering...")
        t_m = time.time()
        # positions are the primary key for ordering
        pos = tensors['positions']
        codes = morton_code(pos)
        order = np.argsort(codes)
        for k in tensors:
            tensors[k] = tensors[k][order]
        if isinstance(sh_compressed, np.ndarray):
            sh_compressed = sh_compressed[order]
        print(f"    Done in {time.time()-t_m:.3f}s")

    # Reorder SH compressed to match
    if isinstance(sh_compressed, np.ndarray) and len(sh_compressed) != N:
        pass  # already handled above

    # ── 5. Chunk + normalize per-chunk ──
    print(f"\n  Chunking ({CHUNK_SIZE} per chunk)...")
    # Split positions into chunks for normalization reference
    # We need to clone the raw float data before chunking normalizes it
    raw_tensors = {k: v.copy() for k, v in tensors.items() if k in CHUNK_FIELDS}
    raw_tensors['quats'] = tensors['quats'].copy()

    chunks = split_chunks(raw_tensors, CHUNK_SIZE)
    print(f"    {len(chunks)} chunks (last: {len(chunks[-1]['positions'])} gaussians)")

    bit_widths = {
        'positions': preset['chunk_pos_bits'],
        'scales': preset['chunk_scale_bits'],
        'colors_dc': preset['chunk_color_bits'],
        'opacity': preset['chunk_opacity_bits'],
    }
    print(f"    Per-chunk bit widths: {bit_widths}")

    for idx, ch in enumerate(chunks):
        meta = normalize_chunk(ch, CHUNK_FIELDS, bit_widths)
        ch['_meta'] = meta
        # Keep raw quats for binary export (already [0,1] from sigmoid)
        ch['quats'] = raw_tensors['quats'][idx * CHUNK_SIZE:
                                            min((idx + 1) * CHUNK_SIZE, N)]
        # Normalize quats to [0,1] (they're unit, but may have neg values)
        qr = ch['quats'].astype(np.float64)
        qmin = qr.min(axis=0)
        qspan = np.ptp(qr, axis=0)
        qspan[qspan < 1e-10] = 1.0
        ch['quats'] = ((qr - qmin) / qspan).astype(np.float32)
        ch['_meta']['quats'] = (qmin.astype(np.float32), (qmin + qspan).astype(np.float32))

    # ── 6. Export binary .gsplat ──
    print(f"\n  Exporting binary .gsplat...")
    export_gsplat_binary(
        chunks, sh_compressed, sh_meta, palette, quality, preset, gsplat_path
    )

    # ── 7. Export compatibility PLY (decompressed) ──
    print(f"\n  Exporting compatibility PLY...")
    export_ply(tensors, compat_ply_path)

    # ── Stats ──
    elapsed = time.time() - t_start
    gsplat_mb = gsplat_path.stat().st_size / (1024 * 1024)
    compat_mb = compat_ply_path.stat().st_size / (1024 * 1024)
    # Rough estimate: gsplat doesn't store full SH, so raw-equivalent is PLY size
    ratio = original_mb / gsplat_mb if gsplat_mb > 0 else float('inf')

    print(f"\n{'='*65}")
    print(f"  Compression complete ({elapsed:.1f}s)")
    print(f"  Original PLY:         {original_mb:>8.1f} MB")
    print(f"  Compat PLY:           {compat_mb:>8.1f} MB  ({original_mb/compat_mb:.1f}x)")
    print(f"  Compressed .gsplat:   {gsplat_mb:>8.1f} MB  ({ratio:.1f}x)")
    print(f"  Gaussians:            {N:,}")
    print(f"  Chunks:               {len(chunks)}")
    print(f"{'='*65}")

    return {
        'input': str(ply_path),
        'quality': quality,
        'gsplat': str(gsplat_path),
        'compat_ply': str(compat_ply_path),
        'original_mb': original_mb,
        'gsplat_mb': gsplat_mb,
        'ratio': ratio,
        'n_gaussians': N,
        'n_chunks': len(chunks),
        'time_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compress 3D Gaussian Splatting PLY models with "
                    "per-chunk quantization and SH clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compress_model.py output/model.ply --quality medium
  python scripts/compress_model.py output/model.ply --quality low -o compressed/
  python scripts/compress_model.py output/model.ply --quality very_low
  python scripts/compress_model.py output/model.ply --standard
  python scripts/compress_model.py output/model.ply --prune-opacity 0.01
  python scripts/compress_model.py --list-qualities
        """
    )
    parser.add_argument("ply", nargs="?", help="Path to trained PLY file")
    parser.add_argument("--quality", "-q", default="medium",
                        choices=list(QUALITY_PRESETS.keys()),
                        help="Compression quality preset")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--standard", action="store_true",
                        help="Export standard 32B/splat .splat instead of .gsplat")
    parser.add_argument("--prune-opacity", type=float, default=None,
                        help="Remove Gaussians with sigmoid(opacity) below threshold")
    parser.add_argument("--prune-screen-size", type=float, default=None,
                        help="Remove Gaussians above screen-size estimate threshold")
    parser.add_argument("--no-reorder", action="store_true",
                        help="Skip Morton reordering (keep original order)")
    parser.add_argument("--list-qualities", action="store_true",
                        help="List available quality presets")

    args = parser.parse_args()

    if args.list_qualities:
        print("Available quality presets:\n")
        for name, cfg in QUALITY_PRESETS.items():
            sh_b = _sh_bytes_per_splat(cfg['sh_mode'], cfg['n_sh_clusters'])
            total_b = (
                (cfg['chunk_pos_bits'] * 3 +
                 cfg['chunk_scale_bits'] * 3 +
                 cfg['chunk_color_bits'] * 3 +
                 cfg['chunk_opacity_bits'] +
                 32 +                     # rotation (4×uint8)
                 sh_b * 8                 # SH bits
                ) + 7
            ) // 8
            pal_kb = cfg['n_sh_clusters'] * 45 * 2 / 1024 if cfg['n_sh_clusters'] > 0 else 0
            pal_str = f' + {pal_kb:.0f}KB palette' if pal_kb > 0 else ''
            print(f"  {name:12s}  {cfg['desc']:30s}  ~{total_b:2d} B/splat{pal_str}")
        return 0

    if args.ply is None:
        candidates = sorted(Path(BASE / "output").glob("arena_3dgs*.ply"))
        candidates = [p for p in candidates if 'compressed' not in p.stem]
        if not candidates:
            print("No arena_3dgs*.ply files found. Train first, or specify a path.")
            return 1
        args.ply = str(candidates[-1])
        print(f"Using: {args.ply}")

    if args.standard:
        out_path = None
        if args.output_dir:
            out_path = str(Path(args.output_dir) / (Path(args.ply).stem + "_standard.splat"))
        result = export_standard_splat(args.ply, output_path=out_path)
        return 0 if result is not None else 1

    result = compress(
        args.ply,
        quality=args.quality,
        output_dir=args.output_dir,
        prune_opacity=args.prune_opacity,
        prune_screen_size=args.prune_screen_size,
        no_reorder=args.no_reorder,
    )
    return 0 if result is not None else 1


if __name__ == "__main__":
    import argparse
    sys.exit(main())
