#!/usr/bin/env python3
"""
Post-training compression for 3D Gaussian Splatting models.

Implements techniques from Aras Pranckevičius's blog series:
  https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/

Key technique: SH Vector Quantization (clustering) + attribute quantization.
SH coefficients (45 per splat) are clustered into a palette of K centroids;
each splat stores a small index instead of 45 floats. Other attributes are
quantized to reduced-precision formats.

Usage:
  # Compress a trained PLY
  python scripts/compress_splat.py output/arena_3dgs.ply --quality medium

  # List available quality presets
  python scripts/compress_splat.py --list-qualities

  # Aggressive compression with SH clustering
  python scripts/compress_splat.py output/arena_3dgs.ply --quality low
"""
import os, sys, math, struct, time, functools
from pathlib import Path
import warnings
import numpy as np

# Force flush on every print (no buffering when piped)
print = functools.partial(print, flush=True)

BASE = Path(__file__).resolve().parent.parent

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData, PlyElement

# ═══════════════════════════════════════════════════════════════════════════════
#  SH Clustering — Mini-batch k-means (numpy-based)
# ═══════════════════════════════════════════════════════════════════════════════

class SHClusterCompressor:
    """Cluster SH rest coefficients (Nx45) into a palette of K centroids.

    Uses mini-batch k-means with chunked centroid processing to avoid O(n*k)
    memory blowup. Same approach as Aras's blog post.
    """
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
        """Find nearest centroid for each point, processing centroids in chunks
        to keep distance matrix at (n_points, cluster_chunk) instead of
        (n_points, n_clusters)."""
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

    def decompress(self, indices):
        return self.centroids[indices]

    def palette_bytes(self):
        return self.n_clusters * 45 * 2

    def index_bytes_per_splat(self):
        bits = max(8, 1 << (self.n_clusters - 1).bit_length())
        return bits // 8


# ═══════════════════════════════════════════════════════════════════════════════
#  Attribute Quantizers (numpy-based)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionQuantizer:
    """Quantize positions to local-space fixed-point.

    Transforms positions to a local coordinate frame (subtract center,
    divide by extent), then quantizes to a normalized integer format.
    """
    def __init__(self, bits=16):
        self.bits = bits
        self.center = None
        self.extent = None
        self.scale = None

    def fit(self, positions):
        self.center = positions.mean(axis=0).astype(np.float32)
        self.extent = float(np.ptp(positions, axis=0).max())
        self.extent = max(self.extent, 1e-6)
        self.scale = float((2 ** (self.bits - 1)) - 1)
        return self

    def compress(self, positions):
        normalized = (positions - self.center) / self.extent
        quantized = np.round(normalized * 2 * self.scale).astype(np.int16)
        return quantized

    def decompress(self, quantized):
        normalized = quantized.astype(np.float32) / (2 * self.scale)
        return normalized * self.extent + self.center

    def bytes_per_splat(self):
        return 3 * (self.bits // 8)


class ScaleQuantizer:
    """Quantize scales in log space.

    Scales are always positive. We take log, normalize to [0,1], and
    quantize to unsigned integer.
    """
    def __init__(self, bits=11):
        self.bits = bits
        self.log_min = None
        self.range = None

    def fit(self, scales):
        log_s = np.log(np.maximum(scales, 1e-10))
        self.log_min = float(log_s.min())
        log_max = float(log_s.max())
        self.range = max(log_max - self.log_min, 1e-10)
        return self

    def compress(self, scales):
        log_s = np.log(np.maximum(scales, 1e-10))
        normalized = (log_s - self.log_min) / self.range
        max_val = (2 ** self.bits) - 1
        quantized = np.round(np.clip(normalized, 0, 1) * max_val).astype(np.int32)
        return quantized

    def decompress(self, quantized):
        max_val = (2 ** self.bits) - 1
        normalized = quantized.astype(np.float32) / max_val
        log_s = normalized * self.range + self.log_min
        return np.exp(log_s)

    def bytes_per_splat(self):
        return 3 * (self.bits // 8)


class RotationQuantizer:
    """Quantize unit quaternions using largest-component packing.

    Drops the largest component (store index of which was dropped),
    quantize the remaining 3 to signed int. Pack the index into the
    last component's low bits.
    """
    def __init__(self, bits_per_comp=10):
        self.bits_per_comp = bits_per_comp
        self.max_val = float((2 ** (bits_per_comp - 1)) - 1)

    def compress(self, quats):
        N = quats.shape[0]
        q = quats.copy().astype(np.float64)

        # Ensure w is positive
        neg = q[:, 3] < 0
        if neg.any():
            q[neg] *= -1

        abs_q = np.abs(q)
        largest = abs_q.argmax(axis=1)

        comps = np.zeros((N, 3), dtype=np.float64)
        idx_map = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
        for drop_idx in range(4):
            mask = largest == drop_idx
            if mask.any():
                comps[mask] = q[np.ix_(mask, idx_map[drop_idx])]

        quantized = np.round(np.clip(comps, -1, 1) * self.max_val).astype(np.int16)
        packed = quantized.astype(np.int32)
        packed[:, 2] = (packed[:, 2] << 2) | largest.astype(np.int32)
        return packed

    def decompress(self, packed):
        N = packed.shape[0]
        p = packed.astype(np.int32)
        largest = p[:, 2] & 3
        comps = p.astype(np.float64) / self.max_val
        comps[:, 2] = (p[:, 2] >> 2).astype(np.float64) / self.max_val

        q = np.zeros((N, 4), dtype=np.float64)
        idx_map = {
            0: [1, 2, 3, 0], 1: [0, 2, 3, 1],
            2: [0, 1, 3, 2], 3: [0, 1, 2, 3]
        }
        for drop_idx in range(4):
            mask = largest == drop_idx
            if mask.any():
                q[np.ix_(mask, idx_map[drop_idx])] = comps[mask]

        q_sq = (q * q).sum(axis=1)
        missing = np.sqrt(np.maximum(1 - q_sq, 0))
        for drop_idx in range(4):
            mask = largest == drop_idx
            if mask.any():
                q[mask, drop_idx] = missing[mask]

        norm = np.maximum(np.linalg.norm(q, axis=1), 1e-10)
        q /= norm[:, np.newaxis]
        return q.astype(np.float32)

    def bytes_per_splat(self):
        return 4 * (self.bits_per_comp // 8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Quality Presets
# ═══════════════════════════════════════════════════════════════════════════════

QUALITY_PRESETS = {
    'very_high': {
        'desc': 'Minimal loss, ~2x compression',
        'sh_mode': 'fp16',
        'pos_bits': 16,
        'scale_bits': 16,
        'rot_bits': 16,
        'color_mode': 'fp16',
        'opacity_mode': 'fp16',
        'n_sh_clusters': 0,
    },
    'high': {
        'desc': 'Low loss, ~3x compression',
        'sh_mode': 'norm11',
        'pos_bits': 16,
        'scale_bits': 11,
        'rot_bits': 10,
        'color_mode': 'fp16',
        'opacity_mode': 'fp16',
        'n_sh_clusters': 0,
    },
    'medium': {
        'desc': 'Balanced, ~5x compression',
        'sh_mode': 'norm565',
        'pos_bits': 11,
        'scale_bits': 11,
        'rot_bits': 10,
        'color_mode': 'norm8',
        'opacity_mode': 'fp16',
        'n_sh_clusters': 0,
    },
    'low': {
        'desc': 'SH clustering 16K, ~15x compression',
        'sh_mode': 'cluster',
        'pos_bits': 11,
        'scale_bits': 6,
        'rot_bits': 10,
        'color_mode': 'norm8',
        'opacity_mode': 'fp16',
        'n_sh_clusters': 16384,
    },
    'very_low': {
        'desc': 'SH clustering 4K, ~18x compression',
        'sh_mode': 'cluster',
        'pos_bits': 11,
        'scale_bits': 6,
        'rot_bits': 10,
        'color_mode': 'norm8',
        'opacity_mode': 'fp16',
        'n_sh_clusters': 4096,
    },
}


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
    """Export a standard-format PLY from a dict of numpy arrays.

    The arrays should contain float32 data in standard 3DGS fields.
    """
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


def export_binary_splat(tensors, metadata, path):
    """Export a custom compressed binary format (.splat).

    Format:
      [4B magic 'SPLT'] [2B version] [4B n_gaussians] [4B n_sh_clusters]
      [12B scene_center: 3xfloat32] [4B scene_extent: float32]
      [1B pos_bits] [1B scale_bits] [1B rot_bits] [1B sh_mode]
      [4B palette_size_bytes] [palette]
      [per-gaussian data: positions, SH, colors_dc, opacity, scales, quats]
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    pos_q = tensors.get('positions_q', tensors['positions'].astype(np.float16))
    sh_data = tensors.get('sh_compressed', tensors['sh_rest'].astype(np.float16))
    palette = tensors.get('sh_palette', None)
    colors_q = tensors.get('colors_q', tensors['colors_dc'].astype(np.float16))
    opacity_q = tensors.get('opacity_q', tensors['opacity'].astype(np.float16))
    scales_q = tensors.get('scales_q', tensors['scales'].astype(np.float16))
    quats_q = tensors.get('quats_q', tensors['quats'].astype(np.float16))

    palette_bytes = palette.tobytes() if palette is not None else b''

    with open(path, 'wb') as f:
        f.write(b'SPLT')
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', len(tensors['positions'])))
        f.write(struct.pack('<I', metadata.get('n_sh_clusters', 0)))
        center = metadata.get('center', tensors['positions'].mean(axis=0))
        extent = metadata.get('extent', float(np.ptp(tensors['positions'], axis=0).max()))
        f.write(struct.pack('<fff', float(center[0]), float(center[1]), float(center[2])))
        f.write(struct.pack('<f', float(extent)))
        f.write(struct.pack('B', metadata.get('pos_bits', 16)))
        f.write(struct.pack('B', metadata.get('scale_bits', 16)))
        f.write(struct.pack('B', metadata.get('rot_bits', 16)))
        f.write(struct.pack('B', {'fp16': 0, 'norm11': 1, 'norm565': 2, 'cluster': 3}
                            .get(metadata.get('sh_mode', 'fp16'), 0)))
        f.write(struct.pack('<I', len(palette_bytes)))
        f.write(palette_bytes)
        f.write(np.ascontiguousarray(pos_q).tobytes())
        f.write(np.ascontiguousarray(sh_data).tobytes())
        f.write(np.ascontiguousarray(colors_q).tobytes())
        f.write(np.ascontiguousarray(opacity_q).tobytes())
        f.write(np.ascontiguousarray(scales_q).tobytes())
        f.write(np.ascontiguousarray(quats_q).tobytes())

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Exported: {path} ({size_mb:.1f} MB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Compression Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_per_splat_budget(preset_name):
    """Estimate per-splat bytes for a given quality preset."""
    cfg = QUALITY_PRESETS[preset_name]
    sh_bytes = {'fp16': 90, 'norm11': 90, 'norm565': 30, 'cluster': 2}[cfg['sh_mode']]
    pos_bytes = 6 if cfg['pos_bits'] == 16 else 3 * (cfg['pos_bits'] // 8) or 6
    scale_bytes = 6 if cfg['scale_bits'] == 16 else 3 * (cfg['scale_bits'] // 8) or 4
    rot_bytes = 8 if cfg['rot_bits'] == 16 else 4 * (cfg['rot_bits'] // 8) or 4
    color_bytes = {'fp16': 6, 'norm8': 3}[cfg['color_mode']]
    opacity_bytes = 2
    total = pos_bytes + sh_bytes + color_bytes + opacity_bytes + scale_bytes + rot_bytes
    palette_kb = cfg['n_sh_clusters'] * 45 * 2 / 1024 if cfg['n_sh_clusters'] > 0 else 0
    return total, palette_kb


def compress_sh_rest(sh_rest, mode, n_clusters=0, cluster_compressor=None):
    """Compress SH rest coefficients.

    Args:
        sh_rest: (N, 45) numpy array
        mode: 'fp16', 'norm11', 'norm565', or 'cluster'
        n_clusters: number of SH clusters (for 'cluster' mode)
        cluster_compressor: pre-fitted SHClusterCompressor (optional)

    Returns:
        (compressed_data, compressor, metadata_dict)
    """
    if mode == 'fp16':
        return sh_rest.astype(np.float16), None, {'mode': 'fp16', 'bytes_per_splat': 90}

    elif mode == 'norm11':
        max_val = (2 ** 10) - 1  # 11-bit signed: 10 bits + sign
        quantized = np.round(np.clip(sh_rest, -1, 1) * max_val).astype(np.int16)
        return quantized, None, {'mode': 'norm11', 'bytes_per_splat': 45 * 2}

    elif mode == 'norm565':
        N, D = sh_rest.shape
        assert D == 45
        sh_3d = sh_rest.reshape(N, 15, 3)
        r = (np.clip(sh_3d[:, :, 0], -1, 1) * 15).round().astype(np.int32) & 0x1F
        g = (np.clip(sh_3d[:, :, 1], -1, 1) * 31).round().astype(np.int32) & 0x3F
        b = (np.clip(sh_3d[:, :, 2], -1, 1) * 15).round().astype(np.int32) & 0x1F
        packed = ((r << 11) | (g << 5) | b).astype(np.uint16)
        return packed, None, {'mode': 'norm565', 'bytes_per_splat': 15 * 2}

    elif mode == 'cluster':
        if cluster_compressor is None:
            cluster_compressor = SHClusterCompressor(n_clusters=n_clusters)
            cluster_compressor.fit(sh_rest)
        indices = cluster_compressor.compress(sh_rest)
        return indices, cluster_compressor, {
            'mode': 'cluster',
            'n_clusters': n_clusters,
            'bytes_per_splat': 2,
            'palette_bytes': cluster_compressor.palette_bytes(),
        }

    else:
        raise ValueError(f"Unknown SH mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Compression
# ═══════════════════════════════════════════════════════════════════════════════

def compress(ply_path, quality='medium', output_dir=None):
    """Compress a trained 3DGS PLY file.

    Args:
        ply_path: Path to trained PLY file
        quality: Quality preset name
        output_dir: Output directory (default: same as input)

    Returns:
        dict with paths and stats, or None on error
    """
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
    compressed_ply_path = output_dir / f"{base_name}_compressed.ply"
    binary_path = output_dir / f"{base_name}_compressed.splat"

    original_mb = ply_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  Compressing: {ply_path.name}")
    print(f"  Quality: {quality} ({preset['desc']})")
    print(f"  Original: {original_mb:.1f} MB")
    print(f"{'='*60}")

    tensors = load_ply(ply_path)
    if tensors is None:
        return None

    N = len(tensors['positions'])
    t_start = time.time()

    # ── Position quantization ────────────────────────────────────────
    pos_bits = preset['pos_bits']
    if pos_bits == 16:
        tensors['positions_q'] = tensors['positions'].astype(np.float16)
        pos_meta = {
            'pos_bits': 16,
            'center': tensors['positions'].mean(axis=0).astype(np.float32),
            'extent': float(np.ptp(tensors['positions'], axis=0).max()),
        }
    else:
        pq = PositionQuantizer(bits=pos_bits)
        pq.fit(tensors['positions'])
        tensors['positions_q'] = pq.compress(tensors['positions'])
        pos_meta = {'pos_bits': pos_bits, 'center': pq.center, 'extent': pq.extent}

    # ── Color DC quantization ────────────────────────────────────────
    if preset['color_mode'] == 'fp16':
        tensors['colors_q'] = tensors['colors_dc'].astype(np.float16)
    else:
        tensors['colors_q'] = np.round(np.clip(tensors['colors_dc'], 0, 1) * 255).astype(np.uint8)

    # ── Opacity quantization ─────────────────────────────────────────
    tensors['opacity_q'] = tensors['opacity'].astype(np.float16)

    # ── Scale quantization ───────────────────────────────────────────
    scale_bits = preset['scale_bits']
    if scale_bits >= 16:
        tensors['scales_q'] = tensors['scales'].astype(np.float16)
    else:
        sq = ScaleQuantizer(bits=scale_bits)
        sq.fit(tensors['scales'])
        tensors['scales_q'] = sq.compress(tensors['scales'])

    # ── Rotation quantization ────────────────────────────────────────
    rot_bits = preset['rot_bits']
    if rot_bits >= 16:
        tensors['quats_q'] = tensors['quats'].astype(np.float16)
    else:
        rq = RotationQuantizer(bits_per_comp=rot_bits)
        tensors['quats_q'] = rq.compress(tensors['quats'])

    # ── SH compression (biggest win) ─────────────────────────────────
    sh_mode = preset['sh_mode']
    n_clusters = preset['n_sh_clusters']
    cluster_comp = None

    print(f"\n  SH compression mode: {sh_mode}", end='')
    if sh_mode == 'cluster':
        print(f" ({n_clusters} clusters)")
    else:
        print()

    sh_compressed, cluster_comp, sh_meta = compress_sh_rest(
        tensors['sh_rest'], sh_mode, n_clusters=n_clusters
    )
    tensors['sh_compressed'] = sh_compressed

    if cluster_comp is not None and cluster_comp.centroids is not None:
        tensors['sh_palette'] = cluster_comp.centroids.astype(np.float16)
        sh_meta['palette_bytes'] = len(tensors['sh_palette'].tobytes())
        indices = sh_compressed
        decoded = cluster_comp.decompress(indices)
        tensors['sh_rest'] = decoded.astype(np.float32)

    # ── Stats ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    per_splat_budget, palette_kb = compute_per_splat_budget(quality)
    compressed_size_nb = N * per_splat_budget + palette_kb * 1024
    compressed_size_mb = compressed_size_nb / (1024 * 1024)

    print(f"\n  Compression stats:")
    print(f"    Per-splat budget:  {per_splat_budget} bytes")
    print(f"    Palette overhead:  {palette_kb:.1f} KB")
    print(f"    Estimated size:    {compressed_size_mb:.1f} MB")
    print(f"    Compression ratio: {original_mb / compressed_size_mb:.1f}x")
    print(f"    Time:              {elapsed:.1f}s")

    # ── Export compatibility PLY ─────────────────────────────────────
    print(f"\n  Exporting compatibility PLY...")
    export_ply(tensors, compressed_ply_path)

    # ── Export binary format ─────────────────────────────────────────
    print(f"  Exporting binary .splat...")
    metadata = {
        'n_sh_clusters': n_clusters if sh_mode == 'cluster' else 0,
        'sh_mode': sh_mode,
        'pos_bits': pos_bits,
        'scale_bits': scale_bits,
        'rot_bits': rot_bits,
        **pos_meta,
    }
    if isinstance(pos_meta.get('center'), np.ndarray):
        metadata['center'] = pos_meta['center']
    if pos_meta.get('extent') is not None:
        metadata['extent'] = pos_meta['extent']
    export_binary_splat(tensors, metadata, binary_path)

    # ── Final summary ────────────────────────────────────────────────
    final_ply_mb = compressed_ply_path.stat().st_size / (1024 * 1024)
    final_bin_mb = binary_path.stat().st_size / (1024 * 1024)

    ratio_binary = original_mb / final_bin_mb if final_bin_mb > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"  Compression complete")
    print(f"  Original PLY:      {original_mb:.1f} MB")
    print(f"  Compressed PLY:    {final_ply_mb:.1f} MB "
          f"({original_mb / final_ply_mb:.1f}x)")
    print(f"  Binary .splat:     {final_bin_mb:.1f} MB "
          f"({ratio_binary:.1f}x)")
    print(f"{'='*60}")

    return {
        'input': str(ply_path),
        'quality': quality,
        'compressed_ply': str(compressed_ply_path),
        'binary': str(binary_path),
        'original_mb': original_mb,
        'compressed_ply_mb': final_ply_mb,
        'compressed_binary_mb': final_bin_mb,
        'ratio_binary': ratio_binary,
        'n_gaussians': N,
        'time_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compress 3D Gaussian Splatting models using SH clustering "
                    "and attribute quantization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compress_splat.py output/arena_3dgs.ply --quality medium
  python scripts/compress_splat.py output/arena_3dgs.ply --quality low
  python scripts/compress_splat.py --list-qualities
        """
    )
    parser.add_argument("ply", nargs="?", help="Path to trained PLY file")
    parser.add_argument("--quality", "-q", default="medium",
                        choices=list(QUALITY_PRESETS.keys()),
                        help="Compression quality preset")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--list-qualities", action="store_true",
                        help="List available quality presets")

    args = parser.parse_args()

    if args.list_qualities:
        print("Available quality presets:\n")
        for name in QUALITY_PRESETS:
            budget, palette_kb = compute_per_splat_budget(name)
            cfg = QUALITY_PRESETS[name]
            pal_str = f' + {palette_kb:.0f}KB palette' if palette_kb > 0 else ''
            print(f"  {name:12s}  {cfg['desc']:40s}  ~{budget:3d} B/splat{pal_str}")
        return 0

    if args.ply is None:
        candidates = sorted(Path(BASE / "output").glob("arena_3dgs*.ply"))
        candidates = [p for p in candidates if 'compressed' not in p.stem]
        if not candidates:
            print("No arena_3dgs*.ply files found. Train first, or specify a path.")
            return 1
        args.ply = str(candidates[-1])
        print(f"Using: {args.ply}")

    result = compress(args.ply, quality=args.quality, output_dir=args.output_dir)
    return 0 if result is not None else 1


if __name__ == "__main__":
    import argparse
    sys.exit(main())
