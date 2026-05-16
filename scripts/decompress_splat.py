#!/usr/bin/env python3
"""
Decompress and view .splat binary files.

Reads the custom compressed .splat format, decompresses all attributes
back to float32, and provides multiple output modes:
  - OpenGL viewer (pyglet, default)
  - Standard PLY export
  - Statistics report

Usage:
  # View a compressed .splat file
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat

  # Export to standard PLY
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat --export

  # Show statistics only
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat --stats

  # Validate PLY compatibility
  python scripts/decompress_splat.py output/arena_3dgs_compressed.ply --validate
"""
import os, sys, math, struct, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
#  Format constants — must match compress_splat.py
# ═══════════════════════════════════════════════════════════════════════════════

SH_MODE_NAMES = {0: 'fp16', 1: 'norm11', 2: 'norm565', 3: 'cluster'}
SH_MODE_CODES = {v: k for k, v in SH_MODE_NAMES.items()}

def read_exactly(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data


def _read_array(f, dtype, count):
    nbytes = count * np.dtype(dtype).itemsize
    return np.frombuffer(read_exactly(f, nbytes), dtype=dtype)


def load_splat(path):
    """Load and decompress a .splat binary file.

    Returns dict of float32 arrays: positions, colors_dc, opacity,
    scales, quats, sh_rest (Nx45), plus metadata.
    """
    path = Path(path)
    print(f"Loading: {path}")
    t0 = time.time()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    with open(path, 'rb') as f:
        magic = read_exactly(f, 4)
        if magic != b'SPLT':
            raise ValueError(f"Bad magic: {magic}")
        version = struct.unpack('<H', read_exactly(f, 2))[0]
        n_gaussians = struct.unpack('<I', read_exactly(f, 4))[0]
        n_sh_clusters = struct.unpack('<I', read_exactly(f, 4))[0]
        center = struct.unpack('<fff', read_exactly(f, 12))
        extent = struct.unpack('<f', read_exactly(f, 4))[0]
        pos_bits = struct.unpack('B', read_exactly(f, 1))[0]
        scale_bits = struct.unpack('B', read_exactly(f, 1))[0]
        rot_bits = struct.unpack('B', read_exactly(f, 1))[0]
        sh_mode = struct.unpack('B', read_exactly(f, 1))[0]
        palette_size = struct.unpack('<I', read_exactly(f, 4))[0]

        print(f"  Version: {version}, Gaussians: {n_gaussians:,}")
        print(f"  SH clusters: {n_sh_clusters}, mode: {SH_MODE_NAMES.get(sh_mode, sh_mode)}")
        print(f"  Pos: {pos_bits}b, Scale: {scale_bits}b, Rot: {rot_bits}b")
        print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), extent: {extent:.2f}")

        # Read palette
        palette = None
        if palette_size > 0:
            palette_raw = np.frombuffer(read_exactly(f, palette_size), dtype=np.float16)
            n_palette = palette_size // (45 * 2)
            palette = palette_raw.reshape(n_palette, 45).astype(np.float32)
            print(f"  SH palette: {n_palette} entries x 45 coeffs")

        # Read per-gaussian data sequentially from file
        # Each field is read in order, same as export_binary_splat writes them

        # 1. Position
        if pos_bits == 16:
            pos_raw = _read_array(f, np.float16, n_gaussians * 3)
            positions = pos_raw.reshape(n_gaussians, 3).astype(np.float32)
        else:
            pos_raw = _read_array(f, np.int16, n_gaussians * 3)
            pos_raw = pos_raw.reshape(n_gaussians, 3).astype(np.float32)
            scale_val = float((2 ** (pos_bits - 1)) - 1)
            positions = pos_raw / (2 * scale_val) * extent + np.array(center, dtype=np.float32)

        # 2. SH data
        if sh_mode == 3:  # cluster
            sh_idx = _read_array(f, np.uint16, n_gaussians)
            sh_rest = palette[sh_idx] if palette is not None else np.zeros((n_gaussians, 45), dtype=np.float32)
            if palette is None:
                print("  WARNING: SH cluster mode but no palette found")
        elif sh_mode == 2:  # norm565
            packed = _read_array(f, np.uint16, n_gaussians * 15).reshape(n_gaussians, 15)
            r = ((packed >> 11) & 0x1F).astype(np.float32) / 15 * 2 - 1
            g = ((packed >> 5) & 0x3F).astype(np.float32) / 31 * 2 - 1
            b = (packed & 0x1F).astype(np.float32) / 15 * 2 - 1
            sh_rest = np.stack([r, g, b], axis=2).reshape(n_gaussians, 45)
        elif sh_mode == 1:  # norm11
            raw_sh = _read_array(f, np.int16, n_gaussians * 45).reshape(n_gaussians, 45).astype(np.float32)
            max_val = (2 ** 10) - 1
            sh_rest = np.clip(raw_sh / max_val, -1, 1)
        else:  # fp16
            sh_rest = _read_array(f, np.float16, n_gaussians * 45).reshape(n_gaussians, 45).astype(np.float32)

        # 3. Color DC — infer format from remaining data size
        # After SH: remaining = N * (color_sz + 2 + scale_sz + rot_sz)
        # Compute expected for both formats and pick the one that matches
        pos_after_sh = f.tell()
        scale_sz = 6 if scale_bits == 16 else 12
        rot_sz = 8 if rot_bits == 16 else 12
        fmt_fp16 = n_gaussians * (6 + 2 + scale_sz + rot_sz)
        fmt_uint8 = n_gaussians * (3 + 2 + scale_sz + rot_sz)
        remaining = path.stat().st_size - pos_after_sh
        if abs(remaining - fmt_fp16) < abs(remaining - fmt_uint8):
            colors_dc = _read_array(f, np.float16, n_gaussians * 3).reshape(n_gaussians, 3).astype(np.float32)
        else:
            colors_dc = _read_array(f, np.uint8, n_gaussians * 3).reshape(n_gaussians, 3).astype(np.float32) / 255

        # 4. Opacity
        opacity = _read_array(f, np.float16, n_gaussians).astype(np.float32)

        # 5. Scale
        if scale_bits == 16:
            scales = _read_array(f, np.float16, n_gaussians * 3).reshape(n_gaussians, 3).astype(np.float32)
        elif scale_bits == 6:
            s_int = _read_array(f, np.int32, n_gaussians * 3).reshape(n_gaussians, 3).astype(np.float32)
            scales = np.clip(s_int * 0.01, 1e-4, 1.0)
        else:
            s_int = _read_array(f, np.int32, n_gaussians * 3).reshape(n_gaussians, 3).astype(np.float32)
            s_max = (2 ** scale_bits) - 1
            scales = np.exp(s_int / s_max * 10 - 5)
            scales = np.clip(scales, 1e-4, 1.0)
            print(f"  NOTE: Scale dequantization is approximate (no fitted range stored)")

        # 6. Rotation
        if rot_bits == 16:
            quats = _read_array(f, np.float16, n_gaussians * 4).reshape(n_gaussians, 4).astype(np.float32)
        else:
            r_int = _read_array(f, np.int32, n_gaussians * 3).reshape(n_gaussians, 3)
            max_val = float((2 ** (rot_bits - 1)) - 1)
            largest = r_int[:, 2] & 3
            comps = r_int.astype(np.float32) / max_val
            comps[:, 2] = (r_int[:, 2] >> 2).astype(np.float32) / max_val
            quats = np.zeros((n_gaussians, 4), dtype=np.float32)
            idx_map = {0: [1, 2, 3], 1: [0, 2, 3],
                       2: [0, 1, 3], 3: [0, 1, 2]}
            for drop_idx in range(4):
                mask = largest == drop_idx
                if mask.any():
                    rows = np.where(mask)[0]
                    quats[rows[:, None], idx_map[drop_idx]] = comps[mask]
            q_sq = (quats * quats).sum(axis=1)
            missing = np.sqrt(np.maximum(1 - q_sq, 0))
            for drop_idx in range(4):
                mask = largest == drop_idx
                if mask.any():
                    quats[mask, drop_idx] = missing[mask]
            norm = np.maximum(np.linalg.norm(quats, axis=1), 1e-10)
            quats /= norm[:, np.newaxis]

    elapsed = time.time() - t0
    print(f"  Decompressed in {elapsed:.1f}s")

    return {
        'positions': positions,
        'colors_dc': colors_dc,
        'opacity': opacity,
        'scales': scales,
        'quats': quats,
        'sh_rest': sh_rest,
        'n_sh_clusters': n_sh_clusters,
        'sh_mode': SH_MODE_NAMES.get(sh_mode, 'unknown'),
    }


def print_stats(data):
    """Print statistics about decompressed gaussian data."""
    N = len(data['positions'])
    print(f"\n{'='*60}")
    print(f"  Gaussian Splat Statistics")
    print(f"{'='*60}")
    print(f"  Gaussians:        {N:,}")
    print(f"  Position bounds:  X[{data['positions'][:,0].min():.2f}, {data['positions'][:,0].max():.2f}]")
    print(f"                    Y[{data['positions'][:,1].min():.2f}, {data['positions'][:,1].max():.2f}]")
    print(f"                    Z[{data['positions'][:,2].min():.2f}, {data['positions'][:,2].max():.2f}]")
    print(f"  Scale range:      [{data['scales'].min():.6f}, {data['scales'].max():.4f}]")
    print(f"  Opacity range:    [{data['opacity'].min():.4f}, {data['opacity'].max():.4f}]")
    visible = (data['opacity'] > 0.01).sum()
    print(f"  Visible (>0.01):  {visible:,} ({100*visible/N:.1f}%)")
    print(f"  SH clusters:      {data.get('n_sh_clusters', 0)}")
    print(f"  SH mode:          {data.get('sh_mode', 'N/A')}")

    # Memory estimate
    mem = (data['positions'].nbytes + data['colors_dc'].nbytes +
           data['opacity'].nbytes + data['scales'].nbytes +
           data['quats'].nbytes + data['sh_rest'].nbytes)
    print(f"  Decompressed mem: {mem / (1024*1024):.1f} MB")
    print()


def export_to_ply(data, path):
    """Export decompressed gaussian data to standard 3DGS PLY."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("Installing plyfile...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "plyfile", "--break-system-packages", "--user"])
        from plyfile import PlyData, PlyElement

    N = len(data['positions'])
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
    elements['x'] = data['positions'][:, 0]
    elements['y'] = data['positions'][:, 1]
    elements['z'] = data['positions'][:, 2]
    elements['f_dc_0'] = data['colors_dc'][:, 0]
    elements['f_dc_1'] = data['colors_dc'][:, 1]
    elements['f_dc_2'] = data['colors_dc'][:, 2]
    for i in range(45):
        elements[f'f_rest_{i}'] = data['sh_rest'][:, i]
    elements['opacity'] = data['opacity']
    elements['scale_0'] = data['scales'][:, 0]
    elements['scale_1'] = data['scales'][:, 1]
    elements['scale_2'] = data['scales'][:, 2]
    elements['rot_0'] = data['quats'][:, 0]
    elements['rot_1'] = data['quats'][:, 1]
    elements['rot_2'] = data['quats'][:, 2]
    elements['rot_3'] = data['quats'][:, 3]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(str(path))
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  Exported: {path} ({mb:.1f} MB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenGL Viewer (pyglet)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import pyglet
    from pyglet import gl, window, clock
    from pyglet.graphics.shader import Shader, ShaderProgram
    HAS_PYGLET = True
except ImportError:
    HAS_PYGLET = False


def matrix_mult(A, B):
    """4x4 matrix multiply (A @ B)."""
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def look_at(eye, target, up):
    """Build a view matrix."""
    fwd = np.array([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]], dtype=np.float32)
    fwd /= max(np.linalg.norm(fwd), 1e-10)
    right = np.cross(fwd, np.array(up, dtype=np.float32))
    right /= max(np.linalg.norm(right), 1e-10)
    up2 = np.cross(right, fwd)
    return np.array([
        [right[0], right[1], right[2], -np.dot(right, eye)],
        [up2[0],   up2[1],   up2[2],   -np.dot(up2, eye)],
        [-fwd[0],  -fwd[1],  -fwd[2],  np.dot(fwd, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def perspective(fov_y, aspect, near, far):
    """Build a perspective projection matrix."""
    f = 1.0 / math.tan(fov_y / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)


VERTEX_SHADER_SRC = """
#version 330
in vec3 in_position;
in vec3 in_color;
in float in_opacity;
in vec3 in_scale;

uniform mat4 u_mvp;
uniform vec2 u_viewport;

out vec3 v_color;
out float v_opacity;
out float v_point_size;

void main() {
    vec4 clip_pos = u_mvp * vec4(in_position, 1.0);
    gl_Position = clip_pos;

    float z_dist = clip_pos.z;
    float scale_screen = length(in_scale) * u_viewport.y * 0.5 / max(-clip_pos.w, 1e-6);
    v_point_size = clamp(scale_screen * 2.0, 1.0, 500.0);
    gl_PointSize = v_point_size;

    v_color = in_color;
    v_opacity = in_opacity;
    v_point_size = v_point_size;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330
in vec3 v_color;
in float v_opacity;
in float v_point_size;
out vec4 frag_color;

void main() {
    vec2 mid = gl_PointCoord - vec2(0.5);
    float dist2 = dot(mid, mid);
    if (dist2 > 0.25) discard;
    float alpha = exp(-dist2 * 12.0) * v_opacity;
    frag_color = vec4(v_color * alpha, alpha);
}
"""





class SplatViewer:
    """Lightweight OpenGL viewer for decompressed gaussian splat data.

    Controls:
      Left drag:  orbit
      Scroll:     zoom
      R:          reset view
      Q / Esc:    quit
    """
    def __init__(self, data, width=1280, height=720, title="Gaussian Splat Viewer"):
        self.data = data
        self.N = len(data['positions'])

        # Camera state
        center = data['positions'].mean(axis=0)
        extent = float(np.ptp(data['positions'], axis=0).max())
        self.cam_dist = extent * 2.5
        self.cam_theta = math.radians(30)
        self.cam_phi = math.radians(30)
        self.cam_target = center
        self.cam_near = extent * 0.001
        self.cam_far = extent * 10

        self.last_mouse = None
        self.width = width
        self.height = height

        # Build pyglet window
        config = gl.Config(sample_buffers=1, samples=4, depth_size=24)
        self.window = window.Window(width=width, height=height,
                                     caption=title, config=config,
                                     resizable=True)
        self.window.on_resize = self.on_resize
        self.window.on_draw = self.on_draw
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.window.on_key_press = self.on_key_press

        # Compensate for retina display
        self.scale = self.window.get_framebuffer_size()[0] / max(self.width, 1)

        # Build OpenGL program
        vert = Shader(VERTEX_SHADER_SRC, 'vertex')
        frag = Shader(FRAGMENT_SHADER_SRC, 'fragment')
        self.program = ShaderProgram(vert, frag)
        self._setup_buffers()

        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.mouse_down = False

    def _setup_buffers(self):
        visible = self.data['opacity'] > 0.005
        self.visible_idx = np.where(visible)[0]
        M = len(self.visible_idx)
        print(f"  Visible gaussians for rendering: {M:,} / {self.N:,}")

        if M == 0:
            print("  WARNING: No visible gaussians!")
            self.vao = None
            return

        pos = self.data['positions'][self.visible_idx]
        color = self.data['colors_dc'][self.visible_idx]
        opacity = self.data['opacity'][self.visible_idx]
        scale = self.data['scales'][self.visible_idx]

        stride = 3 + 3 + 1 + 3
        vertex_data = np.empty(M * stride, dtype=np.float32)
        for i in range(3):
            vertex_data[i::stride] = pos[:, i]
        for i in range(3):
            vertex_data[3 + i::stride] = color[:, i]
        vertex_data[6::stride] = opacity
        for i in range(3):
            vertex_data[7 + i::stride] = scale[:, i]

        vao_id = gl.GLuint()
        gl.glGenVertexArrays(1, vao_id)
        gl.glBindVertexArray(vao_id)

        vbo_id = gl.GLuint()
        gl.glGenBuffers(1, vbo_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        vertex_data.nbytes,
                        vertex_data.ctypes.data,
                        gl.GL_STATIC_DRAW)

        stride_bytes = stride * 4
        pid = self.program.id
        gl.glUseProgram(pid)

        loc = gl.glGetAttribLocation(pid, b"in_position")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 stride_bytes, gl.GLvoidp(0))

        loc = gl.glGetAttribLocation(pid, b"in_color")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 stride_bytes, gl.GLvoidp(3 * 4))

        loc = gl.glGetAttribLocation(pid, b"in_opacity")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 1, gl.GL_FLOAT, gl.GL_FALSE,
                                 stride_bytes, gl.GLvoidp(6 * 4))

        loc = gl.glGetAttribLocation(pid, b"in_scale")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 stride_bytes, gl.GLvoidp(7 * 4))

        gl.glUseProgram(0)
        self.vao = vao_id
        self.vbo = vbo_id
        self.vertex_count = M

    def _compute_mvp(self):
        eye_x = self.cam_target[0] + self.cam_dist * math.sin(self.cam_theta) * math.cos(self.cam_phi)
        eye_y = self.cam_target[1] + self.cam_dist * math.cos(self.cam_theta)
        eye_z = self.cam_target[2] + self.cam_dist * math.sin(self.cam_theta) * math.sin(self.cam_phi)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)
        view = look_at(eye, self.cam_target, (0, 1, 0))
        aspect = self.width / max(self.height, 1)
        proj = perspective(math.radians(45), aspect, self.cam_near, self.cam_far)
        mvp = proj @ view
        return mvp

    def on_resize(self, w, h):
        self.width = max(w, 1)
        self.height = max(h, 1)
        self.scale = self.window.get_framebuffer_size()[0] / max(self.width, 1)
        gl.glViewport(0, 0, self.window.get_framebuffer_size()[0],
                      self.window.get_framebuffer_size()[1])

    def on_mouse_press(self, x, y, button, mods):
        if button == pyglet.window.mouse.LEFT:
            self.last_mouse = (x, y)
            self.mouse_down = True

    def on_mouse_release(self, x, y, button, mods):
        if button == pyglet.window.mouse.LEFT:
            self.mouse_down = False
            self.last_mouse = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        if self.mouse_down:
            self.cam_phi -= math.radians(dx * 0.3)
            self.cam_theta = np.clip(self.cam_theta + math.radians(dy * 0.3),
                                     math.radians(1), math.radians(179))

    def on_mouse_scroll(self, x, y, sx, sy):
        self.cam_dist *= (1 - sy * 0.08)
        self.cam_dist = np.clip(self.cam_dist, self.cam_near * 2, self.cam_far * 0.5)

    def on_key_press(self, key, mods):
        if key == pyglet.window.key.R:
            # Reset camera
            extent = float(np.ptp(self.data['positions'], axis=0).max())
            self.cam_dist = extent * 2.5
            self.cam_theta = math.radians(30)
            self.cam_phi = math.radians(30)
        elif key == pyglet.window.key.Q or key == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

    def on_draw(self):
        gl.glClearColor(0.08, 0.08, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self.vao is None or self.vertex_count == 0:
            return

        mvp = self._compute_mvp()
        pid = self.program.id
        gl.glUseProgram(pid)

        loc_mvp = gl.glGetUniformLocation(pid, b"u_mvp")
        gl.glUniformMatrix4fv(loc_mvp, 1, gl.GL_TRUE, mvp.ctypes.data)

        loc_vp = gl.glGetUniformLocation(pid, b"u_viewport")
        gl.glUniform2f(loc_vp, self.width * self.scale, self.height * self.scale)

        gl.glBindVertexArray(self.vao)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.vertex_count)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

        pyglet.clock.tick()
        fps = pyglet.clock.get_fps()
        self.window.set_caption(f"Gaussian Splat Viewer — {self.N:,} splats — {fps:.0f} FPS")

    def run(self):
        pyglet.app.run()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Decompress and view .splat (compressed 3DGS) files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat --export
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat --stats
  python scripts/decompress_splat.py output/arena_3dgs_compressed.splat --view-only
        """
    )
    parser.add_argument("input", help="Path to .splat or .ply file")
    parser.add_argument("--export", "-e", action="store_true",
                        help="Export decompressed data to PLY")
    parser.add_argument("--stats", "-s", action="store_true",
                        help="Show statistics only (no viewer)")
    parser.add_argument("--view-only", "-v", action="store_true",
                        help="Skip stats, just open viewer")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PLY path (default: input path + _decompressed.ply)")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        return 1

    # Handle .ply files directly (validate + show stats)
    if input_path.suffix == '.ply':
        try:
            from plyfile import PlyData
            ply = PlyData.read(str(input_path))
            v = ply['vertex'].data
            N = len(v)
            has_sh = all(f'f_rest_{i}' in v.dtype.names for i in range(45))
            print(f"PLY file: {input_path.name}")
            print(f"  Gaussians: {N:,}")
            print(f"  Full SH: {has_sh}")
            print(f"  Size: {input_path.stat().st_size / (1024*1024):.1f} MB")
            if not args.view_only and not args.export:
                view_data = {
                    'positions': np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32),
                    'colors_dc': np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).astype(np.float32),
                    'opacity': np.asarray(v['opacity'], dtype=np.float32),
                    'scales': np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']]).astype(np.float32),
                    'quats': np.column_stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).astype(np.float32),
                    'sh_rest': np.zeros((N, 45), dtype=np.float32),
                }
                if has_sh:
                    rest = np.column_stack([v[f'f_rest_{i}'] for i in range(45)]).astype(np.float32)
                    view_data['sh_rest'] = rest
                if not HAS_PYGLET:
                    print("pyglet not available, can't open viewer")
                    print_stats(view_data)
                else:
                    viewer = SplatViewer(view_data)
                    viewer.run()
            return 0
        except Exception as e:
            print(f"Error reading PLY: {e}")
            return 1

    # Load .splat
    data = load_splat(args.input)

    if not args.view_only or args.stats:
        print_stats(data)

    if args.export:
        out_path = args.output or str(input_path).replace('.splat', '_decompressed.ply')
        export_to_ply(data, out_path)

    if args.stats:
        return 0

    # Open viewer
    if not HAS_PYGLET:
        print("pyglet not available. Install with: pip install pyglet")
        print("Falling back to stats only.")
        return 0

    print("Opening viewer... (drag to orbit, scroll to zoom, R=reset, Q=quit)")
    viewer = SplatViewer(data)
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
