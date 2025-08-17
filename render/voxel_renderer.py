"""
INFINITUS Voxel Renderer (Optional)
A lightweight Moderngl-based voxel renderer with graceful fallback when OpenGL
is unavailable. Designed for offscreen or windowed rendering depending on host.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple


class VoxelRenderer:
    """Simple voxel renderer using moderngl if available.

    This class encapsulates setup and draw calls. If moderngl or a GL context is
    not available, it disables itself cleanly so the rest of the game continues
    to run without rendering 3D output.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.available: bool = False
        self._mgl = None
        self._np = None
        self._ctx = None
        self._prog = None
        self._vao = None
        self._vbo = None
        self._cbo = None
        self._nverts: int = 0
        self._brightness: float = 1.0

    def initialize(self) -> None:
        try:
            import moderngl as mgl  # type: ignore
            import numpy as np  # type: ignore
            self._mgl = mgl
            self._np = np
            # Create a standalone context (offscreen). If a windowed context is
            # desired, integrate with a windowing toolkit separately.
            self._ctx = mgl.create_standalone_context()
            self._ctx.enable(mgl.DEPTH_TEST)

            self._prog = self._ctx.program(
                vertex_shader=self._VERT_SRC,
                fragment_shader=self._FRAG_SRC,
            )
            self.available = True
            self.logger.info("🧊 VoxelRenderer initialized (standalone context)")
        except Exception as e:
            self.available = False
            self.logger.warning(f"⚠️ VoxelRenderer unavailable: {e}")

    def load_mesh(self, positions: list[float], colors: list[float]) -> None:
        if not self.available:
            return
        import numpy as np  # type: ignore
        pos_arr = np.array(positions, dtype='f4')
        col_arr = np.array(colors, dtype='f4')
        if self._vbo is not None:
            self._vbo.release()
        if self._cbo is not None:
            self._cbo.release()
        self._vbo = self._ctx.buffer(pos_arr.tobytes())
        self._cbo = self._ctx.buffer(col_arr.tobytes())
        if self._vao is not None:
            self._vao.release()
        self._vao = self._ctx.vertex_array(
            self._prog,
            [
                (self._vbo, '3f', 'in_pos'),
                (self._cbo, '3f', 'in_col'),
            ],
        )
        self._nverts = len(positions) // 3

    def render(self, view_proj: tuple[float, ...], brightness: Optional[float] = None) -> None:
        if not self.available or self._vao is None:
            return
        # Set uniforms
        try:
            self._prog['u_mvp'].write(self._np.array(view_proj, dtype='f4').tobytes())
        except Exception:
            pass
        try:
            if brightness is not None:
                self._brightness = float(brightness)
            self._prog['u_brightness'].value = self._brightness
        except Exception:
            pass
        self._ctx.clear(0.05, 0.05, 0.1, 1.0)
        self._vao.render(self._mgl.TRIANGLES, vertices=self._nverts)

    _VERT_SRC = """
        #version 330
        in vec3 in_pos;
        in vec3 in_col;
        out vec3 v_col;
        uniform mat4 u_mvp;
        void main() {
            v_col = in_col;
            gl_Position = u_mvp * vec4(in_pos, 1.0);
        }
    """

    _FRAG_SRC = """
        #version 330
        in vec3 v_col;
        out vec4 f_color;
        uniform float u_brightness;
        void main() {
            f_color = vec4(v_col * u_brightness, 1.0);
        }
    """