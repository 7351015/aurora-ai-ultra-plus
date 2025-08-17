"""
INFINITUS First-Person Camera
Simple first-person camera supporting yaw/pitch, view and projection matrices.
"""
import math
from typing import Tuple, Optional

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # Fallback: build tuples without numpy when unavailable


class FirstPersonCamera:
    def __init__(self, fov_deg: float = 75.0, near: float = 0.1, far: float = 1000.0):
        self.position = (0.0, 1.8, 0.0)
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.fov_deg = fov_deg
        self.near = near
        self.far = far

    def set_pose(self, position: Tuple[float, float, float], yaw_deg: float, pitch_deg: float) -> None:
        self.position = position
        self.yaw_deg = yaw_deg
        self.pitch_deg = max(-89.0, min(89.0, pitch_deg))

    def add_rotation(self, dyaw: float, dpitch: float) -> None:
        self.yaw_deg = (self.yaw_deg + dyaw) % 360.0
        self.pitch_deg = max(-89.0, min(89.0, self.pitch_deg + dpitch))

    def forward_vector(self) -> Tuple[float, float, float]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        fx = math.cos(pitch) * math.sin(yaw)
        fy = math.sin(pitch)
        fz = math.cos(pitch) * math.cos(yaw)
        return (fx, fy, fz)

    def right_vector(self) -> Tuple[float, float, float]:
        yaw = math.radians(self.yaw_deg + 90.0)
        fx = math.cos(0.0) * math.sin(yaw)
        fz = math.cos(0.0) * math.cos(yaw)
        return (fx, 0.0, fz)

    def view_projection_flat(self, width: int, height: int) -> Tuple[float, ...]:
        view = self._view_matrix()
        proj = self._projection_matrix(width, height)
        return self._mul_flat(proj, view)

    def _view_matrix(self):
        ex, ey, ez = self.position
        fx, fy, fz = self.forward_vector()
        tx, ty, tz = ex + fx, ey + fy, ez + fz
        up = (0.0, 1.0, 0.0)
        return self._look_at((ex, ey, ez), (tx, ty, tz), up)

    def _projection_matrix(self, width: int, height: int):
        aspect = max(0.1, float(width) / float(max(1, height)))
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        n, fa = self.near, self.far
        m = [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (fa + n) / (n - fa), -1.0],
            [0.0, 0.0, (2.0 * fa * n) / (n - fa), 0.0],
        ]
        return m

    def _look_at(self, eye, target, up):
        ex, ey, ez = eye
        tx, ty, tz = target
        ux, uy, uz = up
        zx, zy, zz = ex - tx, ey - ty, ez - tz
        zl = max(1e-6, math.sqrt(zx*zx + zy*zy + zz*zz))
        zx, zy, zz = zx/zl, zy/zl, zz/zl
        xx = uy*zz - uz*zy
        xy = uz*zx - ux*zz
        xz = ux*zy - uy*zx
        xl = max(1e-6, math.sqrt(xx*xx + xy*xy + xz*xz))
        xx, xy, xz = xx/xl, xy/xl, xz/xl
        yx = zy*xz - zz*xy
        yy = zz*xx - zx*xz
        yz = zx*xy - zy*xx
        m = [
            [xx, yx, zx, 0.0],
            [xy, yy, zy, 0.0],
            [xz, yz, zz, 0.0],
            [-(xx*ex + xy*ey + xz*ez), -(yx*ex + yy*ey + yz*ez), -(zx*ex + zy*ey + zz*ez), 1.0],
        ]
        return m

    def _mul_flat(self, a, b) -> Tuple[float, ...]:
        # Multiply 4x4 matrices (a*b) and return flat tuple
        if _np is not None:
            arr = (_np.array(a) @ _np.array(b)).astype('f4').flatten()
            return tuple(float(x) for x in arr)
        # Fallback manual multiply
        res = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                s = 0.0
                for k in range(4):
                    s += a[i][k] * b[k][j]
                res[i][j] = s
        flat = []
        for i in range(4):
            for j in range(4):
                flat.append(res[i][j])
        return tuple(flat)