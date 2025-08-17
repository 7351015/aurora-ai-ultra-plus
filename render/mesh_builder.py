"""
INFINITUS Mesh Builder
Generates triangle meshes for voxel chunks using simple face culling.
"""
from typing import List, Tuple

AIR_ID = 0

# Face normals and vertex offsets for a unit cube at origin
# Each face is two triangles (6 vertices)
_FACE_DEFS = {
    "px": {  # +X
        "normal": (1.0, 0.0, 0.0),
        "verts": [
            (1, 0, 0), (1, 1, 0), (1, 1, 1),
            (1, 0, 0), (1, 1, 1), (1, 0, 1),
        ],
    },
    "nx": {  # -X
        "normal": (-1.0, 0.0, 0.0),
        "verts": [
            (0, 0, 1), (0, 1, 1), (0, 1, 0),
            (0, 0, 1), (0, 1, 0), (0, 0, 0),
        ],
    },
    "py": {  # +Y (top)
        "normal": (0.0, 1.0, 0.0),
        "verts": [
            (0, 1, 1), (1, 1, 1), (1, 1, 0),
            (0, 1, 1), (1, 1, 0), (0, 1, 0),
        ],
    },
    "ny": {  # -Y (bottom)
        "normal": (0.0, -1.0, 0.0),
        "verts": [
            (0, 0, 0), (1, 0, 0), (1, 0, 1),
            (0, 0, 0), (1, 0, 1), (0, 0, 1),
        ],
    },
    "pz": {  # +Z
        "normal": (0.0, 0.0, 1.0),
        "verts": [
            (0, 0, 1), (1, 0, 1), (1, 1, 1),
            (0, 0, 1), (1, 1, 1), (0, 1, 1),
        ],
    },
    "nz": {  # -Z
        "normal": (0.0, 0.0, -1.0),
        "verts": [
            (0, 1, 0), (1, 1, 0), (1, 0, 0),
            (0, 1, 0), (1, 0, 0), (0, 0, 0),
        ],
    },
}


def _block_color(block_id: int) -> Tuple[float, float, float]:
    """Map block id to a display color."""
    if block_id == 1:  # stone
        return (0.5, 0.5, 0.55)
    if block_id == 2:  # dirt
        return (0.4, 0.25, 0.15)
    if block_id == 3:  # grass
        return (0.2, 0.6, 0.2)
    return (0.6, 0.6, 0.6)  # default


def build_chunk_mesh(blocks: List[List[List[int]]]) -> Tuple[List[float], List[float]]:
    """Build a triangle mesh for a 16xH*16 chunk blocks array.

    Returns:
        positions: [x, y, z, ...] float list
        colors: [r, g, b, ...] float list (per vertex)
    """
    if not blocks:
        return [], []

    width = len(blocks)
    height = len(blocks[0]) if width > 0 else 0
    depth = len(blocks[0][0]) if height > 0 else 0

    positions: List[float] = []
    colors: List[float] = []

    def is_air(ix: int, iy: int, iz: int) -> bool:
        if ix < 0 or iy < 0 or iz < 0 or ix >= width or iy >= height or iz >= depth:
            return True
        try:
            return blocks[ix][iy][iz] == AIR_ID
        except Exception:
            return True

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                bid = blocks[x][y][z]
                if bid == AIR_ID:
                    continue
                color = _block_color(bid)

                # Check neighbors and emit faces if exposed
                if is_air(x + 1, y, z):  # +X
                    _emit_face(positions, colors, x, y, z, "px", color)
                if is_air(x - 1, y, z):  # -X
                    _emit_face(positions, colors, x, y, z, "nx", color)
                if is_air(x, y + 1, z):  # +Y
                    _emit_face(positions, colors, x, y, z, "py", color)
                if is_air(x, y - 1, z):  # -Y
                    _emit_face(positions, colors, x, y, z, "ny", color)
                if is_air(x, y, z + 1):  # +Z
                    _emit_face(positions, colors, x, y, z, "pz", color)
                if is_air(x, y, z - 1):  # -Z
                    _emit_face(positions, colors, x, y, z, "nz", color)

    return positions, colors


def _emit_face(positions: List[float], colors: List[float], bx: int, by: int, bz: int, face_key: str, color: Tuple[float, float, float]):
    f = _FACE_DEFS[face_key]
    for vx, vy, vz in f["verts"]:
        positions.extend([bx + vx, by + vy, bz + vz])
        colors.extend([color[0], color[1], color[2]])