"""
INFINITUS Greedy Mesher
Builds a reduced triangle mesh for voxel chunks by merging coplanar faces.
This is a simplified greedy algorithm suitable for 16xH*16 chunks.
"""
from typing import List, Tuple

AIR_ID = 0


def build_chunk_mesh_greedy(blocks: List[List[List[int]]]) -> Tuple[List[float], List[float]]:
    if not blocks:
        return [], []
    width = len(blocks)
    height = len(blocks[0]) if width > 0 else 0
    depth = len(blocks[0][0]) if height > 0 else 0

    positions: List[float] = []
    colors: List[float] = []

    def color_for(bid: int) -> Tuple[float, float, float]:
        if bid == 1:
            return (0.5, 0.5, 0.55)
        if bid == 2:
            return (0.4, 0.25, 0.15)
        if bid == 3:
            return (0.2, 0.6, 0.2)
        return (0.6, 0.6, 0.6)

    # Greedy merge top faces (py) as an example; other faces fall back later
    for y in range(height):
        mask: List[Tuple[int, Tuple[float, float, float]]] = []
        for z in range(depth):
            row: List[Tuple[int, Tuple[float, float, float]]] = []
            for x in range(width):
                bid = blocks[x][y][z]
                top_air = (y + 1 >= height) or (blocks[x][y + 1][z] == AIR_ID)
                if bid != AIR_ID and top_air:
                    row.append((1, color_for(bid)))
                else:
                    row.append((0, (0.0, 0.0, 0.0)))
            mask.extend(row)
        # Merge rectangles in mask
        x = 0
        while x < width:
            z = 0
            while z < depth:
                idx = z * width + x
                if mask[idx][0] == 0:
                    z += 1
                    continue
                # Expand width
                w = 1
                while x + w < width and mask[z * width + (x + w)][0] == 1:
                    w += 1
                # Expand height
                h = 1
                done = False
                while not done and (z + h) < depth:
                    for k in range(w):
                        if mask[(z + h) * width + (x + k)][0] == 0:
                            done = True
                            break
                    if not done:
                        h += 1
                # Emit quad (two triangles) for rectangle [x..x+w), [z..z+h)
                col = mask[idx][1]
                _emit_top_quad(positions, colors, x, y, z, w, h, col)
                # Clear mask
                for dz in range(h):
                    for dx in range(w):
                        mask[(z + dz) * width + (x + dx)] = (0, (0.0, 0.0, 0.0))
                z += h
            x += 1

    # Fallback: emit side faces without merging to avoid missing geometry
    # +X, -X, +Z, -Z, bottom
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                bid = blocks[x][y][z]
                if bid == AIR_ID:
                    continue
                col = color_for(bid)
                # -Y bottom face
                if y - 1 < 0 or blocks[x][y - 1][z] == AIR_ID:
                    _emit_bottom_quad(positions, colors, x, y, z, col)
                # +X
                if x + 1 >= width or blocks[x + 1][y][z] == AIR_ID:
                    _emit_px_quad(positions, colors, x, y, z, col)
                # -X
                if x - 1 < 0 or blocks[x - 1][y][z] == AIR_ID:
                    _emit_nx_quad(positions, colors, x, y, z, col)
                # +Z
                if z + 1 >= depth or blocks[x][y][z + 1] == AIR_ID:
                    _emit_pz_quad(positions, colors, x, y, z, col)
                # -Z
                if z - 1 < 0 or blocks[x][y][z - 1] == AIR_ID:
                    _emit_nz_quad(positions, colors, x, y, z, col)

    return positions, colors


def _emit_top_quad(positions, colors, x, y, z, w, h, color):
    y1 = y + 1
    # two triangles covering rectangle in XZ
    positions.extend([
        x, y1, z + h,  x + w, y1, z + h,  x + w, y1, z,
        x, y1, z + h,  x + w, y1, z,      x, y1, z,
    ])
    for _ in range(6):
        colors.extend([color[0], color[1], color[2]])


def _emit_bottom_quad(positions, colors, x, y, z, color):
    y0 = y
    positions.extend([
        x, y0, z,  x + 1, y0, z,  x + 1, y0, z + 1,
        x, y0, z,  x + 1, y0, z + 1,  x, y0, z + 1,
    ])
    for _ in range(6):
        colors.extend([color[0], color[1], color[2]])


def _emit_px_quad(positions, colors, x, y, z, color):
    x1 = x + 1
    positions.extend([
        x1, y, z,  x1, y + 1, z,  x1, y + 1, z + 1,
        x1, y, z,  x1, y + 1, z + 1,  x1, y, z + 1,
    ])
    for _ in range(6):
        colors.extend([color[0], color[1], color[2]])


def _emit_nx_quad(positions, colors, x, y, z, color):
    x0 = x
    positions.extend([
        x0, y, z + 1,  x0, y + 1, z + 1,  x0, y + 1, z,
        x0, y, z + 1,  x0, y + 1, z,      x0, y, z,
    ])
    for _ in range(6):
        colors.extend([color[0], color[1], color[2]])


def _emit_pz_quad(positions, colors, x, y, z, color):
    z1 = z + 1
    positions.extend([
        x, y, z1,  x + 1, y, z1,  x + 1, y + 1, z1,
        x, y, z1,  x + 1, y + 1, z1,  x, y + 1, z1,
    ])
    for _ in range(6):
        colors.extend([color[0], color[1], color[2]])


def _emit_nz_quad(positions, colors, x, y, z, color):
    z0 = z
    positions.extend([
        x, y + 1, z0,  x + 1, y + 1, z0,  x + 1, y, z0,
        x, y + 1, z0,  x + 1, y, z0,      x, y, z0,
    ])
    for _ in range(6):
        colors.extend([color[0], color[1], color[2]])