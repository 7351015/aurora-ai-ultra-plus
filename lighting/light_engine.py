"""
INFINITUS Light Engine
Computes per-block skylight (0..15) for a chunk using a simple vertical propagation.
"""
from typing import List

AIR = 0
MAX_LIGHT = 15

class LightEngine:
    @staticmethod
    def compute_skylight(blocks: List[List[List[int]]], max_light: int = MAX_LIGHT) -> List[List[List[int]]]:
        """Compute a simple skylight map: start from top and attenuate down until blocked.
        Returns a 3D array (same shape as blocks) with light values 0..max_light.
        """
        if not blocks:
            return []
        width = len(blocks)
        height = len(blocks[0]) if width else 0
        depth = len(blocks[0][0]) if height else 0
        light = [[[0 for _ in range(depth)] for _ in range(height)] for _ in range(width)]
        for x in range(width):
            for z in range(depth):
                level = max_light
                for y in range(height - 1, -1, -1):
                    try:
                        if blocks[x][y][z] == AIR:
                            light[x][y][z] = level
                            # Attenuate slightly even through air to simulate haze
                            if level > 0:
                                level -= 1 if (y % 8 == 0) else 0
                        else:
                            # Blocked: strong attenuation
                            level = max(0, level - 3)
                            light[x][y][z] = level
                    except Exception:
                        light[x][y][z] = 0
        return light