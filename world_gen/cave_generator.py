"""
INFINITUS Cave Generator
Carves simple caves/tunnels using random walk.
"""
from typing import List
import random

AIR = 0

class CaveGenerator:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)

    async def initialize(self):
        return

    async def configure(self, config):
        self.seed = int(config.get('seed', 0))
        self.rng.seed(self.seed)

    async def carve_caves(self, blocks: List[List[List[int]]]):
        width = len(blocks)
        height = len(blocks[0]) if width else 0
        depth = len(blocks[0][0]) if height else 0
        if width == 0 or height == 0 or depth == 0:
            return
        # Random walk tunnels starting points
        for _ in range(3):
            x = self.rng.randrange(2, width - 2)
            y = self.rng.randrange(20, min(70, height - 5))
            z = self.rng.randrange(2, depth - 2)
            dx = self.rng.choice([-1, 1])
            dz = self.rng.choice([-1, 1])
            steps = self.rng.randrange(30, 90)
            for _s in range(steps):
                # Carve a small sphere
                for rx in range(-1, 2):
                    for ry in range(-1, 2):
                        for rz in range(-1, 2):
                            xx, yy, zz = x + rx, y + ry, z + rz
                            if 0 <= xx < width and 0 <= yy < height and 0 <= zz < depth:
                                try:
                                    blocks[xx][yy][zz] = AIR
                                except Exception:
                                    pass
                # Move
                x += dx
                z += dz
                # Random turn
                if self.rng.random() < 0.2:
                    dx = self.rng.choice([-1, 0, 1])
                    dz = self.rng.choice([-1, 0, 1])
                # Clamp
                x = max(2, min(width - 3, x))
                z = max(2, min(depth - 3, z))