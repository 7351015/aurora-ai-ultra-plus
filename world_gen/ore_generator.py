"""
INFINITUS Ore Generator
Populates terrain stone with ore veins based on rarity and depth.
"""
from typing import Dict, Any, List
import random

# Block IDs matching terrain constants
AIR = 0
STONE = 1
COAL = 7
IRON = 8
GOLD = 9
DIAMOND = 10

class OreGenerator:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.seed: int = 0
        self.rng = random.Random()

    async def initialize(self):
        return

    async def configure(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.seed = int(self.config.get('seed', 0))
        self.rng.seed(self.seed)

    async def populate_chunk_ores(self, blocks: List[List[List[int]]]):
        width = len(blocks)
        height = len(blocks[0]) if width else 0
        depth = len(blocks[0][0]) if height else 0
        if width == 0 or height == 0 or depth == 0:
            return
        # Simple ore spawning rules
        for _ in range(64):
            x = self.rng.randrange(0, width)
            z = self.rng.randrange(0, depth)
            y = self.rng.randrange(4, min(64, height))
            ore = None
            r = self.rng.random()
            if y < 20 and r < 0.02:
                ore = DIAMOND
            elif y < 32 and r < 0.05:
                ore = GOLD
            elif y < 60 and r < 0.12:
                ore = IRON
            elif r < 0.2:
                ore = COAL
            if ore is None:
                continue
            # Place small vein
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if 0 <= xx < width and 0 <= yy < height and 0 <= zz < depth:
                            try:
                                if blocks[xx][yy][zz] == STONE:
                                    blocks[xx][yy][zz] = ore
                            except Exception:
                                pass