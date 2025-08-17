"""
🌌 INFINITUS Terrain Generator
Generates terrain for world chunks.
"""

import asyncio
import random
import math
from typing import Dict, List, Any, Optional
import logging

# Block IDs
AIR = 0
STONE = 1
DIRT = 2
GRASS = 3
WATER = 4

class TerrainGenerator:
    """Terrain generation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seed = 0
        self.logger.info("🏔️ Terrain Generator initialized")
    
    async def initialize(self):
        """Initialize the terrain generator."""
        self.logger.info("🔧 Initializing Terrain Generator...")
        self.logger.info("✅ Terrain Generator initialization complete")
    
    async def configure(self, config: Dict[str, Any]):
        """Configure the terrain generator."""
        self.seed = config.get('seed', 0)
        random.seed(self.seed)
    
    async def generate_chunk_terrain(self, chunk_x: int, chunk_z: int, biomes):
        """Generate terrain for a chunk.
        Produces hills using simple trigonometric functions; layers grass/dirt/stone.
        """
        width, height, depth = 16, 128, 16
        terrain: List[List[List[int]]] = [[[AIR for _ in range(depth)] for _ in range(height)] for _ in range(width)]
        base_height = 62
        for x in range(width):
            for z in range(depth):
                wx = chunk_x * 16 + x
                wz = chunk_z * 16 + z
                # Simple height function (deterministic per seed)
                h = base_height
                h += int(6.0 * math.sin((wx + self.seed % 97) * 0.07) + 6.0 * math.cos((wz - self.seed % 131) * 0.05))
                h = max(48, min(height - 2, h))
                # Water level fill
                water_level = 60
                top = h
                # Stone below
                for y in range(0, max(0, top - 4)):
                    terrain[x][y][z] = STONE
                # Dirt layer
                for y in range(max(0, top - 4), max(0, top)):
                    terrain[x][y][z] = DIRT
                # Grass top
                terrain[x][top][z] = GRASS
                # Water fill where below water_level
                if top < water_level:
                    for y in range(top + 1, water_level + 1):
                        if y < height:
                            terrain[x][y][z] = WATER
        return terrain
    
    async def shutdown(self):
        """Shutdown the terrain generator."""
        self.logger.info("🔄 Shutting down Terrain Generator...")
        self.logger.info("✅ Terrain Generator shutdown complete")