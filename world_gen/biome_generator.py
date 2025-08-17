"""
🌌 INFINITUS Biome Generator
Generates diverse biomes for world generation.
"""

import asyncio
import random
from typing import Dict, List, Any, Optional
import logging
import math

class BiomeType:
    """Biome type constants."""
    PLAINS = 1
    FOREST = 2
    DESERT = 3
    MOUNTAINS = 4
    OCEAN = 5
    SWAMP = 6
    TUNDRA = 7
    JUNGLE = 8
    SAVANNA = 9
    TAIGA = 10
    SNOWY_MOUNTAINS = 11

class BiomeGenerator:
    """Biome generation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seed = 0
        self.diversity = 1.0
        self.world_type = None
        
        self.logger.info("🌿 Biome Generator initialized")
    
    async def initialize(self):
        """Initialize the biome generator."""
        self.logger.info("🔧 Initializing Biome Generator...")
        self.logger.info("✅ Biome Generator initialization complete")
    
    async def configure(self, config: Dict[str, Any]):
        """Configure the biome generator."""
        self.seed = config.get('seed', 0)
        self.diversity = config.get('diversity', 1.0)
        self.world_type = config.get('world_type')
        
        random.seed(self.seed)
    
    async def generate_chunk_biomes(self, chunk_x: int, chunk_z: int):
        """Generate biomes for a chunk."""
        # 16x16 biome IDs based on temperature/humidity style noise
        biomes = []
        for x in range(16):
            row = []
            for z in range(16):
                world_x = chunk_x * 16 + x
                world_z = chunk_z * 16 + z
                biome_id = self._pick_biome(world_x, world_z)
                row.append(biome_id)
            biomes.append(row)
        return biomes
    
    def _noise(self, x: float, z: float, scale: float = 0.01, offset: int = 0) -> float:
        n = int((x + offset) * 1619 + (z - offset) * 31337 + self.seed * 1013)
        n = (n << 13) ^ n
        return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0
    
    def _pick_biome(self, x: int, z: int) -> int:
        t = (self._noise(x, z, 0.002, 17) * 0.5 + 0.5)  # temperature 0..1
        h = (self._noise(x, z, 0.002, 73) * 0.5 + 0.5)  # humidity 0..1
        e = (self._noise(x, z, 0.004, 29) * 0.5 + 0.5)  # elevation factor
        # Ocean threshold
        if e < 0.35:
            return BiomeType.OCEAN
        # Mountains
        if e > 0.75:
            return BiomeType.SNOWY_MOUNTAINS if t < 0.4 else BiomeType.MOUNTAINS
        # Hot/dry
        if t > 0.7 and h < 0.35:
            return BiomeType.DESERT if e < 0.6 else BiomeType.SAVANNA
        # Cold
        if t < 0.3:
            return BiomeType.TUNDRA if h < 0.5 else BiomeType.TAIGA
        # Wet and warm
        if h > 0.65 and t > 0.5:
            return BiomeType.JUNGLE
        # Wet and low
        if h > 0.65 and e < 0.55:
            return BiomeType.SWAMP
        # Default
        return BiomeType.FOREST if h > 0.5 else BiomeType.PLAINS
    
    async def get_world_data(self) -> Dict[str, Any]:
        """Get biome world data."""
        return {
            'seed': self.seed,
            'diversity': self.diversity,
            'biome_types': {
                'plains': BiomeType.PLAINS,
                'forest': BiomeType.FOREST,
                'desert': BiomeType.DESERT,
                'mountains': BiomeType.MOUNTAINS,
                'ocean': BiomeType.OCEAN,
                'swamp': BiomeType.SWAMP,
                'tundra': BiomeType.TUNDRA,
                'jungle': BiomeType.JUNGLE,
                'savanna': BiomeType.SAVANNA,
                'taiga': BiomeType.TAIGA,
                'snowy_mountains': BiomeType.SNOWY_MOUNTAINS,
            }
        }
    
    async def shutdown(self):
        """Shutdown the biome generator."""
        self.logger.info("🔄 Shutting down Biome Generator...")
        self.logger.info("✅ Biome Generator shutdown complete")