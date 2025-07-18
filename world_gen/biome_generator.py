"""
🌌 INFINITUS Biome Generator
Generates diverse biomes for world generation.
"""

import asyncio
import random
from typing import Dict, List, Any, Optional
import logging

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
        # Simple biome generation - create a 16x16 array
        biomes = []
        for x in range(16):
            row = []
            for z in range(16):
                # Simple biome selection based on position
                world_x = chunk_x * 16 + x
                world_z = chunk_z * 16 + z
                
                # Use position to determine biome
                biome_id = self._get_biome_at(world_x, world_z)
                row.append(biome_id)
            biomes.append(row)
        
        return biomes
    
    def _get_biome_at(self, x: int, z: int) -> int:
        """Get biome ID at world coordinates."""
        # Simple biome distribution
        distance_from_origin = (x*x + z*z) ** 0.5
        
        if distance_from_origin < 100:
            return BiomeType.PLAINS
        elif distance_from_origin < 500:
            return BiomeType.FOREST
        elif distance_from_origin < 1000:
            return BiomeType.MOUNTAINS
        else:
            return BiomeType.DESERT
    
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
                'jungle': BiomeType.JUNGLE
            }
        }
    
    async def shutdown(self):
        """Shutdown the biome generator."""
        self.logger.info("🔄 Shutting down Biome Generator...")
        self.logger.info("✅ Biome Generator shutdown complete")