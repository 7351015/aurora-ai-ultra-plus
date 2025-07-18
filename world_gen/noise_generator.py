"""
🌌 INFINITUS Noise Generator
Simple noise generation for terrain and world features.
"""

import asyncio
import math
import random
from typing import Dict, List, Any, Optional
import logging

class NoiseGenerator:
    """Simple noise generator for world generation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seed = 0
        self.scale = 1.0
        
        # Simple noise parameters
        self.octaves = 4
        self.persistence = 0.5
        self.lacunarity = 2.0
        
        self.logger.info("🌊 Noise Generator initialized")
    
    async def initialize(self):
        """Initialize the noise generator."""
        self.logger.info("🔧 Initializing Noise Generator...")
        self.logger.info("✅ Noise Generator initialization complete")
    
    async def configure(self, config: Dict[str, Any]):
        """Configure the noise generator."""
        self.seed = config.get('seed', 0)
        self.scale = config.get('scale', 1.0)
        
        # Set random seed
        random.seed(self.seed)
    
    def simple_noise(self, x: float, y: float) -> float:
        """Generate simple 2D noise."""
        # Simple hash-based noise
        n = int(x * 1000 + y * 10000)
        n = (n << 13) ^ n
        return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)
    
    def octave_noise(self, x: float, y: float) -> float:
        """Generate octave-based noise."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for i in range(self.octaves):
            value += self.simple_noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return value / max_value
    
    def height_noise(self, x: float, z: float) -> float:
        """Generate height map noise."""
        return self.octave_noise(x * self.scale, z * self.scale)
    
    def cave_noise(self, x: float, y: float, z: float) -> float:
        """Generate 3D cave noise."""
        return self.simple_noise(x * 0.1, y * 0.1) * self.simple_noise(z * 0.1, y * 0.1)
    
    def temperature_noise(self, x: float, z: float) -> float:
        """Generate temperature noise for biomes."""
        return self.octave_noise(x * 0.02, z * 0.02)
    
    def humidity_noise(self, x: float, z: float) -> float:
        """Generate humidity noise for biomes."""
        return self.octave_noise(x * 0.03, z * 0.03)
    
    async def shutdown(self):
        """Shutdown the noise generator."""
        self.logger.info("🔄 Shutting down Noise Generator...")
        self.logger.info("✅ Noise Generator shutdown complete")