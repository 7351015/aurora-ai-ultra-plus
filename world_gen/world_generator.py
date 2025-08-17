"""
🌌 INFINITUS World Generator
Main world generation system that creates infinite, diverse worlds.
"""

import asyncio
import random
import time
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from core_engine.config import GameConfig
from .biome_generator import BiomeGenerator
from .terrain_generator import TerrainGenerator
from .structure_generator import StructureGenerator
from .civilization_generator import CivilizationGenerator
from .portal_generator import PortalGenerator
from .noise_generator import NoiseGenerator

class WorldType(Enum):
    """Types of worlds that can be generated."""
    NORMAL = "normal"
    SUPERFLAT = "superflat"
    AMPLIFIED = "amplified"
    FLOATING_ISLANDS = "floating_islands"
    UNDERGROUND = "underground"
    SPACE = "space"
    DIMENSION_NEXUS = "dimension_nexus"
    CUSTOM = "custom"

@dataclass
class WorldSettings:
    """World generation settings."""
    world_type: WorldType = WorldType.NORMAL
    seed: Optional[int] = None
    size: str = "infinite"  # small, medium, large, infinite
    biome_diversity: float = 1.0
    terrain_roughness: float = 1.0
    structure_density: float = 1.0
    civilization_frequency: float = 1.0
    portal_frequency: float = 0.1
    enable_caves: bool = True
    enable_dungeons: bool = True
    enable_villages: bool = True
    enable_cities: bool = True
    enable_sky_islands: bool = True
    enable_underground_cities: bool = True
    enable_space_zones: bool = True
    enable_interdimensional_portals: bool = True
    water_level: int = 64
    max_height: int = 320
    min_height: int = -64

@dataclass
class Chunk:
    """World chunk data."""
    x: int
    z: int
    blocks: List[List[List[int]]] = None
    biomes: List[List[int]] = None
    structures: List[Dict] = field(default_factory=list)
    entities: List[Dict] = field(default_factory=list)
    generated: bool = False
    populated: bool = False
    
    def __post_init__(self):
        if self.blocks is None:
            # Initialize with air blocks (16x384x16)
            self.blocks = [[[0 for _ in range(16)] for _ in range(384)] for _ in range(16)]
        if self.biomes is None:
            # Initialize biomes (16x16)
            self.biomes = [[0 for _ in range(16)] for _ in range(16)]

@dataclass
class CreatedWorld:
    """Wrapper for created world information."""
    name: str
    data: Dict[str, Any]

class WorldGenerator:
    """Main world generation system."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # World settings
        self.world_settings = WorldSettings()
        self.current_world: Optional[str] = None
        self.world_seed: Optional[int] = None
        
        # Generators
        self.noise_generator = NoiseGenerator()
        self.biome_generator = BiomeGenerator()
        self.terrain_generator = TerrainGenerator()
        self.structure_generator = StructureGenerator()
        self.civilization_generator = CivilizationGenerator()
        self.portal_generator = PortalGenerator()
        
        # World data
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.loaded_chunks: Dict[Tuple[int, int], Chunk] = {}
        self.world_metadata: Dict[str, Any] = {}
        
        # Generation queue
        self.generation_queue = asyncio.Queue()
        self.generation_workers = []
        self.generation_running = False
        
        # Streaming
        self.streaming_enabled = True
        self.view_distance = 16
        self.player_positions: Dict[str, Tuple[int, int]] = {}
        
        # Performance
        self.chunks_per_tick = 4
        self.generation_time = 0.0
        self.chunks_generated = 0
        
        self.logger.info("🌍 World Generator initialized")
    
    async def initialize(self):
        """Initialize the world generator."""
        self.logger.info("🔧 Initializing World Generator...")
        
        # Initialize sub-generators
        await self.noise_generator.initialize()
        await self.biome_generator.initialize()
        await self.terrain_generator.initialize()
        await self.structure_generator.initialize()
        await self.civilization_generator.initialize()
        await self.portal_generator.initialize()
        
        # Start generation workers
        self.generation_running = True
        for i in range(4):  # 4 worker threads
            worker = asyncio.create_task(self._generation_worker(i))
            self.generation_workers.append(worker)
        
        self.logger.info("✅ World Generator initialization complete")
    
    async def generate_world(self, world_name: str, world_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a new world."""
        try:
            start_time = time.time()
            self.logger.info(f"🌍 Generating world: {world_name}")
            
            # Apply world configuration
            if world_config:
                self._apply_world_config(world_config)
            
            # Set world seed
            if self.world_settings.seed is None:
                self.world_settings.seed = random.randint(0, 2**31 - 1)
            
            self.world_seed = self.world_settings.seed
            self.current_world = world_name
            
            # Initialize random generators with seed
            random.seed(self.world_seed)
            
            # Configure sub-generators
            await self._configure_generators()
            
            # Generate spawn area (initial chunks around 0,0)
            spawn_chunks = await self._generate_spawn_area()
            
            # Generate world metadata
            self.world_metadata = {
                'name': world_name,
                'seed': self.world_seed,
                'settings': self.world_settings,
                'spawn_point': self._find_spawn_point(),
                'generation_time': time.time() - start_time,
                'version': '1.0.0-alpha'
            }
            
            # Package world data
            world_data = {
                'metadata': self.world_metadata,
                'spawn_chunks': spawn_chunks,
                'biome_data': await self.biome_generator.get_world_data(),
                'structure_data': await self.structure_generator.get_world_data(),
                'civilization_data': await self.civilization_generator.get_world_data(),
                'portal_data': await self.portal_generator.get_world_data()
            }
            
            generation_time = time.time() - start_time
            self.logger.info(f"✅ World '{world_name}' generated in {generation_time:.2f}s")
            
            return world_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate world {world_name}: {e}")
            raise
    
    async def create_custom_world(self, world_params: Dict[str, Any]) -> CreatedWorld:
        """Create a custom world using provided parameters and return a wrapper with name and data."""
        config = dict(world_params or {})
        world_name = config.pop("name", "CustomWorld")
        self._apply_world_config(config)
        data = await self.generate_world(world_name, config)
        return CreatedWorld(name=world_name, data=data)
    
    def _apply_world_config(self, config: Dict):
        """Apply world configuration settings."""
        if 'world_type' in config:
            self.world_settings.world_type = WorldType(config['world_type'])
        if 'seed' in config:
            self.world_settings.seed = config['seed']
        if 'biome_diversity' in config:
            self.world_settings.biome_diversity = config['biome_diversity']
        if 'terrain_roughness' in config:
            self.world_settings.terrain_roughness = config['terrain_roughness']
        
        # Apply other settings
        for key, value in config.items():
            if hasattr(self.world_settings, key):
                setattr(self.world_settings, key, value)
    
    async def _configure_generators(self):
        """Configure all sub-generators with world settings."""
        # Configure noise generator
        await self.noise_generator.configure({
            'seed': self.world_seed,
            'scale': self.world_settings.terrain_roughness
        })
        
        # Configure biome generator
        await self.biome_generator.configure({
            'seed': self.world_seed,
            'diversity': self.world_settings.biome_diversity,
            'world_type': self.world_settings.world_type
        })
        
        # Configure terrain generator
        await self.terrain_generator.configure({
            'seed': self.world_seed,
            'world_type': self.world_settings.world_type,
            'roughness': self.world_settings.terrain_roughness,
            'water_level': self.world_settings.water_level,
            'max_height': self.world_settings.max_height,
            'min_height': self.world_settings.min_height
        })
        
        # Configure structure generator
        await self.structure_generator.configure({
            'seed': self.world_seed,
            'density': self.world_settings.structure_density,
            'enable_caves': self.world_settings.enable_caves,
            'enable_dungeons': self.world_settings.enable_dungeons
        })
        
        # Configure civilization generator
        await self.civilization_generator.configure({
            'seed': self.world_seed,
            'frequency': self.world_settings.civilization_frequency,
            'enable_villages': self.world_settings.enable_villages,
            'enable_cities': self.world_settings.enable_cities,
            'enable_underground_cities': self.world_settings.enable_underground_cities
        })
        
        # Configure portal generator
        await self.portal_generator.configure({
            'seed': self.world_seed,
            'frequency': self.world_settings.portal_frequency,
            'enable_portals': self.world_settings.enable_interdimensional_portals
        })
    
    async def _generate_spawn_area(self) -> Dict[str, Any]:
        """Generate the initial spawn area."""
        spawn_chunks = {}
        
        # Generate chunks in a 5x5 area around spawn (0,0)
        for x in range(-2, 3):
            for z in range(-2, 3):
                chunk = await self._generate_chunk(x, z)
                spawn_chunks[f"{x},{z}"] = self._serialize_chunk(chunk)
        
        return spawn_chunks
    
    async def _generate_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """Generate a single chunk."""
        start_time = time.time()
        
        # Create chunk
        chunk = Chunk(x=chunk_x, z=chunk_z)
        
        # Generate biomes for chunk
        chunk.biomes = await self.biome_generator.generate_chunk_biomes(chunk_x, chunk_z)
        
        # Generate terrain
        chunk.blocks = await self.terrain_generator.generate_chunk_terrain(
            chunk_x, chunk_z, chunk.biomes
        )
        
        # Generate structures
        if self.world_settings.enable_caves or self.world_settings.enable_dungeons:
            structures = await self.structure_generator.generate_chunk_structures(
                chunk_x, chunk_z, chunk.blocks, chunk.biomes
            )
            chunk.structures.extend(structures)
        
        # Generate civilizations
        if (self.world_settings.enable_villages or 
            self.world_settings.enable_cities or 
            self.world_settings.enable_underground_cities):
            civilizations = await self.civilization_generator.generate_chunk_civilizations(
                chunk_x, chunk_z, chunk.blocks, chunk.biomes
            )
            chunk.structures.extend(civilizations)
        
        # Generate portals
        if self.world_settings.enable_interdimensional_portals:
            portals = await self.portal_generator.generate_chunk_portals(
                chunk_x, chunk_z, chunk.blocks, chunk.biomes
            )
            chunk.structures.extend(portals)
        
        # Mark as generated
        chunk.generated = True
        chunk.populated = True
        
        # Store chunk
        self.chunks[(chunk_x, chunk_z)] = chunk
        
        # Update statistics
        self.chunks_generated += 1
        self.generation_time += time.time() - start_time
        
        return chunk
    
    def _serialize_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """Serialize chunk data for storage."""
        return {
            'x': chunk.x,
            'z': chunk.z,
            'blocks': chunk.blocks,
            'biomes': chunk.biomes,
            'structures': chunk.structures,
            'entities': chunk.entities,
            'generated': chunk.generated,
            'populated': chunk.populated
        }
    
    def _deserialize_chunk(self, data: Dict[str, Any]) -> Chunk:
        """Deserialize chunk data from storage."""
        chunk = Chunk(
            x=data['x'],
            z=data['z'],
            generated=data.get('generated', False),
            populated=data.get('populated', False)
        )
        
        chunk.blocks = data.get('blocks', [])
        chunk.biomes = data.get('biomes', [])
        chunk.structures = data.get('structures', [])
        chunk.entities = data.get('entities', [])
        
        return chunk
    
    def _find_spawn_point(self) -> Tuple[int, int, int]:
        """Find a suitable spawn point."""
        # For now, just return a point above the terrain at 0,0
        spawn_chunk = self.chunks.get((0, 0))
        if spawn_chunk and spawn_chunk.blocks:
            # Find highest solid block at center of chunk
            center_x, center_z = 8, 8
            for y in range(len(spawn_chunk.blocks[center_x]) - 1, -1, -1):
                if spawn_chunk.blocks[center_x][y][center_z] != 0:  # Not air
                    return (center_x, y + 2, center_z)  # Spawn 2 blocks above
        
        return (0, 100, 0)  # Default spawn point
    
    async def _generation_worker(self, worker_id: int):
        """Background chunk generation worker."""
        while self.generation_running:
            try:
                # Get chunk generation request
                request = await asyncio.wait_for(
                    self.generation_queue.get(), 
                    timeout=1.0
                )
                
                chunk_x, chunk_z = request['chunk_pos']
                
                # Check if chunk already exists
                if (chunk_x, chunk_z) not in self.chunks:
                    # Generate chunk
                    chunk = await self._generate_chunk(chunk_x, chunk_z)
                    
                    # Add to loaded chunks if near players
                    if self._is_chunk_near_players(chunk_x, chunk_z):
                        self.loaded_chunks[(chunk_x, chunk_z)] = chunk
                
                # Mark request as done
                self.generation_queue.task_done()
                
            except asyncio.TimeoutError:
                # No requests, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in generation worker {worker_id}: {e}")
                await asyncio.sleep(1.0)
    
    def _is_chunk_near_players(self, chunk_x: int, chunk_z: int) -> bool:
        """Check if chunk is near any player."""
        for player_id, (px, pz) in self.player_positions.items():
            distance = math.sqrt((chunk_x - px)**2 + (chunk_z - pz)**2)
            if distance <= self.view_distance:
                return True
        return False
    
    async def update(self, delta_time: float):
        """Update world generator (streaming, etc.)."""
        if not self.streaming_enabled:
            return
        
        # Process chunk generation requests
        processed = 0
        while processed < self.chunks_per_tick:
            try:
                # Check if we need to generate chunks near players
                for player_id, (px, pz) in self.player_positions.items():
                    await self._ensure_chunks_loaded(px, pz)
                
                processed += 1
                break  # For now, just break after one iteration
                
            except Exception as e:
                self.logger.error(f"Error in world generator update: {e}")
                break
    
    async def _ensure_chunks_loaded(self, player_x: int, player_z: int):
        """Ensure chunks are loaded around a player position."""
        chunk_x = player_x // 16
        chunk_z = player_z // 16
        
        # Check chunks in view distance
        for dx in range(-self.view_distance, self.view_distance + 1):
            for dz in range(-self.view_distance, self.view_distance + 1):
                cx, cz = chunk_x + dx, chunk_z + dz
                
                # Skip if chunk already loaded
                if (cx, cz) in self.loaded_chunks:
                    continue
                
                # Check if chunk exists
                if (cx, cz) not in self.chunks:
                    # Queue for generation
                    await self.generation_queue.put({
                        'chunk_pos': (cx, cz),
                        'priority': dx*dx + dz*dz  # Distance-based priority
                    })
    
    def set_player_position(self, player_id: str, x: int, z: int):
        """Update player position for chunk streaming."""
        self.player_positions[player_id] = (x // 16, z // 16)
    
    def remove_player(self, player_id: str):
        """Remove player from tracking."""
        self.player_positions.pop(player_id, None)
    
    def get_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """Get a chunk by coordinates."""
        return self.chunks.get((chunk_x, chunk_z))
    
    def get_loaded_chunks(self) -> Dict[Tuple[int, int], Chunk]:
        """Get all currently loaded chunks."""
        return self.loaded_chunks.copy()
    
    def unload_chunk(self, chunk_x: int, chunk_z: int):
        """Unload a chunk from memory."""
        if (chunk_x, chunk_z) in self.loaded_chunks:
            del self.loaded_chunks[(chunk_x, chunk_z)]
    
    def get_world_data(self) -> Dict[str, Any]:
        """Get current world data."""
        return {
            'metadata': self.world_metadata,
            'settings': self.world_settings,
            'chunks': {f"{x},{z}": self._serialize_chunk(chunk) 
                      for (x, z), chunk in self.chunks.items()},
            'statistics': {
                'chunks_generated': self.chunks_generated,
                'chunks_loaded': len(self.loaded_chunks),
                'generation_time': self.generation_time,
                'average_generation_time': self.generation_time / self.chunks_generated if self.chunks_generated > 0 else 0
            }
        }
    
    async def shutdown(self):
        """Shutdown the world generator."""
        self.logger.info("🔄 Shutting down World Generator...")
        
        # Stop generation workers
        self.generation_running = False
        
        # Wait for workers to finish
        for worker in self.generation_workers:
            worker.cancel()
        
        # Shutdown sub-generators
        await self.noise_generator.shutdown()
        await self.biome_generator.shutdown()
        await self.terrain_generator.shutdown()
        await self.structure_generator.shutdown()
        await self.civilization_generator.shutdown()
        await self.portal_generator.shutdown()
        
        # Clear data
        self.chunks.clear()
        self.loaded_chunks.clear()
        self.player_positions.clear()
        
        self.logger.info("✅ World Generator shutdown complete")