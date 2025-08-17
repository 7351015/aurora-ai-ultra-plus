"""
🌌 INFINITUS Minecraft Gameplay Engine
Core gameplay loop for Minecraft-like mechanics: player, blocks, world session, input plumbing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from core_engine.config import GameConfig
from world_gen.world_generator import WorldGenerator


# Minimal block ID mapping used by terrain generator
BLOCK_AIR = 0
BLOCK_STONE = 1


@dataclass
class Player:
    player_id: str
    name: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float] = (0.0, 0.0)
    inventory: Dict[str, int] = field(default_factory=dict)

    def add_item(self, item_id: str, count: int = 1) -> None:
        self.inventory[item_id] = self.inventory.get(item_id, 0) + count

    def remove_item(self, item_id: str, count: int = 1) -> bool:
        if self.inventory.get(item_id, 0) < count:
            return False
        self.inventory[item_id] -= count
        if self.inventory[item_id] <= 0:
            del self.inventory[item_id]
        return True


class MinecraftEngine:
    """High-level gameplay engine used by main loop.

    Provides minimal but extendable gameplay primitives so the game can start,
    create a player, and tick/update without errors. Rendering and input are
    delegated to the GraphicsEngine and UI systems.
    """

    def __init__(self, config: GameConfig, logger, graphics_engine):
        self.config = config
        # Use stdlib logger to avoid coupling to custom logger type
        self.logger = logging.getLogger(__name__)
        self.graphics_engine = graphics_engine

        # World/session
        self.world_generator: Optional[WorldGenerator] = None
        self.world_data: Optional[Dict[str, Any]] = None
        self.loaded: bool = False

        # Players
        self.players: Dict[str, Player] = {}
        self.local_player_id: Optional[str] = None

        # Game loop state
        self._running: bool = False
        self._tick: int = 0
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        self.logger.info("🔧 Initializing Minecraft Gameplay Engine...")
        # Prepare a local world generator for gameplay needs
        self.world_generator = WorldGenerator(self.config)
        await self.world_generator.initialize()
        self.logger.info("✅ Minecraft Gameplay Engine ready")

    async def create_player(self, world_data: Dict[str, Any], name: str = "Player") -> str:
        """Create the local player at the world's spawn point."""
        if not world_data or "metadata" not in world_data:
            raise ValueError("World data missing metadata for spawn")
        spawn = world_data["metadata"].get("spawn_point", (0.0, 80.0, 0.0))
        player_id = "local"
        self.players[player_id] = Player(player_id=player_id, name=name, position=tuple(spawn))
        self.local_player_id = player_id
        self.logger.info(f"👤 Created player '{name}' at {spawn}")
        return player_id

    async def start_game(self) -> None:
        """Start singleplayer session."""
        if self.world_data is None:
            # Generate a default world if none provided yet
            self.world_data = await self.world_generator.generate_world("InfinitusWorld")
        self.loaded = True
        self._running = True
        self.logger.info("🎮 Singleplayer session started")

    async def start_multiplayer_game(self) -> None:
        """Start multiplayer session (placeholder wiring)."""
        self.loaded = True
        self._running = True
        self.logger.info("🌐 Multiplayer session started")

    async def update(self) -> None:
        if not self._running:
            return
        self._tick += 1
        # Placeholder for per-tick gameplay logic
        # Example: simple gravity if above ground
        player = self.get_local_player()
        if player:
            x, y, z = player.position
            # Clamp Y to be non-negative in this placeholder loop
            if y > 0:
                player.position = (x, y - 0.01, z)

    async def handle_input(self) -> None:
        """Process input events from the graphics/input layer if available."""
        if hasattr(self.graphics_engine, "get_events"):
            events = await self.graphics_engine.get_events()
            # Input processing placeholder; integrate with actual input system later
            if not events:
                return
            player = self.get_local_player()
            if not player:
                return
            for ev in events:
                # Expected format depends on GraphicsEngine; keep generic
                if ev.get("type") == "move":
                    dx, dy, dz = ev.get("delta", (0.0, 0.0, 0.0))
                    x, y, z = player.position
                    player.position = (x + dx, y + dy, z + dz)

    async def shutdown(self) -> None:
        self._running = False
        # Cancel background tasks if any
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()
        if self.world_generator:
            await self.world_generator.shutdown()
        self.logger.info("✅ Minecraft Gameplay Engine shutdown complete")

    # ----- Convenience gameplay APIs -----
    def get_local_player(self) -> Optional[Player]:
        if self.local_player_id is None:
            return None
        return self.players.get(self.local_player_id)

    def get_block(self, chunk_x: int, x: int, y: int, chunk_z: int, z: int) -> int:
        """Get a block by chunk-local coords. Returns block ID."""
        # This placeholder accesses spawn chunks if available
        if not self.world_data:
            return BLOCK_AIR
        chunks = self.world_data.get("spawn_chunks", [])
        if not chunks:
            return BLOCK_AIR
        # Use first chunk as a demo space
        first_chunk = chunks[0]
        try:
            return first_chunk["blocks"][x][y][z]
        except Exception:
            return BLOCK_AIR

    def set_block(self, chunk_x: int, x: int, y: int, chunk_z: int, z: int, block_id: int) -> bool:
        """Set a block by chunk-local coords. Returns success."""
        if not self.world_data:
            return False
        chunks = self.world_data.get("spawn_chunks", [])
        if not chunks:
            return False
        first_chunk = chunks[0]
        try:
            first_chunk["blocks"][x][y][z] = block_id
            return True
        except Exception:
            return False