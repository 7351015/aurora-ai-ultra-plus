"""
🌌 INFINITUS Minecraft Gameplay Engine
Core gameplay loop for Minecraft-like mechanics: player, blocks, world session, input plumbing.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from core_engine.config import GameConfig
from world_gen.world_generator import WorldGenerator
from core_engine.physics_engine import PhysicsEngine, PhysicsObject
from core_engine.save_system import SaveSystem
from weather.weather_system import WeatherSystem
from gameplay.crafting_system import CraftingSystem
from gameplay.entity_system import EntitySystem
from gameplay.stats_system import StatsSystem, PlayerStats


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
    hotbar: List[str] = field(default_factory=lambda: ["stone", "dirt", "grass"])
    hotbar_index: int = 0

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

        # Physics
        self.physics: Optional[PhysicsEngine] = None
        self.player_body_id: Optional[str] = None

        # Persistence
        self.save_system: Optional[SaveSystem] = None
        self._autosave_elapsed: float = 0.0

        # Day-night
        self._time_of_day: float = 6000.0  # 0..24000
        
        # Weather
        self.weather: WeatherSystem = WeatherSystem()
        
        # Systems
        self.crafting = CraftingSystem()
        self.entities = EntitySystem()
        self.stats = StatsSystem()

        # Game loop state
        self._running: bool = False
        self._tick: int = 0
        self._tasks: List[asyncio.Task] = []
        self._last_update_time: Optional[float] = None

    async def initialize(self) -> None:
        self.logger.info("🔧 Initializing Minecraft Gameplay Engine...")
        # Prepare a local world generator for gameplay needs
        self.world_generator = WorldGenerator(self.config)
        await self.world_generator.initialize()
        # Physics engine for player controller
        self.physics = PhysicsEngine(self.config)
        await self.physics.initialize()
        # Save system
        self.save_system = SaveSystem()
        await self.save_system.initialize()
        # Weather
        await self.weather.initialize()
        # Entities
        await self.entities.initialize()
        # Stats
        await self.stats.initialize()
        self.logger.info("✅ Minecraft Gameplay Engine ready")

    async def create_player(self, world_data: Dict[str, Any], name: str = "Player") -> str:
        """Create the local player at the world's spawn point."""
        if not world_data or "metadata" not in world_data:
            raise ValueError("World data missing metadata for spawn")
        self.world_data = world_data
        # Initialize world metadata fields as needed
        self.world_data.setdefault("players", {})
        self.world_data.setdefault("metadata", {})
        self.world_data["metadata"].setdefault("time_of_day", self._time_of_day)
        spawn = self.world_data["metadata"].get("spawn_point", (0.0, 80.0, 0.0))
        player_id = "local"
        self.players[player_id] = Player(player_id=player_id, name=name, position=tuple(spawn))
        # Starter inventory for demonstration
        p = self.players[player_id]
        p.inventory.setdefault('wood_log', 3)
        p.inventory.setdefault('cobblestone', 3)
        p.inventory.setdefault('stick', 2)
        self.local_player_id = player_id
        # Create physics body
        if self.physics:
            body = PhysicsObject(
                id="player",
                position=(spawn[0], spawn[1], spawn[2]),
                size=(0.6, 1.8, 0.6),
                mass=70.0,
                friction=0.8,
                restitution=0.0,
                static=False,
                shape="box",
            )
            self.physics.add_object(body)
            self.player_body_id = body.id
        self.logger.info(f"👤 Created player '{name}' at {spawn}")
        return player_id

    async def start_game(self) -> None:
        """Start singleplayer session."""
        if self.world_data is None:
            # Generate a default world if none provided yet
            self.world_data = await self.world_generator.generate_world("InfinitusWorld")
        self.loaded = True
        self._running = True
        self._last_update_time = None
        # Load time of day if present
        try:
            self._time_of_day = float(self.world_data.get("metadata", {}).get("time_of_day", 6000.0))
        except Exception:
            self._time_of_day = 6000.0
        self.logger.info("🎮 Singleplayer session started")

    async def start_multiplayer_game(self) -> None:
        """Start multiplayer session (placeholder wiring)."""
        self.loaded = True
        self._running = True
        self._last_update_time = None
        self.logger.info("🌐 Multiplayer session started")

    async def update(self) -> None:
        if not self._running:
            return
        self._tick += 1
        # Delta time
        now = asyncio.get_event_loop().time()
        if self._last_update_time is None:
            delta_time = 1.0 / 60.0
        else:
            delta_time = max(0.0001, min(0.1, now - self._last_update_time))
        self._last_update_time = now
        # Physics update
        if self.physics:
            await self.physics.update(delta_time)
            # Sync player position
            if self.player_body_id and self.local_player_id in self.players:
                body = self.physics.get_object(self.player_body_id)
                if body:
                    self.players[self.local_player_id].position = tuple(body.position)
        # Stream chunks near player
        player = self.get_local_player()
        if player and self.world_generator:
            try:
                await self.world_generator.request_chunk_streaming(int(player.position[0]), int(player.position[2]))
                self.world_generator.set_player_position(self.local_player_id, int(player.position[0]), int(player.position[2]))
            except Exception:
                pass
        # Day-night cycle
        self._time_of_day = (self._time_of_day + delta_time * 120.0) % 24000.0
        if self.world_data is not None:
            self.world_data.setdefault("metadata", {})
            self.world_data["metadata"]["time_of_day"] = self._time_of_day
        # Weather update
        if self.weather:
            await self.weather.update(delta_time)
        # Entities update
        if self.entities:
            lp = (player.position[0], player.position[1], player.position[2]) if player else (0.0, 0.0, 0.0)
            await self.entities.update(delta_time, lp)
        # Autosave
        self._autosave_elapsed += delta_time
        if self.save_system and self.config.gameplay.auto_save and self._autosave_elapsed >= float(self.config.gameplay.auto_save_interval):
            await self._do_autosave()
            self._autosave_elapsed = 0.0
        # Stats update
        if player:
            pstats = getattr(player, 'stats', None)
            if pstats is None:
                player.stats = PlayerStats()
                pstats = player.stats
            await self.stats.update(delta_time, pstats)

    async def handle_input(self) -> None:
        """Process input events from the graphics/input layer if available."""
        if hasattr(self.graphics_engine, "get_events"):
            events = await self.graphics_engine.get_events()
            if not events:
                return
            player = self.get_local_player()
            if not player:
                return
            for ev in events:
                et = ev.get("type")
                if et == "move":
                    dx, dy, dz = ev.get("delta", (0.0, 0.0, 0.0))
                    # Map to physics velocity/impulse
                    if self.physics and self.player_body_id:
                        body = self.physics.get_object(self.player_body_id)
                        if body:
                            # Move relative to view yaw
                            yaw_rad = math.radians(player.rotation[0])
                            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
                            # Translate input vector into world space (ignoring pitch for movement)
                            vx = dx * cos_y + dz * sin_y
                            vz = dz * cos_y - dx * sin_y
                            body.velocity = (vx * 3.0, body.velocity[1] + (dy * 4.0), vz * 3.0)
                    else:
                        x, y, z = player.position
                        player.position = (x + dx, y + dy, z + dz)
                elif et == "look":
                    # Store simple orientation on player for camera to read
                    mx, my = ev.get("delta", (0.0, 0.0))
                    yaw, pitch = player.rotation
                    player.rotation = (yaw + mx * 0.1, max(-89.0, min(89.0, pitch - my * 0.1)))
                elif et == "action":
                    btn = ev.get("button")
                    if btn == "break":
                        hit = self._raycast_block(player.position, player.rotation)
                        if hit:
                            chunk_key, bx, by, bz = hit
                            self._set_demo_block(chunk_key, bx, by, bz, BLOCK_AIR)
                            if hasattr(self.graphics_engine, "refresh_world_mesh"):
                                self.graphics_engine.refresh_world_mesh()
                    elif btn == "place":
                        hit = self._raycast_block(player.position, player.rotation, place=True)
                        if hit:
                            chunk_key, bx, by, bz = hit
                            block_id = self._selected_block_id(player)
                            self._set_demo_block(chunk_key, bx, by, bz, block_id)
                            if hasattr(self.graphics_engine, "refresh_world_mesh"):
                                self.graphics_engine.refresh_world_mesh()
                elif et == "hotbar":
                    delta = ev.get("delta", 0)
                    player.hotbar_index = (player.hotbar_index + int(delta)) % max(1, len(player.hotbar))
                elif et == "hotbar_select":
                    idx = int(ev.get("index", 0))
                    if 0 <= idx < len(player.hotbar):
                        player.hotbar_index = idx

    async def shutdown(self) -> None:
        self._running = False
        # Cancel background tasks if any
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()
        if self.world_generator:
            await self.world_generator.shutdown()
        if self.physics:
            await self.physics.shutdown()
        if self.save_system:
            await self.save_system.shutdown()
        self.logger.info("✅ Minecraft Gameplay Engine shutdown complete")

    # ----- Convenience gameplay APIs -----
    def get_local_player(self) -> Optional[Player]:
        if self.local_player_id is None:
            return None
        return self.players.get(self.local_player_id)

    def get_block(self, chunk_x: int, x: int, y: int, chunk_z: int, z: int) -> int:
        """Get a block by chunk-local coords. Returns block ID."""
        if not self.world_data:
            return BLOCK_AIR
        chunks = self.world_data.get("spawn_chunks", {})
        if not chunks:
            return BLOCK_AIR
        key = f"{chunk_x},{chunk_z}"
        if key not in chunks:
            key = "0,0" if "0,0" in chunks else next(iter(chunks))
        first_chunk = chunks.get(key, {})
        try:
            return first_chunk.get("blocks", [])[x][y][z]
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

    def get_render_chunks(self) -> List[Tuple[int, int, List[List[List[int]]]]]:
        """Return a list of (chunk_x, chunk_z, blocks) for rendering."""
        chunks_out: List[Tuple[int, int, List[List[List[int]]]]] = []
        # Include spawn chunks from static world_data
        if self.world_data and "spawn_chunks" in self.world_data:
            for key, data in self.world_data["spawn_chunks"].items():
                try:
                    cx_str, cz_str = key.split(",")
                    cx, cz = int(cx_str), int(cz_str)
                    blocks = data.get("blocks", [])
                    if blocks:
                        chunks_out.append((cx, cz, blocks))
                except Exception:
                    continue
        # Include dynamically loaded chunks if any
        if self.world_generator:
            try:
                loaded = self.world_generator.get_loaded_chunks()
                for (cx, cz), chunk in loaded.items():
                    if chunk and chunk.blocks:
                        chunks_out.append((cx, cz, chunk.blocks))
            except Exception:
                pass
        return chunks_out

    def _selected_block_id(self, player: Player) -> int:
        name = player.hotbar[player.hotbar_index] if player.hotbar else "stone"
        return {"stone": 1, "dirt": 2, "grass": 3}.get(name, 1)

    def _set_demo_block(self, chunk_key: str, x: int, y: int, z: int, block_id: int) -> None:
        if not self.world_data:
            return
        chunks = self.world_data.get("spawn_chunks", {})
        chunk = chunks.get(chunk_key)
        if not chunk:
            return
        try:
            chunk["blocks"][x][y][z] = block_id
        except Exception:
            return

    def _raycast_block(self, position: Tuple[float, float, float], rotation: Tuple[float, float], place: bool = False) -> Optional[Tuple[str, int, int, int]]:
        """Simple grid raycast within the origin chunk to find target block.
        Returns (chunk_key, bx, by, bz) or None. If place=True, returns position of adjacent block.
        """
        if not self.world_data:
            return None
        chunks = self.world_data.get("spawn_chunks", {})
        if "0,0" not in chunks:
            return None
        chunk_key = "0,0"
        blocks = chunks[chunk_key].get("blocks", [])
        if not blocks:
            return None
        # Ray from eye position forward
        yaw, pitch = math.radians(rotation[0]), math.radians(rotation[1])
        dirx = math.cos(pitch) * math.sin(yaw)
        diry = math.sin(pitch)
        dirz = math.cos(pitch) * math.cos(yaw)
        ox, oy, oz = position
        max_dist = 6.0
        step = 0.1
        prev = (int(ox), int(oy), int(oz))
        t = 0.0
        while t <= max_dist:
            wx = ox + dirx * t
            wy = oy + diry * t
            wz = oz + dirz * t
            bx = int(wx) % 16
            by = max(0, min(len(blocks[0]) - 1, int(wy)))
            bz = int(wz) % 16
            try:
                if place:
                    # If current block is solid, place at previous empty
                    if blocks[bx][by][bz] != BLOCK_AIR:
                        px, py, pz = prev
                        pbx = px % 16
                        pby = max(0, min(len(blocks[0]) - 1, py))
                        pbz = pz % 16
                        return (chunk_key, pbx, pby, pbz)
                else:
                    if blocks[bx][by][bz] != BLOCK_AIR:
                        return (chunk_key, bx, by, bz)
            except Exception:
                pass
            prev = (int(wx), int(wy), int(wz))
            t += step
        return None

    async def _do_autosave(self) -> None:
        """Save world and player data using SaveSystem."""
        if not self.save_system or not self.world_data:
            return
        # Attach player data
        try:
            pdata = {}
            for pid, p in self.players.items():
                pdata[pid] = {
                    "name": p.name,
                    "position": p.position,
                    "rotation": p.rotation,
                    "inventory": p.inventory,
                    "hotbar": p.hotbar,
                    "hotbar_index": p.hotbar_index,
                    "stats": getattr(p, 'stats', None).__dict__ if getattr(p, 'stats', None) else None,
                }
            self.world_data["players"] = pdata
        except Exception:
            pass
        # Use world name from metadata
        world_name = str(self.world_data.get("metadata", {}).get("name", "World"))
        try:
            await self.save_system.save_world(world_name, self.world_data)
            self.logger.info("💾 Autosaved world")
        except Exception as e:
            self.logger.warning(f"Autosave failed: {e}")