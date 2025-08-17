"""
🌌 INFINITUS Graphics Engine
Advanced graphics and rendering system.
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

from .mesh_builder import build_chunk_mesh
from .greedy_mesher import build_chunk_mesh_greedy
from .voxel_renderer import VoxelRenderer
from .camera import FirstPersonCamera

class GraphicsEngine:
    """Graphics and rendering system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎨 Graphics Engine initialized")
        self._pygame = None
        self._screen = None
        self._clock = None
        self._headless = False
        self._gameplay_engine = None
        self._tile_size = 24
        # 3D pipeline
        self._voxel: Optional[VoxelRenderer] = None
        self._positions: Optional[List[float]] = None
        self._colors: Optional[List[float]] = None
        self._camera: Optional[FirstPersonCamera] = None
    
    def set_gameplay_engine(self, engine: Any) -> None:
        self._gameplay_engine = engine
    
    async def initialize(self):
        self.logger.info("🔧 Initializing Graphics Engine...")
        # Initialize 3D renderer (safe to fail)
        self._voxel = VoxelRenderer(self.config)
        self._voxel.initialize()
        self._camera = FirstPersonCamera(self.config.graphics.fov, 0.1, 2000.0)
        # Lazy-import pygame to avoid dependency at import time
        try:
            import pygame
            self._pygame = pygame
            # Headless fallback
            if not os.environ.get("DISPLAY"):
                os.environ["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            flags = 0
            width, height = self.config.graphics.resolution
            self._screen = pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption("INFINITUS - Next Gen Sandbox")
            self._clock = pygame.time.Clock()
            self._headless = False
            self.logger.info("✅ Graphics Engine initialization complete")
        except Exception as e:
            # Headless mode: no rendering, but still provide event interface
            self._pygame = None
            self._screen = None
            self._clock = None
            self._headless = True
            self.logger.warning(f"⚠️ Graphics initialization failed or headless environment detected: {e}")
            self.logger.info("✅ Headless Graphics mode enabled")
        # Prebuild mesh if world already exists
        self._try_build_spawn_mesh()
    
    def _try_build_spawn_mesh(self) -> None:
        if not self._gameplay_engine:
            return
        wd = getattr(self._gameplay_engine, "world_data", None)
        if not wd:
            return
        spawn_chunks = wd.get("spawn_chunks", {})
        key = "0,0" if "0,0" in spawn_chunks else (next(iter(spawn_chunks)) if spawn_chunks else None)
        if not key:
            return
        chunk = spawn_chunks[key]
        blocks = chunk.get("blocks", [])
        try:
            # Prefer greedy meshing for performance
            pos, col = build_chunk_mesh_greedy(blocks)
            self._positions, self._colors = pos, col
            if self._voxel and self._voxel.available:
                self._voxel.load_mesh(pos, col)
        except Exception as e:
            self.logger.debug(f"Mesh build failed: {e}")
    
    async def render(self):
        """Render the current frame."""
        # Build mesh lazily after world is ready
        if self._positions is None:
            self._try_build_spawn_mesh()
        # If we have a 3D renderer, attempt 3D draw
        if self._voxel and self._voxel.available and self._positions:
            # First-person camera MVP
            width, height = self.config.graphics.resolution
            if self._camera and self._gameplay_engine:
                player = self._gameplay_engine.get_local_player()
                if player:
                    # Align camera to player position; simple yaw progression
                    self._camera.set_pose((player.position[0], player.position[1], player.position[2]), 45.0, -15.0)
                mvp = self._camera.view_projection_flat(width, height)
            else:
                mvp = (
                    1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1,
                )
            try:
                self._voxel.render(mvp)
            except Exception as e:
                self.logger.debug(f"3D render failed: {e}")
            await asyncio.sleep(0)
            return
        # 2D/headless fallback
        if self._headless or self._pygame is None or self._screen is None:
            await asyncio.sleep(0)
            return
        pygame = self._pygame
        screen = self._screen
        screen.fill((10, 12, 24))
        # Draw an overhead 2D map of the first spawn chunk if available
        try:
            if self._gameplay_engine and self._gameplay_engine.world_data:
                spawn_chunks = self._gameplay_engine.world_data.get("spawn_chunks", {})
                # Prefer the origin chunk if present
                chunk_key = "0,0" if "0,0" in spawn_chunks else (next(iter(spawn_chunks)) if spawn_chunks else None)
                if chunk_key:
                    chunk = spawn_chunks[chunk_key]
                    blocks = chunk.get("blocks", [])
                    # Render a 2D slice: top layer projection
                    tile = self._tile_size
                    offset_x, offset_y = 20, 20
                    for x in range(min(16, len(blocks))):
                        column = blocks[x]
                        for z in range(min(16, len(column[0]))):
                            highest_y = 63
                            if len(column) > 0 and len(column[0]) > z:
                                for y in range(len(column) - 1, -1, -1):
                                    try:
                                        if column[y][z] != 0:
                                            highest_y = y
                                            break
                                    except Exception:
                                        break
                            h = max(0, min(255, int(highest_y)))
                            color = (h//2, h//3, h)
                            pygame.draw.rect(screen, color, (offset_x + x*tile, offset_y + z*tile, tile-1, tile-1))
                    # Draw player
                    player = self._gameplay_engine.get_local_player()
                    if player:
                        px, py, pz = player.position
                        pygame.draw.rect(screen, (255, 64, 64), (offset_x + (px%16)*tile, offset_y + (pz%16)*tile, tile-2, tile-2), 2)
            # HUD
            font = pygame.font.Font(None, 24)
            text = font.render("INFINITUS - WASD/arrows to move (demo)", True, (240, 240, 240))
            screen.blit(text, (20, 2))
        except Exception as e:
            # Keep rendering loop resilient
            self.logger.debug(f"Render warning: {e}")
        pygame.display.flip()
        # Cap to target FPS
        target_fps = max(30, int(self.config.performance.max_fps))
        self._clock.tick(target_fps)
    
    async def get_events(self) -> List[Dict[str, Any]]:
        """Get input events."""
        events: List[Dict[str, Any]] = []
        if self._headless or self._pygame is None:
            return events
        pygame = self._pygame
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                events.append({"type": "quit"})
        keys = pygame.key.get_pressed()
        dx = dz = dy = 0.0
        speed = 0.2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dz -= speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dz += speed
        if keys[pygame.K_SPACE]:
            dy += speed
        if dx or dy or dz:
            events.append({"type": "move", "delta": (dx, dy, dz)})
        return events
    
    async def shutdown(self):
        self.logger.info("🔄 Shutting down Graphics Engine...")
        try:
            if self._pygame is not None:
                self._pygame.quit()
        except Exception:
            pass
        self.logger.info("✅ Graphics Engine shutdown complete")