"""
🌌 INFINITUS Graphics Engine
Advanced graphics and rendering system.
"""

import asyncio
import logging
import os
import math
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
        # Meshing loop
        self._meshing_task: Optional[asyncio.Task] = None
        self._mesh_rebuild_requested: bool = False
    
    def set_gameplay_engine(self, engine: Any) -> None:
        self._gameplay_engine = engine
    
    def refresh_world_mesh(self) -> None:
        """Public method to rebuild world mesh on-demand (e.g., after block edits)."""
        self._positions = None
        self._colors = None
        self._mesh_rebuild_requested = True
        self._try_build_spawn_mesh()
    
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
            # Enable relative mouse for FPS look
            try:
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.mouse.get_rel()
            except Exception:
                pass
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
        # Start background meshing loop
        if self._meshing_task is None:
            self._meshing_task = asyncio.create_task(self._meshing_loop())
    
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
    
    async def _meshing_loop(self):
        """Background loop to (re)build combined mesh for visible chunks."""
        while True:
            try:
                await asyncio.sleep(0.5)
                if not self._gameplay_engine:
                    continue
                # Rebuild if requested or periodically
                if not self._mesh_rebuild_requested and self._positions is not None:
                    continue
                chunks = []
                try:
                    chunks = self._gameplay_engine.get_render_chunks()
                except Exception:
                    chunks = []
                if not chunks:
                    continue
                # Combine meshes from chunks
                combined_pos: List[float] = []
                combined_col: List[float] = []
                for (cx, cz, blocks) in chunks:
                    try:
                        pos, col = build_chunk_mesh_greedy(blocks)
                        # Offset positions by chunk origin
                        for i in range(0, len(pos), 3):
                            combined_pos.extend([pos[i] + cx * 16, pos[i + 1], pos[i + 2] + cz * 16])
                        combined_col.extend(col)
                    except Exception:
                        continue
                if combined_pos:
                    self._positions, self._colors = combined_pos, combined_col
                    if self._voxel and self._voxel.available:
                        self._voxel.load_mesh(combined_pos, combined_col)
                self._mesh_rebuild_requested = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Meshing loop error: {e}")
                continue
    
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
                    yaw, pitch = (0.0, 0.0)
                    try:
                        yaw, pitch = player.rotation
                    except Exception:
                        pass
                    self._camera.set_pose((player.position[0], player.position[1], player.position[2]), yaw, pitch)
                mvp = self._camera.view_projection_flat(width, height)
            else:
                mvp = (
                    1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1,
                )
            try:
                # Simple brightness from time-of-day (if available)
                brightness = 1.0
                try:
                    if self._gameplay_engine and getattr(self._gameplay_engine, "_time_of_day", None) is not None:
                        tod = float(self._gameplay_engine._time_of_day)
                        # Peak at noon (~6000), dim at midnight
                        brightness = 0.2 + 0.8 * max(0.0, math.cos((tod - 6000.0) * (3.14159/12000.0)))
                    # Weather dimming
                    if self._gameplay_engine and getattr(self._gameplay_engine, "weather", None) is not None:
                        brightness *= float(self._gameplay_engine.weather.get_brightness_modifier())
                except Exception:
                    pass
                self._voxel.render(mvp, brightness)
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
                    # Draw player marker
                    player = self._gameplay_engine.get_local_player()
                    if player:
                        px, py, pz = player.position
                        pygame.draw.rect(screen, (255, 64, 64), (offset_x + (px%16)*tile, offset_y + (pz%16)*tile, tile-2, tile-2), 2)
            # HUD: crosshair and hotbar
            cx = screen.get_width() // 2
            cy = screen.get_height() // 2
            pygame.draw.line(screen, (240, 240, 240), (cx - 8, cy), (cx + 8, cy), 1)
            pygame.draw.line(screen, (240, 240, 240), (cx, cy - 8), (cx, cy + 8), 1)
            # Hotbar
            player = self._gameplay_engine.get_local_player() if self._gameplay_engine else None
            if player:
                font = pygame.font.Font(None, 20)
                items = player.hotbar
                sel = player.hotbar_index
                bar_w = 36 * max(1, len(items))
                bar_x = (screen.get_width() - bar_w) // 2
                bar_y = screen.get_height() - 48
                for i, name in enumerate(items):
                    x = bar_x + i * 36
                    rect = pygame.Rect(x, bar_y, 32, 32)
                    pygame.draw.rect(screen, (30, 30, 30), rect)
                    pygame.draw.rect(screen, (200, 200, 200) if i == sel else (80, 80, 80), rect, 2)
                    label = font.render(name[:5], True, (220, 220, 220))
                    screen.blit(label, (x + 4, bar_y + 8))
                # Overlay (coords, FPS, time-of-day, weather)
                try:
                    font_small = pygame.font.Font(None, 18)
                    fps = self._clock.get_fps() if self._clock else 0.0
                    tod = 0.0
                    weather = "clear"
                    if hasattr(self._gameplay_engine, "_time_of_day"):
                        tod = float(self._gameplay_engine._time_of_day)
                    if hasattr(self._gameplay_engine, "weather") and self._gameplay_engine.weather:
                        w = self._gameplay_engine.weather.get_overlay_info()
                        weather = f"{w['state']} {w['intensity']:.2f}"
                    overlay = f"XYZ: {player.position[0]:.1f} {player.position[1]:.1f} {player.position[2]:.1f}  |  FPS: {fps:.1f}  |  TOD: {tod:.0f}  |  WX: {weather}"
                    screen.blit(font_small.render(overlay, True, (230, 230, 230)), (8, 8))
                except Exception:
                    pass
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
            elif evt.type == pygame.MOUSEBUTTONDOWN:
                if evt.button == 1:
                    events.append({"type": "action", "button": "break"})
                elif evt.button == 3:
                    events.append({"type": "action", "button": "place"})
                elif evt.button in (4, 5):
                    events.append({"type": "hotbar", "delta": -1 if evt.button == 4 else 1})
        # Mouse look (relative movement)
        try:
            mx, my = pygame.mouse.get_rel()
            if mx or my:
                events.append({"type": "look", "delta": (float(mx), float(my))})
        except Exception:
            pass
        keys = pygame.key.get_pressed()
        dx = dz = dy = 0.0
        base_speed = 0.08
        speed = base_speed * (2.0 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
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
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            dy -= speed
        if dx or dy or dz:
            events.append({"type": "move", "delta": (dx, dy, dz)})
        return events
    
    async def shutdown(self):
        self.logger.info("🔄 Shutting down Graphics Engine...")
        try:
            if self._meshing_task:
                self._meshing_task.cancel()
            if self._pygame is not None:
                self._pygame.quit()
        except Exception:
            pass
        self.logger.info("✅ Graphics Engine shutdown complete")