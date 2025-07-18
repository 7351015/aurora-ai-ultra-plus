"""
🌌 INFINITUS Main Game Engine
The core orchestrator for the ultimate sandbox survival crafting game.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import gc
import os

from .config import GameConfig
from .logger import get_logger, log_performance, log_error_with_context
from .event_system import EventSystem, GameEvent
from .physics_engine import PhysicsEngine
from .time_manager import TimeManager
from .resource_manager import ResourceManager
from .save_system import SaveSystem

@dataclass
class GameState:
    """Current game state information."""
    running: bool = False
    paused: bool = False
    world_loaded: bool = False
    current_world: Optional[str] = None
    player_count: int = 0
    game_time: float = 0.0
    real_time: float = 0.0
    frame_count: int = 0
    
@dataclass
class PerformanceMetrics:
    """Performance monitoring data."""
    fps: float = 0.0
    frame_time: float = 0.0
    update_time: float = 0.0
    render_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    entity_count: int = 0
    chunk_count: int = 0

class GameEngine:
    """Main game engine that orchestrates all systems."""
    
    def __init__(self, config: GameConfig, **systems):
        self.config = config
        self.logger = get_logger()
        
        # Core systems
        self.graphics_engine = systems.get('graphics_engine')
        self.world_generator = systems.get('world_generator')
        self.consciousness_engine = systems.get('consciousness_engine')
        self.avatar_system = systems.get('avatar_system')
        self.narrative_engine = systems.get('narrative_engine')
        self.network_manager = systems.get('network_manager')
        
        # Internal systems
        self.event_system = EventSystem()
        self.physics_engine = PhysicsEngine(config)
        self.time_manager = TimeManager()
        self.resource_manager = ResourceManager()
        self.save_system = SaveSystem()
        
        # Game state
        self.state = GameState()
        self.performance = PerformanceMetrics()
        
        # Timing
        self.last_frame_time = 0.0
        self.target_fps = config.performance.max_fps
        self.frame_time_target = 1.0 / self.target_fps
        
        # Performance monitoring
        self.performance_update_interval = 1.0  # seconds
        self.last_performance_update = 0.0
        
        # Event queues
        self.event_queue = asyncio.Queue()
        self.system_events = []
        
        # Subsystem managers
        self.subsystems = {}
        
        self.logger.info("🎮 Game Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize all game systems."""
        try:
            self.logger.info("🔧 Initializing game engine systems...")
            
            # Initialize core systems
            await self.event_system.initialize()
            await self.physics_engine.initialize()
            await self.time_manager.initialize()
            await self.resource_manager.initialize()
            await self.save_system.initialize()
            
            # Initialize subsystems
            for name, system in self.subsystems.items():
                if hasattr(system, 'initialize'):
                    await system.initialize()
                    self.logger.info(f"✅ {name} initialized")
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Start performance monitoring
            self._start_performance_monitoring()
            
            self.logger.info("✅ Game engine initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize game engine: {e}")
            log_error_with_context(e, "game_engine_init")
            return False
    
    async def create_world(self, world_name: str, world_config: Optional[Dict] = None) -> bool:
        """Create a new world."""
        try:
            self.logger.info(f"🌍 Creating new world: {world_name}")
            
            # Generate world using world generator
            if self.world_generator:
                world_data = await self.world_generator.generate_world(world_name, world_config)
                
                # Initialize physics for the world
                await self.physics_engine.load_world(world_data)
                
                # Initialize AI consciousness for the world
                if self.consciousness_engine:
                    await self.consciousness_engine.initialize_world(world_data)
                
                # Initialize narrative for the world
                if self.narrative_engine:
                    await self.narrative_engine.initialize_world(world_data)
                
                self.state.world_loaded = True
                self.state.current_world = world_name
                
                # Fire world created event
                await self.event_system.fire_event(GameEvent(
                    type="world_created",
                    data={"world_name": world_name, "world_data": world_data}
                ))
                
                self.logger.info(f"✅ World '{world_name}' created successfully")
                return True
            else:
                self.logger.error("❌ World generator not available")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create world: {e}")
            log_error_with_context(e, "create_world")
            return False
    
    async def load_world(self, world_name: str) -> bool:
        """Load an existing world."""
        try:
            self.logger.info(f"📂 Loading world: {world_name}")
            
            # Load world data
            world_data = await self.save_system.load_world(world_name)
            
            if world_data:
                # Load into physics engine
                await self.physics_engine.load_world(world_data)
                
                # Load AI consciousness state
                if self.consciousness_engine:
                    await self.consciousness_engine.load_world_state(world_data)
                
                # Load narrative state
                if self.narrative_engine:
                    await self.narrative_engine.load_world_state(world_data)
                
                self.state.world_loaded = True
                self.state.current_world = world_name
                
                # Fire world loaded event
                await self.event_system.fire_event(GameEvent(
                    type="world_loaded",
                    data={"world_name": world_name, "world_data": world_data}
                ))
                
                self.logger.info(f"✅ World '{world_name}' loaded successfully")
                return True
            else:
                self.logger.error(f"❌ World '{world_name}' not found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load world: {e}")
            log_error_with_context(e, "load_world")
            return False
    
    async def save_world(self) -> bool:
        """Save the current world."""
        try:
            if not self.state.world_loaded:
                self.logger.warning("⚠️ No world loaded to save")
                return False
            
            self.logger.info(f"💾 Saving world: {self.state.current_world}")
            
            # Collect world data from all systems
            world_data = {}
            
            # Physics state
            world_data['physics'] = await self.physics_engine.get_world_state()
            
            # AI consciousness state
            if self.consciousness_engine:
                world_data['consciousness'] = await self.consciousness_engine.get_world_state()
            
            # Narrative state
            if self.narrative_engine:
                world_data['narrative'] = await self.narrative_engine.get_world_state()
            
            # Player data
            if self.avatar_system:
                world_data['players'] = await self.avatar_system.get_all_player_data()
            
            # Save to disk
            success = await self.save_system.save_world(self.state.current_world, world_data)
            
            if success:
                self.logger.info("✅ World saved successfully")
                return True
            else:
                self.logger.error("❌ Failed to save world")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save world: {e}")
            log_error_with_context(e, "save_world")
            return False
    
    async def update(self) -> None:
        """Update all game systems."""
        try:
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Update game time
            self.state.game_time += delta_time * self.config.gameplay.time_scale
            self.state.real_time = current_time
            self.state.frame_count += 1
            
            # Update time manager
            await self.time_manager.update(delta_time)
            
            # Update physics
            physics_start = time.time()
            await self.physics_engine.update(delta_time)
            physics_time = time.time() - physics_start
            
            # Update AI consciousness
            if self.consciousness_engine:
                await self.consciousness_engine.update(delta_time)
            
            # Update narrative engine
            if self.narrative_engine:
                await self.narrative_engine.update(delta_time)
            
            # Update avatar system
            if self.avatar_system:
                await self.avatar_system.update(delta_time)
            
            # Update world generator (streaming)
            if self.world_generator:
                await self.world_generator.update(delta_time)
            
            # Update network manager
            if self.network_manager:
                await self.network_manager.update(delta_time)
            
            # Process events
            await self.event_system.process_events()
            
            # Update performance metrics
            update_time = time.time() - current_time
            self.performance.update_time = update_time * 1000  # Convert to milliseconds
            
            # Update performance monitoring
            await self._update_performance_metrics(delta_time)
            
            # Auto-save if enabled
            if self.config.gameplay.auto_save:
                await self._check_auto_save()
            
        except Exception as e:
            self.logger.error(f"❌ Error in game update: {e}")
            log_error_with_context(e, "game_update")
    
    async def render(self) -> None:
        """Render the game."""
        try:
            if self.graphics_engine:
                render_start = time.time()
                await self.graphics_engine.render()
                render_time = time.time() - render_start
                self.performance.render_time = render_time * 1000  # Convert to milliseconds
                
        except Exception as e:
            self.logger.error(f"❌ Error in game render: {e}")
            log_error_with_context(e, "game_render")
    
    async def get_events(self) -> List[GameEvent]:
        """Get pending game events."""
        events = []
        
        # Get events from graphics engine
        if self.graphics_engine:
            graphics_events = await self.graphics_engine.get_events()
            events.extend(graphics_events)
        
        # Get events from network manager
        if self.network_manager:
            network_events = await self.network_manager.get_events()
            events.extend(network_events)
        
        # Get system events
        events.extend(self.system_events)
        self.system_events.clear()
        
        return events
    
    async def handle_event(self, event: GameEvent) -> None:
        """Handle a game event."""
        try:
            await self.event_system.fire_event(event)
            
            # Handle engine-specific events
            if event.type == "quit":
                self.state.running = False
            elif event.type == "pause":
                self.state.paused = not self.state.paused
                self.logger.info(f"Game {'paused' if self.state.paused else 'resumed'}")
            elif event.type == "save":
                await self.save_world()
            elif event.type == "load":
                world_name = event.data.get('world_name')
                if world_name:
                    await self.load_world(world_name)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling event: {e}")
            log_error_with_context(e, "handle_event")
    
    async def shutdown(self) -> None:
        """Shutdown the game engine."""
        try:
            self.logger.info("🔄 Shutting down game engine...")
            
            # Save current world if loaded
            if self.state.world_loaded:
                await self.save_world()
            
            # Shutdown all systems
            if self.graphics_engine:
                await self.graphics_engine.shutdown()
            
            if self.consciousness_engine:
                await self.consciousness_engine.shutdown()
            
            if self.network_manager:
                await self.network_manager.shutdown()
            
            # Shutdown core systems
            await self.physics_engine.shutdown()
            await self.resource_manager.shutdown()
            await self.save_system.shutdown()
            
            # Shutdown subsystems
            for name, system in self.subsystems.items():
                if hasattr(system, 'shutdown'):
                    await system.shutdown()
            
            self.state.running = False
            self.logger.info("✅ Game engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
            log_error_with_context(e, "shutdown")
    
    def _setup_event_handlers(self):
        """Setup event handlers for the game engine."""
        # Register event handlers
        self.event_system.register_handler("system_error", self._handle_system_error)
        self.event_system.register_handler("performance_warning", self._handle_performance_warning)
        self.event_system.register_handler("world_changed", self._handle_world_changed)
    
    async def _handle_system_error(self, event: GameEvent):
        """Handle system error events."""
        error_data = event.data
        self.logger.error(f"System error: {error_data.get('message', 'Unknown error')}")
    
    async def _handle_performance_warning(self, event: GameEvent):
        """Handle performance warning events."""
        warning_data = event.data
        self.logger.warning(f"Performance warning: {warning_data.get('message', 'Unknown warning')}")
    
    async def _handle_world_changed(self, event: GameEvent):
        """Handle world change events."""
        world_data = event.data
        self.logger.info(f"World changed: {world_data.get('change_type', 'Unknown change')}")
    
    def _start_performance_monitoring(self):
        """Start performance monitoring."""
        self.last_performance_update = time.time()
        self.performance.fps = 0.0
        self.performance.frame_time = 0.0
    
    async def _update_performance_metrics(self, delta_time: float):
        """Update performance metrics."""
        current_time = time.time()
        
        # Update FPS
        if delta_time > 0:
            self.performance.fps = 1.0 / delta_time
            self.performance.frame_time = delta_time * 1000  # Convert to milliseconds
        
        # Update memory usage (simplified)
        try:
            # Try to get memory info if available
            import resource
            memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, ru_maxrss is in KB, on macOS it's in bytes
            self.performance.memory_usage = memory_kb / 1024  # MB
        except:
            self.performance.memory_usage = 0.0
        
        # Update CPU usage (simplified)
        self.performance.cpu_usage = 0.0  # Placeholder
        
        # Log performance metrics periodically
        if current_time - self.last_performance_update >= self.performance_update_interval:
            log_performance(
                self.performance.fps,
                self.performance.frame_time,
                self.performance.memory_usage
            )
            self.last_performance_update = current_time
            
            # Check for performance issues
            if self.performance.fps < 30:
                await self.event_system.fire_event(GameEvent(
                    type="performance_warning",
                    data={"message": f"Low FPS: {self.performance.fps:.1f}"}
                ))
            
            if self.performance.memory_usage > self.config.performance.memory_limit:
                await self.event_system.fire_event(GameEvent(
                    type="performance_warning",
                    data={"message": f"High memory usage: {self.performance.memory_usage:.1f}MB"}
                ))
                
                # Force garbage collection
                gc.collect()
    
    async def _check_auto_save(self):
        """Check if auto-save should be performed."""
        if not hasattr(self, '_last_auto_save'):
            self._last_auto_save = time.time()
        
        current_time = time.time()
        if current_time - self._last_auto_save >= self.config.gameplay.auto_save_interval:
            await self.save_world()
            self._last_auto_save = current_time
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get current performance information."""
        return {
            "fps": self.performance.fps,
            "frame_time": self.performance.frame_time,
            "update_time": self.performance.update_time,
            "render_time": self.performance.render_time,
            "memory_usage": self.performance.memory_usage,
            "cpu_usage": self.performance.cpu_usage,
            "entity_count": self.performance.entity_count,
            "chunk_count": self.performance.chunk_count
        }
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state information."""
        return {
            "running": self.state.running,
            "paused": self.state.paused,
            "world_loaded": self.state.world_loaded,
            "current_world": self.state.current_world,
            "player_count": self.state.player_count,
            "game_time": self.state.game_time,
            "real_time": self.state.real_time,
            "frame_count": self.state.frame_count
        }