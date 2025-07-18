"""
🌌 INFINITUS Event System
Advanced event handling and inter-system communication.
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class GameEvent:
    """Game event data structure."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    target: Optional[str] = None
    handled: bool = False
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class EventSystem:
    """Advanced event system for inter-system communication."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Event handlers
        self.handlers: Dict[str, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        
        # Event queues
        self.event_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        
        # Event history
        self.event_history: List[GameEvent] = []
        self.max_history_size = 1000
        
        # Statistics
        self.events_processed = 0
        self.events_per_second = 0.0
        self.last_stats_update = time.time()
        
        # Processing
        self.processing = False
        self.batch_size = 100
        
    async def initialize(self):
        """Initialize the event system."""
        self.logger.info("🎯 Event System initializing...")
        self.processing = True
        
        # Start event processing task
        asyncio.create_task(self._process_events_loop())
        
        self.logger.info("✅ Event System initialized")
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler for a specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for event type: {event_type}")
    
    def register_global_handler(self, handler: Callable):
        """Register a global event handler that receives all events."""
        self.global_handlers.append(handler)
        self.logger.debug("Registered global event handler")
    
    def unregister_handler(self, event_type: str, handler: Callable):
        """Unregister an event handler."""
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(handler)
                if not self.handlers[event_type]:
                    del self.handlers[event_type]
                self.logger.debug(f"Unregistered handler for event type: {event_type}")
            except ValueError:
                self.logger.warning(f"Handler not found for event type: {event_type}")
    
    async def fire_event(self, event: GameEvent):
        """Fire an event to be processed."""
        if event.priority == EventPriority.CRITICAL:
            # Process critical events immediately
            await self._handle_event(event)
        else:
            # Queue other events
            await self.event_queue.put(event)
    
    async def fire_event_sync(self, event: GameEvent):
        """Fire an event and wait for it to be processed."""
        await self._handle_event(event)
    
    async def process_events(self):
        """Process pending events (called from game loop)."""
        processed = 0
        
        # Process events from queue
        while not self.event_queue.empty() and processed < self.batch_size:
            try:
                event = self.event_queue.get_nowait()
                await self._handle_event(event)
                processed += 1
            except asyncio.QueueEmpty:
                break
        
        return processed
    
    async def _process_events_loop(self):
        """Background event processing loop."""
        while self.processing:
            try:
                # Process events from queue
                if not self.event_queue.empty():
                    event = await self.event_queue.get()
                    await self._handle_event(event)
                else:
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
    
    async def _handle_event(self, event: GameEvent):
        """Handle a single event."""
        try:
            # Update statistics
            self.events_processed += 1
            self._update_statistics()
            
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)
            
            # Call global handlers first
            for handler in self.global_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Error in global event handler: {e}")
            
            # Call specific handlers
            if event.type in self.handlers:
                for handler in self.handlers[event.type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        self.logger.error(f"Error in event handler for {event.type}: {e}")
            
            event.handled = True
            
        except Exception as e:
            self.logger.error(f"Error handling event {event.type}: {e}")
    
    def _update_statistics(self):
        """Update event processing statistics."""
        current_time = time.time()
        if current_time - self.last_stats_update >= 1.0:
            self.events_per_second = self.events_processed / (current_time - self.last_stats_update)
            self.events_processed = 0
            self.last_stats_update = current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event system statistics."""
        return {
            "events_processed": self.events_processed,
            "events_per_second": self.events_per_second,
            "queue_size": self.event_queue.qsize(),
            "handler_count": sum(len(handlers) for handlers in self.handlers.values()),
            "global_handler_count": len(self.global_handlers),
            "history_size": len(self.event_history)
        }
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[GameEvent]:
        """Get event history, optionally filtered by type."""
        if event_type:
            filtered = [e for e in self.event_history if e.type == event_type]
            return filtered[-limit:]
        else:
            return self.event_history[-limit:]
    
    async def shutdown(self):
        """Shutdown the event system."""
        self.logger.info("🔄 Shutting down Event System...")
        self.processing = False
        
        # Process remaining events
        while not self.event_queue.empty():
            try:
                event = await self.event_queue.get()
                await self._handle_event(event)
            except:
                break
        
        self.logger.info("✅ Event System shutdown complete")

# Common event types
class EventTypes:
    """Common event type constants."""
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    
    # Game events
    GAME_START = "game_start"
    GAME_PAUSE = "game_pause"
    GAME_RESUME = "game_resume"
    GAME_QUIT = "game_quit"
    
    # World events
    WORLD_CREATED = "world_created"
    WORLD_LOADED = "world_loaded"
    WORLD_SAVED = "world_saved"
    WORLD_CHANGED = "world_changed"
    
    # Player events
    PLAYER_JOIN = "player_join"
    PLAYER_LEAVE = "player_leave"
    PLAYER_SPAWN = "player_spawn"
    PLAYER_DEATH = "player_death"
    PLAYER_RESPAWN = "player_respawn"
    
    # Input events
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    
    # UI events
    UI_BUTTON_CLICK = "ui_button_click"
    UI_MENU_OPEN = "ui_menu_open"
    UI_MENU_CLOSE = "ui_menu_close"
    
    # Physics events
    COLLISION = "collision"
    EXPLOSION = "explosion"
    OBJECT_CREATED = "object_created"
    OBJECT_DESTROYED = "object_destroyed"
    
    # AI events
    NPC_SPAWN = "npc_spawn"
    NPC_DEATH = "npc_death"
    NPC_CONVERSATION = "npc_conversation"
    NPC_EMOTION_CHANGE = "npc_emotion_change"
    
    # Performance events
    PERFORMANCE_WARNING = "performance_warning"
    MEMORY_LOW = "memory_low"
    FPS_DROP = "fps_drop"

# Event factory functions
def create_system_event(event_type: str, data: Dict[str, Any] = None) -> GameEvent:
    """Create a system event."""
    return GameEvent(
        type=event_type,
        data=data or {},
        priority=EventPriority.HIGH,
        source="system"
    )

def create_player_event(event_type: str, player_id: str, data: Dict[str, Any] = None) -> GameEvent:
    """Create a player event."""
    event_data = {"player_id": player_id}
    if data:
        event_data.update(data)
    
    return GameEvent(
        type=event_type,
        data=event_data,
        source="player",
        target=player_id
    )

def create_world_event(event_type: str, world_name: str, data: Dict[str, Any] = None) -> GameEvent:
    """Create a world event."""
    event_data = {"world_name": world_name}
    if data:
        event_data.update(data)
    
    return GameEvent(
        type=event_type,
        data=event_data,
        source="world",
        priority=EventPriority.HIGH
    )

def create_ui_event(event_type: str, element_id: str, data: Dict[str, Any] = None) -> GameEvent:
    """Create a UI event."""
    event_data = {"element_id": element_id}
    if data:
        event_data.update(data)
    
    return GameEvent(
        type=event_type,
        data=event_data,
        source="ui"
    )