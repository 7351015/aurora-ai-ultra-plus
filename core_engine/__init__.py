"""
🌌 INFINITUS Core Engine Package
The heart of the ultimate sandbox survival crafting game.
"""

__version__ = "1.0.0-alpha"
__author__ = "AI Assistant"

# Core engine exports
from .game_engine import GameEngine
from .config import GameConfig
from .logger import setup_logging
from .physics_engine import PhysicsEngine
from .event_system import EventSystem
from .resource_manager import ResourceManager
from .time_manager import TimeManager
from .save_system import SaveSystem

__all__ = [
    "GameEngine",
    "GameConfig", 
    "setup_logging",
    "PhysicsEngine",
    "EventSystem",
    "ResourceManager",
    "TimeManager",
    "SaveSystem"
]