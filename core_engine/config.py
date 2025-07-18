"""
🌌 INFINITUS Game Configuration System
Manages all game settings, parameters, and configuration data.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

@dataclass
class GraphicsConfig:
    """Graphics and rendering configuration."""
    resolution: tuple = (1920, 1080)
    fullscreen: bool = False
    vsync: bool = True
    anti_aliasing: int = 4
    texture_quality: str = "ultra"  # low, medium, high, ultra
    shadow_quality: str = "ultra"
    particle_quality: str = "ultra"
    render_distance: int = 1000
    fov: float = 75.0
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0

@dataclass
class AudioConfig:
    """Audio configuration."""
    master_volume: float = 1.0
    music_volume: float = 0.8
    sfx_volume: float = 1.0
    voice_volume: float = 1.0
    ambient_volume: float = 0.7
    audio_quality: str = "high"  # low, medium, high
    spatial_audio: bool = True
    audio_device: Optional[str] = None

@dataclass
class GameplayConfig:
    """Gameplay settings."""
    difficulty: str = "normal"  # peaceful, easy, normal, hard, nightmare
    auto_save: bool = True
    auto_save_interval: int = 300  # seconds
    permadeath: bool = False
    friendly_fire: bool = False
    pvp_enabled: bool = True
    creative_mode: bool = False
    cheats_enabled: bool = False
    time_scale: float = 1.0
    weather_enabled: bool = True
    day_night_cycle: bool = True
    hunger_enabled: bool = True
    thirst_enabled: bool = True
    temperature_enabled: bool = True

@dataclass
class WorldConfig:
    """World generation configuration."""
    world_size: str = "infinite"  # small, medium, large, infinite
    world_type: str = "normal"  # normal, superflat, amplified, custom
    seed: Optional[int] = None
    biome_diversity: float = 1.0
    structure_generation: bool = True
    ore_generation: bool = True
    cave_generation: bool = True
    dungeon_generation: bool = True
    village_generation: bool = True
    civilization_generation: bool = True
    portal_generation: bool = True
    sky_islands: bool = True
    underground_cities: bool = True
    space_zones: bool = True

@dataclass
class AIConfig:
    """AI and NPC configuration."""
    ai_difficulty: str = "normal"  # simple, normal, advanced, genius
    npc_intelligence: str = "high"  # low, medium, high, sentient
    npc_memory: bool = True
    npc_emotions: bool = True
    npc_relationships: bool = True
    npc_evolution: bool = True
    npc_voice_interaction: bool = True
    ai_content_generation: bool = True
    consciousness_simulation: bool = True
    dream_simulation: bool = True

@dataclass
class MultiplayerConfig:
    """Multiplayer configuration."""
    max_players: int = 100
    server_name: str = "Infinitus Server"
    server_description: str = "The ultimate sandbox experience"
    server_password: Optional[str] = None
    public_server: bool = False
    cross_platform: bool = True
    voice_chat: bool = True
    text_chat: bool = True
    player_collision: bool = True
    shared_inventory: bool = False

@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    max_fps: int = 144
    cpu_threads: int = -1  # -1 for auto-detect
    gpu_acceleration: bool = True
    memory_limit: int = 8192  # MB
    cache_size: int = 2048  # MB
    chunk_loading_distance: int = 16
    entity_limit: int = 10000
    particle_limit: int = 50000
    dynamic_loading: bool = True
    level_of_detail: bool = True
    occlusion_culling: bool = True

@dataclass
class ModdingConfig:
    """Modding and customization settings."""
    mods_enabled: bool = True
    mod_directory: str = "mods"
    auto_update_mods: bool = True
    mod_verification: bool = True
    custom_content: bool = True
    scripting_enabled: bool = True
    lua_scripting: bool = True
    python_scripting: bool = True
    visual_scripting: bool = True

class GameConfig:
    """Main game configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config")
        self.config_file = self.config_path / "game_config.yaml"
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration sections
        self.graphics = GraphicsConfig()
        self.audio = AudioConfig()
        self.gameplay = GameplayConfig()
        self.world = WorldConfig()
        self.ai = AIConfig()
        self.multiplayer = MultiplayerConfig()
        self.performance = PerformanceConfig()
        self.modding = ModdingConfig()
        
        # Additional settings
        self.debug_mode = False
        self.developer_mode = False
        self.telemetry_enabled = True
        self.auto_update = True
        self.language = "en"
        self.region = "US"
        
        # Create config directory if it doesn't exist
        self.config_path.mkdir(parents=True, exist_ok=True)
    
    async def load(self) -> bool:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                self.logger.info(f"Loading configuration from {self.config_file}")
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Load each section
                if 'graphics' in config_data:
                    self.graphics = GraphicsConfig(**config_data['graphics'])
                if 'audio' in config_data:
                    self.audio = AudioConfig(**config_data['audio'])
                if 'gameplay' in config_data:
                    self.gameplay = GameplayConfig(**config_data['gameplay'])
                if 'world' in config_data:
                    self.world = WorldConfig(**config_data['world'])
                if 'ai' in config_data:
                    self.ai = AIConfig(**config_data['ai'])
                if 'multiplayer' in config_data:
                    self.multiplayer = MultiplayerConfig(**config_data['multiplayer'])
                if 'performance' in config_data:
                    self.performance = PerformanceConfig(**config_data['performance'])
                if 'modding' in config_data:
                    self.modding = ModdingConfig(**config_data['modding'])
                
                # Load additional settings
                self.debug_mode = config_data.get('debug_mode', False)
                self.developer_mode = config_data.get('developer_mode', False)
                self.telemetry_enabled = config_data.get('telemetry_enabled', True)
                self.auto_update = config_data.get('auto_update', True)
                self.language = config_data.get('language', 'en')
                self.region = config_data.get('region', 'US')
                
                self.logger.info("✅ Configuration loaded successfully")
                return True
            else:
                self.logger.info("No configuration file found, using defaults")
                await self.save()  # Save default configuration
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return False
    
    async def save(self) -> bool:
        """Save configuration to file."""
        try:
            config_data = {
                'graphics': asdict(self.graphics),
                'audio': asdict(self.audio),
                'gameplay': asdict(self.gameplay),
                'world': asdict(self.world),
                'ai': asdict(self.ai),
                'multiplayer': asdict(self.multiplayer),
                'performance': asdict(self.performance),
                'modding': asdict(self.modding),
                'debug_mode': self.debug_mode,
                'developer_mode': self.developer_mode,
                'telemetry_enabled': self.telemetry_enabled,
                'auto_update': self.auto_update,
                'language': self.language,
                'region': self.region
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        try:
            keys = key.split('.')
            value = self
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set a configuration value by key."""
        try:
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                else:
                    return False
            
            if hasattr(obj, keys[-1]):
                setattr(obj, keys[-1], value)
                return True
            return False
            
        except Exception:
            return False
    
    def reset_to_defaults(self):
        """Reset all configuration to default values."""
        self.graphics = GraphicsConfig()
        self.audio = AudioConfig()
        self.gameplay = GameplayConfig()
        self.world = WorldConfig()
        self.ai = AIConfig()
        self.multiplayer = MultiplayerConfig()
        self.performance = PerformanceConfig()
        self.modding = ModdingConfig()
        
        self.debug_mode = False
        self.developer_mode = False
        self.telemetry_enabled = True
        self.auto_update = True
        self.language = "en"
        self.region = "US"
        
        self.logger.info("Configuration reset to defaults")
    
    def validate(self) -> bool:
        """Validate configuration values."""
        try:
            # Validate graphics settings
            if self.graphics.resolution[0] < 800 or self.graphics.resolution[1] < 600:
                self.logger.warning("Resolution too low, setting to minimum")
                self.graphics.resolution = (800, 600)
            
            # Validate audio settings
            self.audio.master_volume = max(0.0, min(1.0, self.audio.master_volume))
            self.audio.music_volume = max(0.0, min(1.0, self.audio.music_volume))
            self.audio.sfx_volume = max(0.0, min(1.0, self.audio.sfx_volume))
            
            # Validate performance settings
            if self.performance.max_fps < 30:
                self.performance.max_fps = 30
            if self.performance.memory_limit < 1024:
                self.performance.memory_limit = 1024
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_display_info(self) -> Dict[str, Any]:
        """Get display information for UI."""
        return {
            "Resolution": f"{self.graphics.resolution[0]}x{self.graphics.resolution[1]}",
            "Fullscreen": "Yes" if self.graphics.fullscreen else "No",
            "Graphics Quality": self.graphics.texture_quality.title(),
            "Audio Quality": self.audio.audio_quality.title(),
            "Difficulty": self.gameplay.difficulty.title(),
            "World Type": self.world.world_type.title(),
            "AI Intelligence": self.ai.npc_intelligence.title(),
            "Max Players": str(self.multiplayer.max_players),
            "Mods Enabled": "Yes" if self.modding.mods_enabled else "No"
        }