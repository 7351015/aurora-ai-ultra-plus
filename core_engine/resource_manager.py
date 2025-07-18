"""
🌌 INFINITUS Resource Manager
Advanced resource loading, caching, and management system.
"""

import asyncio
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import weakref

# Resource loading libraries
try:
    from PIL import Image
    import numpy as np
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False

class ResourceType(Enum):
    """Types of resources."""
    TEXTURE = "texture"
    MODEL = "model"
    SOUND = "sound"
    MUSIC = "music"
    SCRIPT = "script"
    DATA = "data"
    SHADER = "shader"
    FONT = "font"
    ANIMATION = "animation"
    PARTICLE = "particle"
    MATERIAL = "material"
    WORLD = "world"
    CONFIG = "config"

class ResourceState(Enum):
    """Resource loading states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    CACHED = "cached"

@dataclass
class ResourceInfo:
    """Resource metadata."""
    id: str
    type: ResourceType
    path: str
    size: int = 0
    hash: str = ""
    state: ResourceState = ResourceState.UNLOADED
    load_time: float = 0.0
    last_access: float = 0.0
    reference_count: int = 0
    dependencies: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

class ResourceCache:
    """Intelligent resource caching system."""
    
    def __init__(self, max_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_size = max_size
        self.current_size = 0
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.resource_sizes: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def get(self, resource_id: str) -> Optional[Any]:
        """Get a resource from cache."""
        with self.lock:
            if resource_id in self.cache:
                self.access_times[resource_id] = time.time()
                return self.cache[resource_id]
            return None
    
    def put(self, resource_id: str, resource: Any, size: int):
        """Put a resource in cache."""
        with self.lock:
            # Remove if already exists
            if resource_id in self.cache:
                self.remove(resource_id)
            
            # Make space if needed
            while self.current_size + size > self.max_size and self.cache:
                self._evict_lru()
            
            # Add to cache
            self.cache[resource_id] = resource
            self.access_times[resource_id] = time.time()
            self.resource_sizes[resource_id] = size
            self.current_size += size
    
    def remove(self, resource_id: str):
        """Remove a resource from cache."""
        with self.lock:
            if resource_id in self.cache:
                del self.cache[resource_id]
                del self.access_times[resource_id]
                size = self.resource_sizes.pop(resource_id, 0)
                self.current_size -= size
    
    def _evict_lru(self):
        """Evict least recently used resource."""
        if not self.access_times:
            return
        
        lru_id = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.remove(lru_id)
    
    def clear(self):
        """Clear all cached resources."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.resource_sizes.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': self.current_size,
                'max_size': self.max_size,
                'count': len(self.cache),
                'usage': self.current_size / self.max_size if self.max_size > 0 else 0
            }

class ResourceManager:
    """Advanced resource management system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Resource registry
        self.resources: Dict[str, ResourceInfo] = {}
        self.resource_cache = ResourceCache()
        
        # Resource paths
        self.resource_paths = {
            ResourceType.TEXTURE: "assets/textures",
            ResourceType.MODEL: "assets/models",
            ResourceType.SOUND: "assets/sounds",
            ResourceType.MUSIC: "assets/music",
            ResourceType.SCRIPT: "assets/scripts",
            ResourceType.DATA: "assets/data",
            ResourceType.SHADER: "assets/shaders",
            ResourceType.FONT: "assets/fonts",
            ResourceType.ANIMATION: "assets/animations",
            ResourceType.PARTICLE: "assets/particles",
            ResourceType.MATERIAL: "assets/materials",
            ResourceType.WORLD: "worlds",
            ResourceType.CONFIG: "config"
        }
        
        # Resource loaders
        self.loaders: Dict[ResourceType, Callable] = {
            ResourceType.TEXTURE: self._load_texture,
            ResourceType.MODEL: self._load_model,
            ResourceType.SOUND: self._load_sound,
            ResourceType.MUSIC: self._load_music,
            ResourceType.SCRIPT: self._load_script,
            ResourceType.DATA: self._load_data,
            ResourceType.SHADER: self._load_shader,
            ResourceType.FONT: self._load_font,
            ResourceType.ANIMATION: self._load_animation,
            ResourceType.PARTICLE: self._load_particle,
            ResourceType.MATERIAL: self._load_material,
            ResourceType.WORLD: self._load_world,
            ResourceType.CONFIG: self._load_config
        }
        
        # Loading state
        self.loading_tasks: Dict[str, asyncio.Task] = {}
        self.loading_callbacks: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.load_count = 0
        self.total_load_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Resource watching
        self.watch_enabled = False
        self.watched_files: Dict[str, float] = {}
        
        self.logger.info("📦 Resource Manager initialized")
    
    async def initialize(self):
        """Initialize the resource manager."""
        self.logger.info("🔧 Initializing Resource Manager...")
        
        # Create resource directories
        for resource_type, path in self.resource_paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Scan for existing resources
        await self._scan_resources()
        
        # Start file watching if enabled
        if self.watch_enabled:
            asyncio.create_task(self._watch_files())
        
        self.logger.info("✅ Resource Manager initialization complete")
    
    async def _scan_resources(self):
        """Scan for existing resources."""
        self.logger.info("🔍 Scanning for resources...")
        
        resource_count = 0
        for resource_type, base_path in self.resource_paths.items():
            path = Path(base_path)
            if path.exists():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        resource_id = self._generate_resource_id(file_path, resource_type)
                        
                        # Create resource info
                        resource_info = ResourceInfo(
                            id=resource_id,
                            type=resource_type,
                            path=str(file_path),
                            size=file_path.stat().st_size,
                            hash=self._calculate_file_hash(file_path)
                        )
                        
                        self.resources[resource_id] = resource_info
                        resource_count += 1
        
        self.logger.info(f"✅ Found {resource_count} resources")
    
    def _generate_resource_id(self, file_path: Path, resource_type: ResourceType) -> str:
        """Generate a unique resource ID."""
        # Use relative path without extension as base ID
        base_path = Path(self.resource_paths[resource_type])
        try:
            relative_path = file_path.relative_to(base_path)
            return str(relative_path.with_suffix(''))
        except ValueError:
            # Fallback to filename
            return file_path.stem
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    async def load_resource(self, resource_id: str, resource_type: Optional[ResourceType] = None) -> Optional[Any]:
        """Load a resource asynchronously."""
        try:
            # Check cache first
            cached_resource = self.resource_cache.get(resource_id)
            if cached_resource is not None:
                self.cache_hits += 1
                return cached_resource
            
            self.cache_misses += 1
            
            # Check if resource exists
            if resource_id not in self.resources:
                # Try to find resource by scanning
                if resource_type:
                    await self._find_resource(resource_id, resource_type)
                
                if resource_id not in self.resources:
                    self.logger.warning(f"Resource not found: {resource_id}")
                    return None
            
            resource_info = self.resources[resource_id]
            
            # Check if already loading
            if resource_id in self.loading_tasks:
                return await self.loading_tasks[resource_id]
            
            # Start loading
            resource_info.state = ResourceState.LOADING
            load_task = asyncio.create_task(self._load_resource_async(resource_info))
            self.loading_tasks[resource_id] = load_task
            
            try:
                resource = await load_task
                return resource
            finally:
                self.loading_tasks.pop(resource_id, None)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load resource {resource_id}: {e}")
            if resource_id in self.resources:
                self.resources[resource_id].state = ResourceState.ERROR
            return None
    
    async def _find_resource(self, resource_id: str, resource_type: ResourceType):
        """Try to find a resource by ID and type."""
        base_path = Path(self.resource_paths[resource_type])
        
        # Common extensions for each resource type
        extensions = {
            ResourceType.TEXTURE: ['.png', '.jpg', '.jpeg', '.bmp', '.tga', '.dds'],
            ResourceType.MODEL: ['.obj', '.fbx', '.dae', '.3ds', '.ply', '.stl'],
            ResourceType.SOUND: ['.wav', '.mp3', '.ogg', '.flac'],
            ResourceType.MUSIC: ['.wav', '.mp3', '.ogg', '.flac'],
            ResourceType.SCRIPT: ['.py', '.lua', '.js'],
            ResourceType.DATA: ['.json', '.yaml', '.xml', '.csv'],
            ResourceType.SHADER: ['.glsl', '.hlsl', '.vert', '.frag'],
            ResourceType.FONT: ['.ttf', '.otf', '.woff', '.woff2'],
            ResourceType.ANIMATION: ['.anim', '.fbx', '.dae'],
            ResourceType.PARTICLE: ['.json', '.xml'],
            ResourceType.MATERIAL: ['.json', '.mtl'],
            ResourceType.WORLD: ['.world', '.json'],
            ResourceType.CONFIG: ['.json', '.yaml', '.ini']
        }
        
        # Try to find file with common extensions
        for ext in extensions.get(resource_type, []):
            potential_path = base_path / f"{resource_id}{ext}"
            if potential_path.exists():
                # Create resource info
                resource_info = ResourceInfo(
                    id=resource_id,
                    type=resource_type,
                    path=str(potential_path),
                    size=potential_path.stat().st_size,
                    hash=self._calculate_file_hash(potential_path)
                )
                
                self.resources[resource_id] = resource_info
                break
    
    async def _load_resource_async(self, resource_info: ResourceInfo) -> Optional[Any]:
        """Load a resource asynchronously."""
        start_time = time.time()
        
        try:
            # Get the appropriate loader
            loader = self.loaders.get(resource_info.type)
            if not loader:
                self.logger.error(f"No loader for resource type: {resource_info.type}")
                return None
            
            # Load the resource
            resource = await loader(resource_info.path)
            
            if resource is not None:
                # Update resource info
                resource_info.state = ResourceState.LOADED
                resource_info.load_time = time.time() - start_time
                resource_info.last_access = time.time()
                resource_info.reference_count += 1
                
                # Cache the resource
                self.resource_cache.put(resource_info.id, resource, resource_info.size)
                
                # Update statistics
                self.load_count += 1
                self.total_load_time += resource_info.load_time
                
                # Fire callbacks
                await self._fire_load_callbacks(resource_info.id, resource)
                
                self.logger.debug(f"✅ Loaded resource: {resource_info.id} ({resource_info.load_time:.3f}s)")
                return resource
            else:
                resource_info.state = ResourceState.ERROR
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading resource {resource_info.id}: {e}")
            resource_info.state = ResourceState.ERROR
            return None
    
    # Resource loaders
    async def _load_texture(self, path: str) -> Optional[Any]:
        """Load a texture resource."""
        if not IMAGING_AVAILABLE:
            self.logger.warning("PIL not available, cannot load textures")
            return None
        
        try:
            image = Image.open(path)
            # Convert to numpy array for easier manipulation
            texture_data = np.array(image)
            return {
                'data': texture_data,
                'width': image.width,
                'height': image.height,
                'format': image.mode,
                'path': path
            }
        except Exception as e:
            self.logger.error(f"Failed to load texture {path}: {e}")
            return None
    
    async def _load_model(self, path: str) -> Optional[Any]:
        """Load a 3D model resource."""
        # Placeholder for model loading
        try:
            with open(path, 'rb') as f:
                model_data = f.read()
            return {
                'data': model_data,
                'path': path,
                'format': Path(path).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to load model {path}: {e}")
            return None
    
    async def _load_sound(self, path: str) -> Optional[Any]:
        """Load a sound resource."""
        try:
            with open(path, 'rb') as f:
                sound_data = f.read()
            return {
                'data': sound_data,
                'path': path,
                'format': Path(path).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to load sound {path}: {e}")
            return None
    
    async def _load_music(self, path: str) -> Optional[Any]:
        """Load a music resource."""
        return await self._load_sound(path)  # Same as sound for now
    
    async def _load_script(self, path: str) -> Optional[Any]:
        """Load a script resource."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            return {
                'content': script_content,
                'path': path,
                'language': Path(path).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to load script {path}: {e}")
            return None
    
    async def _load_data(self, path: str) -> Optional[Any]:
        """Load a data resource."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = f.read()
            return {
                'data': data,
                'path': path,
                'format': Path(path).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to load data {path}: {e}")
            return None
    
    async def _load_shader(self, path: str) -> Optional[Any]:
        """Load a shader resource."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                shader_code = f.read()
            return {
                'code': shader_code,
                'path': path,
                'type': Path(path).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to load shader {path}: {e}")
            return None
    
    async def _load_font(self, path: str) -> Optional[Any]:
        """Load a font resource."""
        try:
            with open(path, 'rb') as f:
                font_data = f.read()
            return {
                'data': font_data,
                'path': path,
                'format': Path(path).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to load font {path}: {e}")
            return None
    
    async def _load_animation(self, path: str) -> Optional[Any]:
        """Load an animation resource."""
        return await self._load_data(path)  # Treat as data for now
    
    async def _load_particle(self, path: str) -> Optional[Any]:
        """Load a particle system resource."""
        return await self._load_data(path)  # Treat as data for now
    
    async def _load_material(self, path: str) -> Optional[Any]:
        """Load a material resource."""
        return await self._load_data(path)  # Treat as data for now
    
    async def _load_world(self, path: str) -> Optional[Any]:
        """Load a world resource."""
        return await self._load_data(path)  # Treat as data for now
    
    async def _load_config(self, path: str) -> Optional[Any]:
        """Load a configuration resource."""
        return await self._load_data(path)  # Treat as data for now
    
    async def _fire_load_callbacks(self, resource_id: str, resource: Any):
        """Fire callbacks for resource loading."""
        if resource_id in self.loading_callbacks:
            for callback in self.loading_callbacks[resource_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(resource_id, resource)
                    else:
                        callback(resource_id, resource)
                except Exception as e:
                    self.logger.error(f"Error in load callback: {e}")
    
    def register_load_callback(self, resource_id: str, callback: Callable):
        """Register a callback for when a resource is loaded."""
        if resource_id not in self.loading_callbacks:
            self.loading_callbacks[resource_id] = []
        
        self.loading_callbacks[resource_id].append(callback)
    
    def unload_resource(self, resource_id: str):
        """Unload a resource from memory."""
        if resource_id in self.resources:
            resource_info = self.resources[resource_id]
            resource_info.reference_count = max(0, resource_info.reference_count - 1)
            
            if resource_info.reference_count == 0:
                self.resource_cache.remove(resource_id)
                resource_info.state = ResourceState.UNLOADED
                self.logger.debug(f"Unloaded resource: {resource_id}")
    
    def preload_resources(self, resource_ids: List[str]):
        """Preload a list of resources."""
        async def preload():
            tasks = []
            for resource_id in resource_ids:
                if resource_id in self.resources:
                    task = asyncio.create_task(self.load_resource(resource_id))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        asyncio.create_task(preload())
        self.logger.info(f"Started preloading {len(resource_ids)} resources")
    
    def get_resource_info(self, resource_id: str) -> Optional[ResourceInfo]:
        """Get information about a resource."""
        return self.resources.get(resource_id)
    
    def get_resources_by_type(self, resource_type: ResourceType) -> List[ResourceInfo]:
        """Get all resources of a specific type."""
        return [info for info in self.resources.values() if info.type == resource_type]
    
    def get_resources_by_tag(self, tag: str) -> List[ResourceInfo]:
        """Get all resources with a specific tag."""
        return [info for info in self.resources.values() if tag in info.tags]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        return {
            'total_resources': len(self.resources),
            'loaded_resources': len([r for r in self.resources.values() if r.state == ResourceState.LOADED]),
            'cache_stats': self.resource_cache.get_stats(),
            'load_count': self.load_count,
            'average_load_time': self.total_load_time / self.load_count if self.load_count > 0 else 0,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    async def _watch_files(self):
        """Watch for file changes and reload resources."""
        while self.watch_enabled:
            try:
                for resource_info in self.resources.values():
                    if resource_info.state == ResourceState.LOADED:
                        path = Path(resource_info.path)
                        if path.exists():
                            current_mtime = path.stat().st_mtime
                            if resource_info.path not in self.watched_files:
                                self.watched_files[resource_info.path] = current_mtime
                            elif current_mtime > self.watched_files[resource_info.path]:
                                # File changed, reload
                                self.logger.info(f"File changed, reloading: {resource_info.id}")
                                self.resource_cache.remove(resource_info.id)
                                await self.load_resource(resource_info.id)
                                self.watched_files[resource_info.path] = current_mtime
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in file watching: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    def enable_file_watching(self):
        """Enable file watching for hot reloading."""
        self.watch_enabled = True
        asyncio.create_task(self._watch_files())
        self.logger.info("File watching enabled")
    
    def disable_file_watching(self):
        """Disable file watching."""
        self.watch_enabled = False
        self.logger.info("File watching disabled")
    
    async def shutdown(self):
        """Shutdown the resource manager."""
        self.logger.info("🔄 Shutting down Resource Manager...")
        
        # Disable file watching
        self.watch_enabled = False
        
        # Cancel loading tasks
        for task in self.loading_tasks.values():
            task.cancel()
        
        # Clear cache
        self.resource_cache.clear()
        
        # Clear resources
        self.resources.clear()
        self.loading_callbacks.clear()
        
        self.logger.info("✅ Resource Manager shutdown complete")