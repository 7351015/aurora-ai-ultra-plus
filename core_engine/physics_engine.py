"""
🌌 INFINITUS Physics Engine
Advanced physics simulation for the ultimate sandbox survival crafting game.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import time
import math

# Physics simulation libraries
try:
    import pybullet as p
    import pymunk
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False

from .config import GameConfig
from .event_system import GameEvent, EventPriority

@dataclass
class PhysicsObject:
    """Physics object representation."""
    id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0, 0, 0)
    angular_velocity: Tuple[float, float, float] = (0, 0, 0)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3
    shape: str = "box"  # box, sphere, cylinder, mesh
    size: Tuple[float, float, float] = (1, 1, 1)
    static: bool = False
    collision_group: int = 1
    collision_mask: int = 0xFFFFFFFF
    material: str = "default"
    temperature: float = 20.0  # Celsius
    
@dataclass
class PhysicsWorld:
    """Physics world configuration."""
    gravity: Tuple[float, float, float] = (0, -9.81, 0)
    air_density: float = 1.225  # kg/m³
    wind_velocity: Tuple[float, float, float] = (0, 0, 0)
    time_scale: float = 1.0
    enable_collisions: bool = True
    enable_gravity: bool = True
    enable_wind: bool = True
    enable_temperature: bool = True
    enable_fluid_dynamics: bool = True
    
class PhysicsEngine:
    """Advanced physics engine with comprehensive simulation capabilities."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Physics world
        self.world = PhysicsWorld()
        self.objects: Dict[str, PhysicsObject] = {}
        
        # Simulation state
        self.simulation_running = False
        self.time_step = 1.0 / 60.0  # 60 FPS physics
        self.accumulated_time = 0.0
        
        # Performance tracking
        self.physics_time = 0.0
        self.collision_count = 0
        self.object_count = 0
        
        # Collision detection
        self.collision_callbacks = []
        self.collision_history = []
        
        # Advanced features
        self.enable_advanced_physics = True
        self.enable_fluid_simulation = True
        self.enable_destruction = True
        self.enable_particle_physics = True
        
        # Spatial partitioning for optimization
        self.spatial_grid = {}
        self.grid_size = 10.0
        
        # Physics engine backends
        self.bullet_world = None
        self.pymunk_space = None
        
        self.logger.info("🔬 Physics Engine initialized")
    
    async def initialize(self):
        """Initialize the physics engine."""
        try:
            self.logger.info("🔧 Initializing Physics Engine...")
            
            if not PHYSICS_AVAILABLE:
                self.logger.warning("⚠️ Physics libraries not available, using simplified physics")
                self.enable_advanced_physics = False
            else:
                # Initialize Bullet Physics for 3D simulation
                self.bullet_world = p.connect(p.DIRECT)
                p.setGravity(0, self.world.gravity[1], 0)
                p.setRealTimeSimulation(0)
                
                # Initialize Pymunk for 2D physics (UI, particles, etc.)
                self.pymunk_space = pymunk.Space()
                self.pymunk_space.gravity = (0, -981)  # Pymunk uses different scale
                
                self.logger.info("✅ Advanced physics backends initialized")
            
            self.simulation_running = True
            self.logger.info("✅ Physics Engine initialization complete")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Physics Engine: {e}")
            raise
    
    async def update(self, delta_time: float):
        """Update physics simulation."""
        if not self.simulation_running:
            return
        
        start_time = time.time()
        
        try:
            # Accumulate time for fixed timestep
            self.accumulated_time += delta_time
            
            # Run physics steps
            steps = 0
            while self.accumulated_time >= self.time_step and steps < 10:  # Max 10 steps per frame
                await self._physics_step(self.time_step)
                self.accumulated_time -= self.time_step
                steps += 1
            
            # Update spatial partitioning
            self._update_spatial_grid()
            
            # Update performance metrics
            self.physics_time = time.time() - start_time
            self.object_count = len(self.objects)
            
        except Exception as e:
            self.logger.error(f"❌ Error in physics update: {e}")
    
    async def _physics_step(self, dt: float):
        """Perform a single physics simulation step."""
        if self.enable_advanced_physics and self.bullet_world is not None:
            # Step Bullet Physics
            p.stepSimulation()
            
            # Update object positions from Bullet
            for obj_id, obj in self.objects.items():
                if not obj.static:
                    # Get updated position from Bullet
                    # This would be implemented with actual Bullet object IDs
                    pass
        else:
            # Simple physics simulation
            await self._simple_physics_step(dt)
        
        # Check for collisions
        await self._check_collisions()
        
        # Apply environmental effects
        await self._apply_environmental_effects(dt)
    
    async def _simple_physics_step(self, dt: float):
        """Simple physics simulation for when advanced physics is not available."""
        for obj_id, obj in self.objects.items():
            if obj.static:
                continue
            
            # Apply gravity
            if self.world.enable_gravity:
                gravity_force = (0, self.world.gravity[1] * obj.mass, 0)
                obj.velocity = (
                    obj.velocity[0] + gravity_force[0] * dt,
                    obj.velocity[1] + gravity_force[1] * dt,
                    obj.velocity[2] + gravity_force[2] * dt
                )
            
            # Apply wind
            if self.world.enable_wind:
                wind_force = self._calculate_wind_force(obj)
                obj.velocity = (
                    obj.velocity[0] + wind_force[0] * dt,
                    obj.velocity[1] + wind_force[1] * dt,
                    obj.velocity[2] + wind_force[2] * dt
                )
            
            # Apply friction
            friction_factor = 1.0 - (obj.friction * dt)
            obj.velocity = (
                obj.velocity[0] * friction_factor,
                obj.velocity[1] * friction_factor,
                obj.velocity[2] * friction_factor
            )
            
            # Update position
            obj.position = (
                obj.position[0] + obj.velocity[0] * dt,
                obj.position[1] + obj.velocity[1] * dt,
                obj.position[2] + obj.velocity[2] * dt
            )
    
    def _calculate_wind_force(self, obj: PhysicsObject) -> Tuple[float, float, float]:
        """Calculate wind force on an object."""
        # Simplified wind calculation
        relative_velocity = (
            self.world.wind_velocity[0] - obj.velocity[0],
            self.world.wind_velocity[1] - obj.velocity[1],
            self.world.wind_velocity[2] - obj.velocity[2]
        )
        
        # Calculate drag force
        drag_coefficient = 0.47  # Sphere approximation
        area = math.pi * (obj.size[0] / 2) ** 2  # Cross-sectional area
        
        force_magnitude = 0.5 * self.world.air_density * drag_coefficient * area
        
        return (
            relative_velocity[0] * force_magnitude,
            relative_velocity[1] * force_magnitude,
            relative_velocity[2] * force_magnitude
        )
    
    async def _check_collisions(self):
        """Check for collisions between objects."""
        if not self.world.enable_collisions:
            return
        
        collisions = []
        
        # Simple collision detection (would be replaced with spatial partitioning)
        objects_list = list(self.objects.values())
        for i, obj1 in enumerate(objects_list):
            for obj2 in objects_list[i+1:]:
                if self._objects_colliding(obj1, obj2):
                    collision_data = {
                        "object1": obj1.id,
                        "object2": obj2.id,
                        "position": self._get_collision_point(obj1, obj2),
                        "normal": self._get_collision_normal(obj1, obj2),
                        "impulse": self._calculate_collision_impulse(obj1, obj2)
                    }
                    collisions.append(collision_data)
                    
                    # Resolve collision
                    await self._resolve_collision(obj1, obj2, collision_data)
        
        # Fire collision events
        for collision in collisions:
            await self._fire_collision_event(collision)
        
        self.collision_count = len(collisions)
    
    def _objects_colliding(self, obj1: PhysicsObject, obj2: PhysicsObject) -> bool:
        """Check if two objects are colliding."""
        # Simple bounding box collision detection
        dx = abs(obj1.position[0] - obj2.position[0])
        dy = abs(obj1.position[1] - obj2.position[1])
        dz = abs(obj1.position[2] - obj2.position[2])
        
        return (dx < (obj1.size[0] + obj2.size[0]) / 2 and
                dy < (obj1.size[1] + obj2.size[1]) / 2 and
                dz < (obj1.size[2] + obj2.size[2]) / 2)
    
    def _get_collision_point(self, obj1: PhysicsObject, obj2: PhysicsObject) -> Tuple[float, float, float]:
        """Get the collision point between two objects."""
        return (
            (obj1.position[0] + obj2.position[0]) / 2,
            (obj1.position[1] + obj2.position[1]) / 2,
            (obj1.position[2] + obj2.position[2]) / 2
        )
    
    def _get_collision_normal(self, obj1: PhysicsObject, obj2: PhysicsObject) -> Tuple[float, float, float]:
        """Get the collision normal vector."""
        dx = obj2.position[0] - obj1.position[0]
        dy = obj2.position[1] - obj1.position[1]
        dz = obj2.position[2] - obj1.position[2]
        
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            return (dx/length, dy/length, dz/length)
        return (0, 1, 0)
    
    def _calculate_collision_impulse(self, obj1: PhysicsObject, obj2: PhysicsObject) -> float:
        """Calculate collision impulse."""
        # Simplified impulse calculation
        relative_velocity = (
            obj1.velocity[0] - obj2.velocity[0],
            obj1.velocity[1] - obj2.velocity[1],
            obj1.velocity[2] - obj2.velocity[2]
        )
        
        velocity_magnitude = math.sqrt(
            relative_velocity[0]**2 + relative_velocity[1]**2 + relative_velocity[2]**2
        )
        
        return velocity_magnitude * (obj1.mass + obj2.mass) / 2
    
    async def _resolve_collision(self, obj1: PhysicsObject, obj2: PhysicsObject, collision_data: Dict):
        """Resolve collision between two objects."""
        if obj1.static and obj2.static:
            return
        
        # Calculate collision response
        normal = collision_data["normal"]
        impulse = collision_data["impulse"]
        
        # Apply impulse to objects
        if not obj1.static:
            impulse_factor = impulse / obj1.mass
            obj1.velocity = (
                obj1.velocity[0] - normal[0] * impulse_factor,
                obj1.velocity[1] - normal[1] * impulse_factor,
                obj1.velocity[2] - normal[2] * impulse_factor
            )
        
        if not obj2.static:
            impulse_factor = impulse / obj2.mass
            obj2.velocity = (
                obj2.velocity[0] + normal[0] * impulse_factor,
                obj2.velocity[1] + normal[1] * impulse_factor,
                obj2.velocity[2] + normal[2] * impulse_factor
            )
        
        # Separate objects to prevent overlap
        separation_distance = 0.01
        if not obj1.static:
            obj1.position = (
                obj1.position[0] - normal[0] * separation_distance,
                obj1.position[1] - normal[1] * separation_distance,
                obj1.position[2] - normal[2] * separation_distance
            )
        
        if not obj2.static:
            obj2.position = (
                obj2.position[0] + normal[0] * separation_distance,
                obj2.position[1] + normal[1] * separation_distance,
                obj2.position[2] + normal[2] * separation_distance
            )
    
    async def _fire_collision_event(self, collision_data: Dict):
        """Fire a collision event."""
        # This would fire an event through the event system
        pass
    
    async def _apply_environmental_effects(self, dt: float):
        """Apply environmental effects like temperature, pressure, etc."""
        if not self.world.enable_temperature:
            return
        
        # Apply temperature effects
        for obj in self.objects.values():
            # Simple temperature simulation
            ambient_temp = 20.0  # Celsius
            temp_diff = ambient_temp - obj.temperature
            obj.temperature += temp_diff * 0.01 * dt  # Heat transfer
    
    def _update_spatial_grid(self):
        """Update spatial partitioning grid for optimization."""
        self.spatial_grid.clear()
        
        for obj in self.objects.values():
            grid_x = int(obj.position[0] / self.grid_size)
            grid_y = int(obj.position[1] / self.grid_size)
            grid_z = int(obj.position[2] / self.grid_size)
            
            grid_key = (grid_x, grid_y, grid_z)
            if grid_key not in self.spatial_grid:
                self.spatial_grid[grid_key] = []
            
            self.spatial_grid[grid_key].append(obj.id)
    
    def add_object(self, obj: PhysicsObject) -> bool:
        """Add a physics object to the simulation."""
        try:
            if obj.id in self.objects:
                self.logger.warning(f"Physics object {obj.id} already exists")
                return False
            
            self.objects[obj.id] = obj
            
            # Add to advanced physics engine if available
            if self.enable_advanced_physics and self.bullet_world is not None:
                # Create Bullet physics body
                # This would be implemented with actual Bullet API calls
                pass
            
            self.logger.debug(f"Added physics object: {obj.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add physics object {obj.id}: {e}")
            return False
    
    def remove_object(self, obj_id: str) -> bool:
        """Remove a physics object from the simulation."""
        try:
            if obj_id not in self.objects:
                self.logger.warning(f"Physics object {obj_id} not found")
                return False
            
            del self.objects[obj_id]
            
            # Remove from advanced physics engine if available
            if self.enable_advanced_physics and self.bullet_world is not None:
                # Remove Bullet physics body
                # This would be implemented with actual Bullet API calls
                pass
            
            self.logger.debug(f"Removed physics object: {obj_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove physics object {obj_id}: {e}")
            return False
    
    def get_object(self, obj_id: str) -> Optional[PhysicsObject]:
        """Get a physics object by ID."""
        return self.objects.get(obj_id)
    
    def apply_force(self, obj_id: str, force: Tuple[float, float, float]) -> bool:
        """Apply a force to a physics object."""
        obj = self.objects.get(obj_id)
        if not obj or obj.static:
            return False
        
        # Apply force (F = ma, so a = F/m)
        acceleration = (
            force[0] / obj.mass,
            force[1] / obj.mass,
            force[2] / obj.mass
        )
        
        # Add to velocity
        obj.velocity = (
            obj.velocity[0] + acceleration[0] * self.time_step,
            obj.velocity[1] + acceleration[1] * self.time_step,
            obj.velocity[2] + acceleration[2] * self.time_step
        )
        
        return True
    
    def apply_impulse(self, obj_id: str, impulse: Tuple[float, float, float]) -> bool:
        """Apply an impulse to a physics object."""
        obj = self.objects.get(obj_id)
        if not obj or obj.static:
            return False
        
        # Apply impulse (J = mv, so v = J/m)
        velocity_change = (
            impulse[0] / obj.mass,
            impulse[1] / obj.mass,
            impulse[2] / obj.mass
        )
        
        obj.velocity = (
            obj.velocity[0] + velocity_change[0],
            obj.velocity[1] + velocity_change[1],
            obj.velocity[2] + velocity_change[2]
        )
        
        return True
    
    async def create_explosion(self, position: Tuple[float, float, float], 
                              force: float, radius: float):
        """Create an explosion effect."""
        self.logger.info(f"💥 Creating explosion at {position} with force {force}")
        
        # Find objects within explosion radius
        affected_objects = []
        for obj in self.objects.values():
            distance = math.sqrt(
                (obj.position[0] - position[0])**2 +
                (obj.position[1] - position[1])**2 +
                (obj.position[2] - position[2])**2
            )
            
            if distance <= radius:
                affected_objects.append((obj, distance))
        
        # Apply explosion forces
        for obj, distance in affected_objects:
            if obj.static:
                continue
            
            # Calculate force based on distance (inverse square law)
            if distance > 0:
                force_magnitude = force / (distance ** 2)
                
                # Direction from explosion center to object
                direction = (
                    (obj.position[0] - position[0]) / distance,
                    (obj.position[1] - position[1]) / distance,
                    (obj.position[2] - position[2]) / distance
                )
                
                # Apply force
                explosion_force = (
                    direction[0] * force_magnitude,
                    direction[1] * force_magnitude,
                    direction[2] * force_magnitude
                )
                
                self.apply_force(obj.id, explosion_force)
    
    async def load_world(self, world_data: Dict[str, Any]):
        """Load physics world from world data."""
        try:
            self.logger.info("🌍 Loading physics world...")
            
            # Load world settings
            if 'physics_settings' in world_data:
                settings = world_data['physics_settings']
                self.world.gravity = tuple(settings.get('gravity', self.world.gravity))
                self.world.air_density = settings.get('air_density', self.world.air_density)
                self.world.wind_velocity = tuple(settings.get('wind_velocity', self.world.wind_velocity))
            
            # Load physics objects
            if 'physics_objects' in world_data:
                for obj_data in world_data['physics_objects']:
                    obj = PhysicsObject(**obj_data)
                    self.add_object(obj)
            
            self.logger.info("✅ Physics world loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load physics world: {e}")
            raise
    
    async def get_world_state(self) -> Dict[str, Any]:
        """Get current physics world state."""
        return {
            'physics_settings': {
                'gravity': self.world.gravity,
                'air_density': self.world.air_density,
                'wind_velocity': self.world.wind_velocity,
                'time_scale': self.world.time_scale
            },
            'physics_objects': [
                {
                    'id': obj.id,
                    'position': obj.position,
                    'velocity': obj.velocity,
                    'angular_velocity': obj.angular_velocity,
                    'mass': obj.mass,
                    'friction': obj.friction,
                    'restitution': obj.restitution,
                    'shape': obj.shape,
                    'size': obj.size,
                    'static': obj.static,
                    'material': obj.material,
                    'temperature': obj.temperature
                }
                for obj in self.objects.values()
            ],
            'statistics': {
                'object_count': self.object_count,
                'collision_count': self.collision_count,
                'physics_time': self.physics_time
            }
        }
    
    async def shutdown(self):
        """Shutdown the physics engine."""
        try:
            self.logger.info("🔄 Shutting down Physics Engine...")
            
            self.simulation_running = False
            
            # Cleanup physics engines
            if self.bullet_world is not None:
                p.disconnect(self.bullet_world)
                self.bullet_world = None
            
            if self.pymunk_space is not None:
                self.pymunk_space = None
            
            # Clear objects
            self.objects.clear()
            self.spatial_grid.clear()
            
            self.logger.info("✅ Physics Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Physics Engine shutdown: {e}")
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get physics engine performance information."""
        return {
            "physics_time": self.physics_time,
            "object_count": self.object_count,
            "collision_count": self.collision_count,
            "simulation_running": self.simulation_running,
            "advanced_physics": self.enable_advanced_physics,
            "grid_cells": len(self.spatial_grid)
        }