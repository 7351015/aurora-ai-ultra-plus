"""
🌌 INFINITUS Time Manager
Advanced time management system for day/night cycles, seasons, and temporal mechanics.
"""

import asyncio
import time
import math
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class Season(Enum):
    """Seasons of the year."""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class TimeOfDay(Enum):
    """Time of day periods."""
    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"
    MIDNIGHT = "midnight"
    LATE_NIGHT = "late_night"

@dataclass
class GameTime:
    """Game time representation."""
    year: int = 1
    day: int = 1
    hour: int = 6  # Start at dawn
    minute: int = 0
    second: float = 0.0
    
    def get_total_seconds(self) -> float:
        """Get total seconds since game start."""
        return (
            (self.year - 1) * 365 * 24 * 3600 +
            (self.day - 1) * 24 * 3600 +
            self.hour * 3600 +
            self.minute * 60 +
            self.second
        )
    
    def get_day_progress(self) -> float:
        """Get progress through the day (0.0 to 1.0)."""
        total_seconds = self.hour * 3600 + self.minute * 60 + self.second
        return total_seconds / (24 * 3600)
    
    def get_season_progress(self) -> float:
        """Get progress through the year (0.0 to 1.0)."""
        return (self.day - 1) / 365.0
    
    def get_season(self) -> Season:
        """Get current season."""
        progress = self.get_season_progress()
        if progress < 0.25:
            return Season.SPRING
        elif progress < 0.5:
            return Season.SUMMER
        elif progress < 0.75:
            return Season.AUTUMN
        else:
            return Season.WINTER
    
    def get_time_of_day(self) -> TimeOfDay:
        """Get current time of day."""
        if 5 <= self.hour < 7:
            return TimeOfDay.DAWN
        elif 7 <= self.hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= self.hour < 13:
            return TimeOfDay.NOON
        elif 13 <= self.hour < 18:
            return TimeOfDay.AFTERNOON
        elif 18 <= self.hour < 20:
            return TimeOfDay.DUSK
        elif 20 <= self.hour < 24:
            return TimeOfDay.NIGHT
        elif 0 <= self.hour < 1:
            return TimeOfDay.MIDNIGHT
        else:
            return TimeOfDay.LATE_NIGHT

@dataclass
class WeatherState:
    """Weather state representation."""
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # Percentage
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # Degrees
    precipitation: float = 0.0  # mm/hour
    cloud_cover: float = 0.0  # 0.0 to 1.0
    visibility: float = 1.0  # 0.0 to 1.0
    pressure: float = 1013.25  # hPa
    
class TimeManager:
    """Advanced time management system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Game time
        self.game_time = GameTime()
        self.time_scale = 1.0  # Real seconds per game second
        self.paused = False
        
        # Real time tracking
        self.real_time_start = time.time()
        self.last_update = time.time()
        
        # Day/night cycle
        self.day_length = 1200.0  # 20 minutes in real time
        self.night_length = 600.0  # 10 minutes in real time
        self.enable_day_night_cycle = True
        
        # Seasons
        self.year_length = 120.0 * 60.0  # 2 hours in real time
        self.enable_seasons = True
        
        # Weather system
        self.weather = WeatherState()
        self.enable_weather = True
        self.weather_change_rate = 0.1  # How fast weather changes
        
        # Event callbacks
        self.time_callbacks: Dict[str, List[Callable]] = {
            'hour_change': [],
            'day_change': [],
            'season_change': [],
            'weather_change': []
        }
        
        # Scheduled events
        self.scheduled_events = []
        
        # Celestial bodies
        self.sun_position = (0.0, 0.0, 0.0)
        self.moon_position = (0.0, 0.0, 0.0)
        self.moon_phase = 0.0  # 0.0 to 1.0
        
        # Time zones (for multiplayer)
        self.time_zones = {}
        
        self.logger.info("⏰ Time Manager initialized")
    
    async def initialize(self):
        """Initialize the time manager."""
        self.logger.info("🔧 Initializing Time Manager...")
        
        # Calculate initial celestial positions
        self._update_celestial_bodies()
        
        # Initialize weather
        self._initialize_weather()
        
        self.logger.info("✅ Time Manager initialization complete")
    
    async def update(self, delta_time: float):
        """Update time and related systems."""
        if self.paused:
            return
        
        try:
            # Store previous state for change detection
            prev_hour = self.game_time.hour
            prev_day = self.game_time.day
            prev_season = self.game_time.get_season()
            prev_weather = self.weather.temperature
            
            # Update game time
            self._update_game_time(delta_time)
            
            # Update celestial bodies
            self._update_celestial_bodies()
            
            # Update weather
            if self.enable_weather:
                self._update_weather(delta_time)
            
            # Check for time-based events
            await self._check_scheduled_events()
            
            # Fire callbacks for changes
            if self.game_time.hour != prev_hour:
                await self._fire_time_callbacks('hour_change')
            
            if self.game_time.day != prev_day:
                await self._fire_time_callbacks('day_change')
            
            if self.game_time.get_season() != prev_season:
                await self._fire_time_callbacks('season_change')
            
            if abs(self.weather.temperature - prev_weather) > 1.0:
                await self._fire_time_callbacks('weather_change')
            
        except Exception as e:
            self.logger.error(f"❌ Error in time manager update: {e}")
    
    def _update_game_time(self, delta_time: float):
        """Update the game time."""
        # Apply time scale
        game_delta = delta_time * self.time_scale
        
        # Update seconds
        self.game_time.second += game_delta
        
        # Handle overflow
        if self.game_time.second >= 60.0:
            minutes_to_add = int(self.game_time.second // 60.0)
            self.game_time.second = self.game_time.second % 60.0
            self.game_time.minute += minutes_to_add
        
        if self.game_time.minute >= 60:
            hours_to_add = self.game_time.minute // 60
            self.game_time.minute = self.game_time.minute % 60
            self.game_time.hour += hours_to_add
        
        if self.game_time.hour >= 24:
            days_to_add = self.game_time.hour // 24
            self.game_time.hour = self.game_time.hour % 24
            self.game_time.day += days_to_add
        
        if self.game_time.day > 365:
            years_to_add = (self.game_time.day - 1) // 365
            self.game_time.day = ((self.game_time.day - 1) % 365) + 1
            self.game_time.year += years_to_add
    
    def _update_celestial_bodies(self):
        """Update positions of sun, moon, and other celestial bodies."""
        day_progress = self.game_time.get_day_progress()
        
        # Sun position (simplified)
        sun_angle = day_progress * 2 * math.pi
        sun_elevation = math.sin(sun_angle) * 90  # Degrees above horizon
        sun_azimuth = (day_progress * 360) % 360  # Degrees from north
        
        # Convert to 3D position
        elevation_rad = math.radians(sun_elevation)
        azimuth_rad = math.radians(sun_azimuth)
        
        self.sun_position = (
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad),
            math.cos(elevation_rad) * math.cos(azimuth_rad)
        )
        
        # Moon position (simplified - opposite to sun with phase)
        moon_angle = sun_angle + math.pi
        moon_elevation = math.sin(moon_angle) * 90
        moon_azimuth = ((day_progress + 0.5) * 360) % 360
        
        moon_elevation_rad = math.radians(moon_elevation)
        moon_azimuth_rad = math.radians(moon_azimuth)
        
        self.moon_position = (
            math.cos(moon_elevation_rad) * math.sin(moon_azimuth_rad),
            math.sin(moon_elevation_rad),
            math.cos(moon_elevation_rad) * math.cos(moon_azimuth_rad)
        )
        
        # Moon phase (29.5 day cycle)
        moon_cycle_progress = (self.game_time.day % 29.5) / 29.5
        self.moon_phase = moon_cycle_progress
    
    def _initialize_weather(self):
        """Initialize weather system."""
        # Set initial weather based on season
        season = self.game_time.get_season()
        
        if season == Season.SPRING:
            self.weather.temperature = 15.0
            self.weather.humidity = 60.0
            self.weather.precipitation = 0.1
        elif season == Season.SUMMER:
            self.weather.temperature = 25.0
            self.weather.humidity = 40.0
            self.weather.precipitation = 0.0
        elif season == Season.AUTUMN:
            self.weather.temperature = 10.0
            self.weather.humidity = 70.0
            self.weather.precipitation = 0.2
        elif season == Season.WINTER:
            self.weather.temperature = 0.0
            self.weather.humidity = 80.0
            self.weather.precipitation = 0.3
    
    def _update_weather(self, delta_time: float):
        """Update weather system."""
        # Seasonal temperature base
        season = self.game_time.get_season()
        season_progress = self.game_time.get_season_progress()
        
        base_temps = {
            Season.SPRING: 15.0,
            Season.SUMMER: 25.0,
            Season.AUTUMN: 10.0,
            Season.WINTER: 0.0
        }
        
        target_temp = base_temps[season]
        
        # Daily temperature variation
        day_progress = self.game_time.get_day_progress()
        daily_variation = math.sin(day_progress * 2 * math.pi) * 10.0
        target_temp += daily_variation
        
        # Gradual weather changes
        temp_diff = target_temp - self.weather.temperature
        self.weather.temperature += temp_diff * self.weather_change_rate * delta_time
        
        # Update other weather parameters
        import random
        self.weather.wind_speed += (random.random() - 0.5) * delta_time
        self.weather.wind_speed = max(0.0, min(30.0, self.weather.wind_speed))
        
        self.weather.cloud_cover += (random.random() - 0.5) * 0.1 * delta_time
        self.weather.cloud_cover = max(0.0, min(1.0, self.weather.cloud_cover))
        
        # Precipitation based on cloud cover and humidity
        if self.weather.cloud_cover > 0.7 and self.weather.humidity > 70:
            self.weather.precipitation = min(10.0, self.weather.cloud_cover * self.weather.humidity / 10.0)
        else:
            self.weather.precipitation = max(0.0, self.weather.precipitation - delta_time)
    
    async def _check_scheduled_events(self):
        """Check and execute scheduled events."""
        current_time = self.game_time.get_total_seconds()
        
        events_to_remove = []
        for event in self.scheduled_events:
            if current_time >= event['time']:
                try:
                    await event['callback']()
                    events_to_remove.append(event)
                except Exception as e:
                    self.logger.error(f"Error executing scheduled event: {e}")
        
        # Remove executed events
        for event in events_to_remove:
            self.scheduled_events.remove(event)
    
    async def _fire_time_callbacks(self, event_type: str):
        """Fire time-based callbacks."""
        if event_type in self.time_callbacks:
            for callback in self.time_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self.game_time, self.weather)
                    else:
                        callback(self.game_time, self.weather)
                except Exception as e:
                    self.logger.error(f"Error in time callback: {e}")
    
    def register_time_callback(self, event_type: str, callback: Callable):
        """Register a callback for time events."""
        if event_type not in self.time_callbacks:
            self.time_callbacks[event_type] = []
        
        self.time_callbacks[event_type].append(callback)
        self.logger.debug(f"Registered time callback for {event_type}")
    
    def schedule_event(self, delay_seconds: float, callback: Callable):
        """Schedule an event to happen after a delay."""
        execution_time = self.game_time.get_total_seconds() + delay_seconds
        
        self.scheduled_events.append({
            'time': execution_time,
            'callback': callback
        })
        
        self.logger.debug(f"Scheduled event for {delay_seconds} seconds from now")
    
    def schedule_event_at_time(self, target_time: GameTime, callback: Callable):
        """Schedule an event to happen at a specific time."""
        execution_time = target_time.get_total_seconds()
        
        self.scheduled_events.append({
            'time': execution_time,
            'callback': callback
        })
        
        self.logger.debug(f"Scheduled event for {target_time.hour}:{target_time.minute}")
    
    def set_time_scale(self, scale: float):
        """Set the time scale (real seconds per game second)."""
        self.time_scale = max(0.1, min(100.0, scale))
        self.logger.info(f"Time scale set to {self.time_scale}x")
    
    def pause_time(self):
        """Pause time progression."""
        self.paused = True
        self.logger.info("Time paused")
    
    def resume_time(self):
        """Resume time progression."""
        self.paused = False
        self.logger.info("Time resumed")
    
    def set_time(self, new_time: GameTime):
        """Set the game time."""
        self.game_time = new_time
        self._update_celestial_bodies()
        self.logger.info(f"Time set to Year {new_time.year}, Day {new_time.day}, {new_time.hour}:{new_time.minute:02d}")
    
    def get_light_level(self) -> float:
        """Get current light level (0.0 to 1.0)."""
        # Based on sun position and weather
        sun_elevation = math.asin(self.sun_position[1])  # Radians
        sun_light = max(0.0, math.sin(sun_elevation))
        
        # Reduce light based on cloud cover
        cloud_reduction = 1.0 - (self.weather.cloud_cover * 0.7)
        
        return sun_light * cloud_reduction
    
    def get_ambient_temperature(self) -> float:
        """Get current ambient temperature."""
        return self.weather.temperature
    
    def get_wind_vector(self) -> tuple:
        """Get current wind as a 3D vector."""
        wind_rad = math.radians(self.weather.wind_direction)
        return (
            math.cos(wind_rad) * self.weather.wind_speed,
            0.0,
            math.sin(wind_rad) * self.weather.wind_speed
        )
    
    def get_time_info(self) -> Dict[str, Any]:
        """Get comprehensive time information."""
        return {
            'game_time': {
                'year': self.game_time.year,
                'day': self.game_time.day,
                'hour': self.game_time.hour,
                'minute': self.game_time.minute,
                'second': self.game_time.second
            },
            'season': self.game_time.get_season().value,
            'time_of_day': self.game_time.get_time_of_day().value,
            'day_progress': self.game_time.get_day_progress(),
            'season_progress': self.game_time.get_season_progress(),
            'celestial': {
                'sun_position': self.sun_position,
                'moon_position': self.moon_position,
                'moon_phase': self.moon_phase,
                'light_level': self.get_light_level()
            },
            'weather': {
                'temperature': self.weather.temperature,
                'humidity': self.weather.humidity,
                'wind_speed': self.weather.wind_speed,
                'wind_direction': self.weather.wind_direction,
                'precipitation': self.weather.precipitation,
                'cloud_cover': self.weather.cloud_cover,
                'visibility': self.weather.visibility,
                'pressure': self.weather.pressure
            },
            'settings': {
                'time_scale': self.time_scale,
                'paused': self.paused,
                'day_night_cycle': self.enable_day_night_cycle,
                'seasons': self.enable_seasons,
                'weather': self.enable_weather
            }
        }
    
    async def shutdown(self):
        """Shutdown the time manager."""
        self.logger.info("🔄 Shutting down Time Manager...")
        
        # Clear callbacks and scheduled events
        self.time_callbacks.clear()
        self.scheduled_events.clear()
        
        self.logger.info("✅ Time Manager shutdown complete")