"""
INFINITUS Weather System
Manages weather states (clear, rain, snow) and transitions.
"""
from __future__ import annotations

import random
from typing import Dict, Any

class WeatherState:
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"

class WeatherSystem:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.state = WeatherState.CLEAR
        self.intensity = 0.0  # 0..1
        self._time_to_next_change = self.rng.uniform(60.0, 180.0)  # seconds

    async def initialize(self):
        return

    async def update(self, delta_time: float):
        self._time_to_next_change -= delta_time
        # Smoothly vary intensity
        target = 0.0 if self.state == WeatherState.CLEAR else self.rng.uniform(0.4, 1.0)
        self.intensity += (target - self.intensity) * 0.02
        if self._time_to_next_change <= 0.0:
            self._time_to_next_change = self.rng.uniform(120.0, 420.0)
            # 20% chance to switch states
            if self.rng.random() < 0.2:
                self.state = self.rng.choice([WeatherState.CLEAR, WeatherState.RAIN, WeatherState.SNOW])

    def get_brightness_modifier(self) -> float:
        # Rain/snow dim the world slightly
        if self.state == WeatherState.CLEAR:
            return 1.0
        return max(0.7, 1.0 - 0.2 * self.intensity)

    def get_overlay_info(self) -> Dict[str, Any]:
        return {"state": self.state, "intensity": self.intensity}