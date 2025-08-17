"""
INFINITUS Player Stats System
Manages health, hunger, and simple regeneration/decay over time.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class PlayerStats:
    health: float = 20.0
    max_health: float = 20.0
    hunger: float = 20.0
    max_hunger: float = 20.0
    saturation: float = 5.0
    armor: float = 0.0

class StatsSystem:
    def __init__(self):
        self.regen_cooldown = 0.0

    async def initialize(self):
        return

    async def update(self, delta_time: float, stats: PlayerStats):
        # Hunger decay
        stats.hunger = max(0.0, stats.hunger - 0.003 * delta_time * 60.0)
        # Saturation decay if moving/doing actions would be handled elsewhere
        if stats.hunger <= 0.0:
            # Starvation damage
            stats.health = max(0.0, stats.health - 0.02 * delta_time * 60.0)
        else:
            # Small regen when fed and healthy
            self.regen_cooldown -= delta_time
            if self.regen_cooldown <= 0.0 and stats.health < stats.max_health and stats.hunger > 9.0:
                stats.health = min(stats.max_health, stats.health + 0.1)
                stats.hunger = max(0.0, stats.hunger - 0.2)
                self.regen_cooldown = 2.0

    def apply_damage(self, stats: PlayerStats, amount: float):
        # Armor reduces damage naively
        mitigated = max(0.0, amount - stats.armor * 0.04 * amount)
        stats.health = max(0.0, stats.health - mitigated)