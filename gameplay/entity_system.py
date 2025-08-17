"""
INFINITUS Entity System
Simple entities with wander/chase AI.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Any, List

class Entity:
    def __init__(self, eid: str, x: float, y: float, z: float, kind: str = "mob"):
        self.id = eid
        self.kind = kind
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.state = "wander"  # wander or chase
        self._timer = 0.0

class EntitySystem:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.entities: Dict[str, Entity] = {}
        self._spawn_timer = 0.0

    async def initialize(self):
        return

    async def update(self, delta_time: float, player_pos: tuple[float, float, float]):
        self._spawn_timer += delta_time
        # Spawn every ~10s up to a small cap
        if self._spawn_timer >= 10.0 and len(self.entities) < 10:
            self._spawn_timer = 0.0
            eid = f"mob_{self.rng.randrange(100000)}"
            ex = player_pos[0] + self.rng.uniform(-20, 20)
            ez = player_pos[2] + self.rng.uniform(-20, 20)
            self.entities[eid] = Entity(eid, ex, player_pos[1], ez)
        # Update AI
        px, py, pz = player_pos
        for e in list(self.entities.values()):
            # Switch to chase if within 12 blocks
            dx = px - e.x
            dz = pz - e.z
            dist2 = dx*dx + dz*dz
            e.state = "chase" if dist2 < 12*12 else "wander"
            if e.state == "wander":
                e._timer -= delta_time
                if e._timer <= 0.0:
                    e.vx = self.rng.uniform(-1, 1)
                    e.vz = self.rng.uniform(-1, 1)
                    e._timer = self.rng.uniform(2.0, 5.0)
            else:
                # Move towards player slowly
                l = math.sqrt(dist2) or 1.0
                e.vx = dx / l
                e.vz = dz / l
            # Integrate
            e.x += e.vx * delta_time * 2.0
            e.z += e.vz * delta_time * 2.0

    def get_entities(self) -> List[Dict[str, Any]]:
        return [
            {"id": e.id, "kind": e.kind, "position": (e.x, e.y, e.z), "state": e.state}
            for e in self.entities.values()
        ]