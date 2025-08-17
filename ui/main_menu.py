"""
🌌 INFINITUS Main Menu
Main menu interface for the game.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

class MainMenu:
    """Main menu system."""
    
    def __init__(self, config, logger: Optional[logging.Logger] = None, graphics_engine: Optional[Any] = None):
        self.config = config
        self.graphics_engine = graphics_engine
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎮 Main Menu initialized")
    
    async def initialize(self) -> None:
        self.logger.info("🔧 Initializing Main Menu...")
        self.logger.info("✅ Main Menu initialization complete")
    
    async def show(self) -> str:
        """Show the main menu and get user choice."""
        print("\n" + "="*60)
        print("🌌 INFINITUS: The Ultimate Sandbox Survival Crafting Game")
        print("="*60)
        print("1. Singleplayer")
        print("2. Multiplayer")
        print("3. Create World")
        print("4. Settings")
        print("5. Exit")
        print("="*60)
        
        # Auto-select singleplayer in non-interactive environments
        print("🎯 Starting singleplayer automatically...")
        return "singleplayer"

    async def show_multiplayer_menu(self) -> str:
        """Return action for multiplayer menu."""
        # Default to creating a server in demo mode
        return "create_server"

    async def get_server_address(self) -> str:
        """Prompt or return default server address."""
        return "localhost:25565"

    async def get_world_creation_params(self) -> Dict[str, Any]:
        """Return default world creation parameters."""
        return {
            "world_type": "normal",
            "seed": None,
            "biome_diversity": 1.0,
        }

    async def show_settings(self) -> None:
        """Show settings UI (placeholder)."""
        print("⚙️ Settings are currently default.")

    async def handle_events(self) -> None:
        """Process UI events if any (placeholder)."""
        await asyncio.sleep(0)

    # Additional UI hooks that gameplay can call
    async def show_crafting(self, player_inventory: Dict[str, int], crafting_system) -> Optional[Dict[str, Any]]:
        """Simulate crafting selection: try crafting a pickaxe if possible."""
        # Prefer iron, then stone, then wood
        for recipe in (["iron_ingot", "stick", "stick"], ["cobblestone", "stick", "stick"], ["planks", "stick", "stick"]):
            if crafting_system.can_craft(player_inventory, recipe):
                crafting_system.craft(player_inventory, recipe)
                return {"crafted": True, "recipe": recipe}
        return None

    async def save_world(self, engine) -> bool:
        try:
            if hasattr(engine, "_do_autosave"):
                await engine._do_autosave()
                return True
        except Exception:
            return False
        return False