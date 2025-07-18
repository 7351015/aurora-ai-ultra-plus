"""
🌌 INFINITUS Main Menu
Main menu interface for the game.
"""

import asyncio
import logging
from typing import Dict, List, Any

class MainMenu:
    """Main menu system."""
    
    def __init__(self, config, game_engine):
        self.config = config
        self.game_engine = game_engine
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎮 Main Menu initialized")
    
    async def show(self):
        """Show the main menu and get user choice."""
        print("\n" + "="*60)
        print("🌌 INFINITUS: The Ultimate Sandbox Survival Crafting Game")
        print("="*60)
        print("1. New Game")
        print("2. Load Game")
        print("3. Multiplayer")
        print("4. Settings")
        print("5. Exit")
        print("="*60)
        
        # For now, just automatically start a new game
        print("🎯 Starting new game automatically...")
        return "new_game"