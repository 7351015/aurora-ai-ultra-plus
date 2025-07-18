#!/usr/bin/env python3
"""
🌌 INFINITUS: The Ultimate Sandbox Survival Crafting God-Engine
Main entry point for the most powerful open-world sandbox game ever created.

Author: AI Assistant
License: MIT
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules
from core_engine.game_engine import GameEngine
from core_engine.config import GameConfig
from core_engine.logger import setup_logging
from ui.main_menu import MainMenu
from render.graphics_engine import GraphicsEngine
from world_gen.world_generator import WorldGenerator
from ai_system.consciousness_engine import ConsciousnessEngine
from player.avatar_system import AvatarSystem
from story.narrative_engine import NarrativeEngine
from multiplayer.network_manager import NetworkManager

# Version information
__version__ = "1.0.0-alpha"
__codename__ = "Genesis"

class InfinitusLauncher:
    """Main launcher for the Infinitus game engine."""
    
    def __init__(self):
        self.logger = None
        self.config = None
        self.game_engine = None
        self.running = False
        
    async def initialize(self):
        """Initialize all core systems."""
        print("🌌 Initializing INFINITUS God-Engine...")
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info(f"Starting Infinitus v{__version__} ({__codename__})")
        
        # Load configuration
        self.config = GameConfig()
        await self.config.load()
        
        # Initialize core systems
        self.logger.info("Initializing core systems...")
        
        # Graphics engine
        graphics_engine = GraphicsEngine(self.config)
        await graphics_engine.initialize()
        
        # World generator
        world_generator = WorldGenerator(self.config)
        await world_generator.initialize()
        
        # AI consciousness engine
        consciousness_engine = ConsciousnessEngine(self.config)
        await consciousness_engine.initialize()
        
        # Avatar system
        avatar_system = AvatarSystem(self.config)
        await avatar_system.initialize()
        
        # Narrative engine
        narrative_engine = NarrativeEngine(self.config)
        await narrative_engine.initialize()
        
        # Network manager
        network_manager = NetworkManager(self.config)
        await network_manager.initialize()
        
        # Main game engine
        self.game_engine = GameEngine(
            config=self.config,
            graphics_engine=graphics_engine,
            world_generator=world_generator,
            consciousness_engine=consciousness_engine,
            avatar_system=avatar_system,
            narrative_engine=narrative_engine,
            network_manager=network_manager
        )
        
        await self.game_engine.initialize()
        
        self.logger.info("✅ All systems initialized successfully!")
        return True
    
    async def show_main_menu(self):
        """Display the main menu."""
        main_menu = MainMenu(self.config, self.game_engine)
        choice = await main_menu.show()
        
        if choice == "new_game":
            await self.start_new_game()
        elif choice == "load_game":
            await self.load_game()
        elif choice == "multiplayer":
            await self.join_multiplayer()
        elif choice == "settings":
            await self.show_settings()
        elif choice == "exit":
            await self.shutdown()
            return False
        
        return True
    
    async def start_new_game(self):
        """Start a new single-player game."""
        self.logger.info("🎮 Starting new game...")
        
        # Generate new world
        world_name = f"World_{asyncio.get_event_loop().time():.0f}"
        await self.game_engine.create_world(world_name)
        
        # Start game loop
        self.running = True
        await self.game_loop()
    
    async def load_game(self):
        """Load an existing game."""
        self.logger.info("📂 Loading saved game...")
        # Implementation for loading saved games
        pass
    
    async def join_multiplayer(self):
        """Join or host a multiplayer game."""
        self.logger.info("🌐 Joining multiplayer...")
        # Implementation for multiplayer
        pass
    
    async def show_settings(self):
        """Show game settings."""
        self.logger.info("⚙️ Opening settings...")
        # Implementation for settings
        pass
    
    async def game_loop(self):
        """Main game loop."""
        self.logger.info("🎯 Entering main game loop...")
        
        while self.running:
            try:
                # Update all systems
                await self.game_engine.update()
                
                # Render frame
                await self.game_engine.render()
                
                # Handle events
                events = await self.game_engine.get_events()
                for event in events:
                    if event.type == "quit":
                        self.running = False
                        break
                    await self.game_engine.handle_event(event)
                
                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)
                
            except KeyboardInterrupt:
                self.logger.info("⚠️ Received interrupt signal")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"❌ Error in game loop: {e}")
                # Don't crash the game, just log the error
                continue
    
    async def shutdown(self):
        """Gracefully shutdown the game."""
        self.logger.info("🔄 Shutting down Infinitus...")
        
        if self.game_engine:
            await self.game_engine.shutdown()
        
        self.logger.info("✅ Shutdown complete. Thank you for playing Infinitus!")

async def main():
    """Main entry point."""
    try:
        # ASCII art welcome message
        print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║    🌌 INFINITUS: The Ultimate Sandbox Survival Crafting Game    ║
    ║                                                                  ║
    ║    "In Infinitus, you don't just play the game -                ║
    ║     you become the universe."                                    ║
    ║                                                                  ║
    ║    Version: {:<10} Codename: {:<20}                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
        """.format(__version__, __codename__))
        
        # Initialize and run the game
        launcher = InfinitusLauncher()
        
        if await launcher.initialize():
            # Show main menu and handle user choice
            while await launcher.show_main_menu():
                pass
        
        await launcher.shutdown()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.exception("Fatal error in main()")
        sys.exit(1)

if __name__ == "__main__":
    # Run the game
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Game interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start game: {e}")
        sys.exit(1)