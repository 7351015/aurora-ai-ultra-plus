#!/usr/bin/env python3
"""
INFINITUS - Next Generation Minecraft 2025
The Ultimate Sandbox Survival Crafting Experience
"""

import asyncio
import sys
import time
import threading
from pathlib import Path

# Core game imports
from core_engine.game_engine import GameEngine
from core_engine.config import GameConfig
from core_engine.logger import GameLogger
from render.graphics_engine import GraphicsEngine
from world_gen.world_generator import WorldGenerator
from gameplay.minecraft_engine import MinecraftEngine
from ui.main_menu import MainMenu
from multiplayer.network_manager import NetworkManager

class InfinitusGame:
    """Main game class orchestrating all systems"""
    
    def __init__(self):
        self.config = GameConfig()
        self.logger = GameLogger()
        self.running = False
        self.game_engine = None
        self.graphics_engine = None
        self.minecraft_engine = None
        self.world_generator = None
        self.network_manager = None
        self.main_menu = None
        
    async def initialize(self):
        """Initialize all game systems"""
        try:
            self.logger.info("🚀 Initializing INFINITUS - Next Gen Minecraft 2025")
            
            # Initialize core systems
            self.game_engine = GameEngine(self.config, self.logger)
            await self.game_engine.initialize()
            
            # Initialize graphics engine
            self.graphics_engine = GraphicsEngine(self.config, self.logger)
            await self.graphics_engine.initialize()
            
            # Initialize world generator
            self.world_generator = WorldGenerator(self.config, self.logger)
            await self.world_generator.initialize()
            
            # Initialize Minecraft-style gameplay engine
            self.minecraft_engine = MinecraftEngine(self.config, self.logger, self.graphics_engine)
            await self.minecraft_engine.initialize()
            
            # Initialize networking
            self.network_manager = NetworkManager(self.config, self.logger)
            await self.network_manager.initialize()
            
            # Initialize UI
            self.main_menu = MainMenu(self.config, self.logger, self.graphics_engine)
            await self.main_menu.initialize()
            
            self.logger.info("✅ All systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize game: {e}")
            return False
    
    async def run(self):
        """Main game loop"""
        self.running = True
        
        # Show splash screen
        await self.show_splash_screen()
        
        # Show main menu
        menu_choice = await self.main_menu.show()
        
        if menu_choice == "singleplayer":
            await self.start_singleplayer()
        elif menu_choice == "multiplayer":
            await self.start_multiplayer()
        elif menu_choice == "create_world":
            await self.create_world()
        elif menu_choice == "settings":
            await self.show_settings()
        elif menu_choice == "exit":
            await self.shutdown()
            return
        
        # Main game loop
        while self.running:
            try:
                # Update all systems
                await self.game_engine.update()
                await self.minecraft_engine.update()
                await self.graphics_engine.render()
                
                # Handle events
                await self.handle_events()
                
                # Sleep for frame rate control
                await asyncio.sleep(1.0 / self.config.graphics.target_fps)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"❌ Game loop error: {e}")
                break
    
    async def show_splash_screen(self):
        """Show the game splash screen"""
        splash_art = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                      ║
    ║    ██╗███╗   ██╗███████╗██╗███╗   ██╗██╗████████╗██╗   ██╗███████╗                 ║
    ║    ██║████╗  ██║██╔════╝██║████╗  ██║██║╚══██╔══╝██║   ██║██╔════╝                 ║
    ║    ██║██╔██╗ ██║█████╗  ██║██╔██╗ ██║██║   ██║   ██║   ██║███████╗                 ║
    ║    ██║██║╚██╗██║██╔══╝  ██║██║╚██╗██║██║   ██║   ██║   ██║╚════██║                 ║
    ║    ██║██║ ╚████║██║     ██║██║ ╚████║██║   ██║   ╚██████╔╝███████║                 ║
    ║    ╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ ╚══════╝                 ║
    ║                                                                                      ║
    ║                    🌟 NEXT GENERATION MINECRAFT 2025 🌟                             ║
    ║                  The Ultimate Sandbox Survival Experience                           ║
    ║                                                                                      ║
    ║    ✨ Features:                                                                      ║
    ║    • Infinite Procedural Worlds with 50+ Biomes                                     ║
    ║    • Advanced AI NPCs with Consciousness & Memory                                    ║
    ║    • Real-time Physics & Voxel Deformation                                          ║
    ║    • All Minecraft Mods Integrated (Magic, Tech, Automation)                        ║
    ║    • Dynamic Weather, Seasons & Day/Night Cycles                                     ║
    ║    • Multiplayer Universe with Persistent Worlds                                     ║
    ║    • Epic Storyline Across 7 Cosmic Orders                                          ║
    ║    • Unlimited Creativity & Building Tools                                           ║
    ║    • Ray-Traced Graphics & Advanced Lighting                                         ║
    ║    • Voice Commands & AI Assistant Integration                                       ║
    ║                                                                                      ║
    ║                        Press ENTER to begin your journey...                         ║
    ║                                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        print(splash_art)
        input()  # Wait for user input
    
    async def start_singleplayer(self):
        """Start singleplayer game"""
        self.logger.info("🎮 Starting singleplayer game")
        
        # Generate or load world
        world_name = "MyWorld"
        world = await self.world_generator.generate_world(world_name)
        
        # Initialize player
        await self.minecraft_engine.create_player(world)
        
        # Start game systems
        await self.minecraft_engine.start_game()
        
        self.logger.info("✅ Singleplayer game started successfully")
    
    async def start_multiplayer(self):
        """Start multiplayer game"""
        self.logger.info("🌐 Starting multiplayer game")
        
        # Connect to server or create server
        server_choice = await self.main_menu.show_multiplayer_menu()
        
        if server_choice == "create_server":
            await self.network_manager.create_server()
        elif server_choice == "join_server":
            server_address = await self.main_menu.get_server_address()
            await self.network_manager.connect_to_server(server_address)
        
        # Start multiplayer game
        await self.minecraft_engine.start_multiplayer_game()
        
        self.logger.info("✅ Multiplayer game started successfully")
    
    async def create_world(self):
        """Create a new world"""
        self.logger.info("🌍 Creating new world")
        
        # Get world creation parameters
        world_params = await self.main_menu.get_world_creation_params()
        
        # Generate world
        world = await self.world_generator.create_custom_world(world_params)
        
        self.logger.info(f"✅ World '{world.name}' created successfully")
    
    async def show_settings(self):
        """Show game settings"""
        await self.main_menu.show_settings()
    
    async def handle_events(self):
        """Handle game events"""
        # Process input events
        await self.minecraft_engine.handle_input()
        
        # Process network events
        await self.network_manager.process_events()
        
        # Process UI events
        await self.main_menu.handle_events()
    
    async def shutdown(self):
        """Shutdown game gracefully"""
        self.logger.info("🛑 Shutting down INFINITUS")
        self.running = False
        
        # Shutdown all systems
        if self.minecraft_engine:
            await self.minecraft_engine.shutdown()
        if self.graphics_engine:
            await self.graphics_engine.shutdown()
        if self.network_manager:
            await self.network_manager.shutdown()
        if self.game_engine:
            await self.game_engine.shutdown()
        
        self.logger.info("✅ Shutdown complete")

async def main():
    """Main entry point"""
    game = InfinitusGame()
    
    try:
        # Initialize game
        if await game.initialize():
            # Run game
            await game.run()
        else:
            print("❌ Failed to initialize game")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)
    finally:
        await game.shutdown()

if __name__ == "__main__":
    # Run the game
    asyncio.run(main())