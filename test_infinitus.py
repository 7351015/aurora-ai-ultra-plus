#!/usr/bin/env python3
"""
🌌 INFINITUS Test Script
Simple test to verify the game systems work correctly.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_infinitus():
    """Test the Infinitus game systems."""
    print("🧪 Testing INFINITUS Game Systems...")
    
    try:
        # Test core engine imports
        print("📦 Testing core engine imports...")
        from core_engine.config import GameConfig
        from core_engine.logger import setup_logging
        print("✅ Core engine imports successful")
        
        # Test world generation imports
        print("🌍 Testing world generation imports...")
        from world_gen.world_generator import WorldGenerator
        from world_gen.noise_generator import NoiseGenerator
        from world_gen.biome_generator import BiomeGenerator
        print("✅ World generation imports successful")
        
        # Test other system imports
        print("🎮 Testing other system imports...")
        from render.graphics_engine import GraphicsEngine
        from ai_system.consciousness_engine import ConsciousnessEngine
        from player.avatar_system import AvatarSystem
        from story.narrative_engine import NarrativeEngine
        from multiplayer.network_manager import NetworkManager
        from ui.main_menu import MainMenu
        print("✅ All system imports successful")
        
        # Test basic initialization
        print("🔧 Testing basic initialization...")
        config = GameConfig()
        logger = setup_logging()
        
        # Test world generator
        world_gen = WorldGenerator(config)
        await world_gen.initialize()
        print("✅ World generator initialized")
        
        # Test world generation
        print("🌍 Testing world generation...")
        world_data = await world_gen.generate_world("TestWorld")
        print(f"✅ World generated: {world_data['metadata']['name']}")
        print(f"   Seed: {world_data['metadata']['seed']}")
        print(f"   Chunks: {len(world_data['spawn_chunks'])}")
        
        # Test other systems
        graphics = GraphicsEngine(config)
        await graphics.initialize()
        print("✅ Graphics engine initialized")
        
        consciousness = ConsciousnessEngine(config)
        await consciousness.initialize()
        print("✅ Consciousness engine initialized")
        
        avatar = AvatarSystem(config)
        await avatar.initialize()
        print("✅ Avatar system initialized")
        
        narrative = NarrativeEngine(config)
        await narrative.initialize()
        print("✅ Narrative engine initialized")
        
        network = NetworkManager(config)
        await network.initialize()
        print("✅ Network manager initialized")
        
        # Test shutdown
        print("🔄 Testing shutdown...")
        await world_gen.shutdown()
        await graphics.shutdown()
        await consciousness.shutdown()
        await avatar.shutdown()
        await narrative.shutdown()
        await network.shutdown()
        print("✅ All systems shut down successfully")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("🌌 INFINITUS is ready to launch!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_infinitus())
    if success:
        print("\n🚀 Ready to launch INFINITUS!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)