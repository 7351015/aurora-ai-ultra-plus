"""
INFINITUS Crafting System
Simple shapeless crafting logic backed by dict-based inventory.
"""
from typing import Dict, List, Tuple, Optional

class CraftingSystem:
    def __init__(self):
        # Define some example shapeless recipes: tuple of ingredients -> (result, count)
        # Ingredients are item ids (strings). Inventory is a dict of item->count.
        self.recipes: Dict[Tuple[str, ...], Tuple[str, int]] = {
            tuple(sorted(["wood_log"])): ("planks", 4),
            tuple(sorted(["planks", "planks"])): ("stick", 4),
            tuple(sorted(["planks", "stick", "stick"])): ("wood_pickaxe", 1),
            tuple(sorted(["cobblestone", "stick", "stick"])): ("stone_pickaxe", 1),
            tuple(sorted(["iron_ingot", "stick", "stick"])): ("iron_pickaxe", 1),
            tuple(sorted(["iron_ore"])): ("iron_ingot", 1),
        }

    def can_craft(self, inventory: Dict[str, int], ingredients: List[str]) -> bool:
        ingredients = sorted([i for i in ingredients if i])
        key = tuple(ingredients)
        if key not in self.recipes:
            return False
        # Check availability in inventory
        need: Dict[str, int] = {}
        for i in ingredients:
            need[i] = need.get(i, 0) + 1
        for i, n in need.items():
            if inventory.get(i, 0) < n:
                return False
        return True

    def craft(self, inventory: Dict[str, int], ingredients: List[str]) -> Optional[Tuple[str, int]]:
        """Attempt to craft using provided ingredients. Returns result (item,count) or None."""
        if not self.can_craft(inventory, ingredients):
            return None
        key = tuple(sorted([i for i in ingredients if i]))
        result = self.recipes[key]
        # Consume ingredients
        need: Dict[str, int] = {}
        for i in key:
            need[i] = need.get(i, 0) + 1
        for i, n in need.items():
            inventory[i] = inventory.get(i, 0) - n
            if inventory[i] <= 0:
                inventory.pop(i, None)
        # Add result
        res_item, res_count = result
        inventory[res_item] = inventory.get(res_item, 0) + res_count
        return result