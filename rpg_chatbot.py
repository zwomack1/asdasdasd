#!/usr/bin/env python3
"""
RPG Chatbot System for HRM-Gemini AI
Creates immersive RPG experiences with persistent memory and character development
"""

import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class RPGCharacter:
    """Represents an RPG character with stats, inventory, and progression"""

    def __init__(self, name: str, character_class: str = "Adventurer"):
        self.name = name
        self.character_class = character_class
        self.level = 1
        self.experience = 0
        self.health = 100
        self.max_health = 100
        self.mana = 50
        self.max_mana = 50

        # Stats
        self.stats = {
            'strength': 10,
            'dexterity': 10,
            'constitution': 10,
            'intelligence': 10,
            'wisdom': 10,
            'charisma': 10
        }

        # Inventory and equipment
        self.inventory = []
        self.equipment = {
            'weapon': None,
            'armor': None,
            'shield': None,
            'accessory': None
        }

        # Skills and abilities
        self.skills = {}
        self.abilities = []

        # Quest and story progress
        self.completed_quests = []
        self.active_quests = []
        self.story_progress = {}

        # Relationships and reputation
        self.relationships = {}
        self.reputation = {}

        # Creation timestamp
        self.created_at = datetime.now().isoformat()
        self.last_played = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert character to dictionary for storage"""
        return {
            'name': self.name,
            'character_class': self.character_class,
            'level': self.level,
            'experience': self.experience,
            'health': self.health,
            'max_health': self.max_health,
            'mana': self.mana,
            'max_mana': self.max_mana,
            'stats': self.stats,
            'inventory': self.inventory,
            'equipment': self.equipment,
            'skills': self.skills,
            'abilities': self.abilities,
            'completed_quests': self.completed_quests,
            'active_quests': self.active_quests,
            'story_progress': self.story_progress,
            'relationships': self.relationships,
            'reputation': self.reputation,
            'created_at': self.created_at,
            'last_played': self.last_played
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RPGCharacter':
        """Create character from dictionary"""
        char = cls(data['name'], data.get('character_class', 'Adventurer'))

        # Restore all attributes
        for key, value in data.items():
            if hasattr(char, key):
                setattr(char, key, value)

        return char

    def gain_experience(self, amount: int):
        """Gain experience and potentially level up"""
        self.experience += amount
        original_level = self.level

        # Level up calculation (simple exponential)
        while self.experience >= self._exp_needed_for_level(self.level + 1):
            self.level += 1

        if self.level > original_level:
            self._level_up()
            return f"🎉 {self.name} leveled up to level {self.level}!"
        else:
            return f"📈 Gained {amount} experience points!"

    def _exp_needed_for_level(self, level: int) -> int:
        """Calculate experience needed for a level"""
        return level * level * 100  # Quadratic scaling

    def _level_up(self):
        """Handle level up effects"""
        # Increase stats
        stat_increases = random.randint(1, 3)
        stat_choices = ['strength', 'dexterity', 'constitution', 'intelligence', 'wisdom', 'charisma']

        for _ in range(stat_increases):
            stat = random.choice(stat_choices)
            self.stats[stat] += 1

        # Increase health and mana
        self.max_health += random.randint(5, 15)
        self.max_mana += random.randint(3, 8)
        self.health = self.max_health  # Full heal on level up
        self.mana = self.max_mana

    def add_item(self, item: Dict[str, Any]):
        """Add item to inventory"""
        self.inventory.append(item)

    def equip_item(self, item_name: str) -> str:
        """Equip an item from inventory"""
        for item in self.inventory:
            if item['name'].lower() == item_name.lower():
                item_type = item.get('type', 'misc')

                # Unequip current item if any
                if self.equipment.get(item_type):
                    old_item = self.equipment[item_type]
                    self.inventory.append(old_item)

                # Equip new item
                self.equipment[item_type] = item
                self.inventory.remove(item)

                return f"✅ Equipped {item['name']} as {item_type}!"

        return f"❌ Item '{item_name}' not found in inventory!"

    def get_status(self) -> str:
        """Get character status summary"""
        equipped_items = [f"{slot}: {item['name'] if item else 'None'}"
                         for slot, item in self.equipment.items()]

        return f"""
🎮 **{self.name}** - Level {self.level} {self.character_class}
⭐ Experience: {self.experience}/{self._exp_needed_for_level(self.level + 1)}

❤️ Health: {self.health}/{self.max_health}
🔵 Mana: {self.mana}/{self.max_mana}

📊 Stats:
   • Strength: {self.stats['strength']}
   • Dexterity: {self.stats['dexterity']}
   • Constitution: {self.stats['constitution']}
   • Intelligence: {self.stats['intelligence']}
   • Wisdom: {self.stats['wisdom']}
   • Charisma: {self.stats['charisma']}

🎒 Equipment:
   • {chr(10).join(equipped_items)}

🎯 Active Quests: {len(self.active_quests)}
🏆 Completed Quests: {len(self.completed_quests)}
        """.strip()

class RPGWorld:
    """Represents the RPG world with locations, NPCs, and quests"""

    def __init__(self):
        self.locations = self._generate_locations()
        self.npcs = self._generate_npcs()
        self.quests = self._generate_quests()
        self.current_location = "starting_village"

    def _generate_locations(self) -> Dict[str, Dict[str, Any]]:
        """Generate game world locations"""
        return {
            "starting_village": {
                "name": "Eldoria Village",
                "description": "A peaceful village surrounded by forests",
                "npcs": ["village_elder", "merchant", "innkeeper"],
                "connections": ["dark_forest", "mountain_path"]
            },
            "dark_forest": {
                "name": "Dark Forest",
                "description": "A mysterious forest with ancient trees",
                "npcs": ["mysterious_stranger"],
                "connections": ["starting_village", "abandoned_temple"]
            },
            "mountain_path": {
                "name": "Mountain Path",
                "description": "A steep path leading to the mountains",
                "npcs": ["mountain_hermit"],
                "connections": ["starting_village", "dragon_cave"]
            },
            "abandoned_temple": {
                "name": "Abandoned Temple",
                "description": "An ancient temple filled with mystery",
                "npcs": ["ancient_spirit"],
                "connections": ["dark_forest"]
            },
            "dragon_cave": {
                "name": "Dragon's Cave",
                "description": "A cave that legends say houses a dragon",
                "npcs": ["dragon"],
                "connections": ["mountain_path"]
            }
        }

    def _generate_npcs(self) -> Dict[str, Dict[str, Any]]:
        """Generate NPCs for the world"""
        return {
            "village_elder": {
                "name": "Elder Thorne",
                "description": "The wise elder of Eldoria Village",
                "dialogue": "Welcome, adventurer! The world needs heroes like you.",
                "quests": ["save_village", "find_artifact"]
            },
            "merchant": {
                "name": "Trader Mira",
                "description": "A traveling merchant with exotic goods",
                "dialogue": "Looking to buy or sell? I've got the finest wares!",
                "items": ["health_potion", "mana_crystal", "iron_sword"]
            },
            "mysterious_stranger": {
                "name": "Stranger",
                "description": "A hooded figure who seems to know more than they let on",
                "dialogue": "The forest holds many secrets... and dangers.",
                "quests": ["investigate_temple"]
            },
            "mountain_hermit": {
                "name": "Hermit Galen",
                "description": "A wise hermit living in the mountains",
                "dialogue": "Peace is found not in silence, but in understanding.",
                "abilities": ["teach_meditation"]
            }
        }

    def _generate_quests(self) -> Dict[str, Dict[str, Any]]:
        """Generate available quests"""
        return {
            "save_village": {
                "name": "Save the Village",
                "description": "Bandits have been attacking the village. Drive them away!",
                "objectives": ["Defeat bandit leader", "Recover stolen goods"],
                "rewards": {"experience": 500, "gold": 100, "item": "villager_sword"},
                "difficulty": "Easy"
            },
            "find_artifact": {
                "name": "Find the Lost Artifact",
                "description": "An ancient artifact has been lost in the forest.",
                "objectives": ["Search Dark Forest", "Find artifact location", "Retrieve artifact"],
                "rewards": {"experience": 1000, "gold": 200, "item": "magic_amulet"},
                "difficulty": "Medium"
            },
            "investigate_temple": {
                "name": "Investigate the Abandoned Temple",
                "description": "Strange occurrences have been reported at the old temple.",
                "objectives": ["Enter temple", "Investigate strange sounds", "Discover the source"],
                "rewards": {"experience": 800, "gold": 150, "item": "ancient_scroll"},
                "difficulty": "Medium"
            }
        }

    def get_location_info(self, location_id: str) -> Dict[str, Any]:
        """Get information about a location"""
        return self.locations.get(location_id, {})

    def get_npc_info(self, npc_id: str) -> Dict[str, Any]:
        """Get information about an NPC"""
        return self.npcs.get(npc_id, {})

    def get_quest_info(self, quest_id: str) -> Dict[str, Any]:
        """Get information about a quest"""
        return self.quests.get(quest_id, {})

    def travel_to(self, location_id: str) -> str:
        """Travel to a new location"""
        if location_id in self.locations:
            if location_id in self.locations[self.current_location]["connections"]:
                old_location = self.locations[self.current_location]["name"]
                self.current_location = location_id
                new_location = self.locations[location_id]["name"]
                return f"🏔️ Traveled from {old_location} to {new_location}"
            else:
                return "❌ You can't travel there from your current location!"
        else:
            return "❌ Location not found!"

class RPGChatbot:
    """Main RPG chatbot that integrates with HRM memory system"""

    def __init__(self, memory_system=None):
        self.memory_system = memory_system
        self.world = RPGWorld()
        self.characters = {}
        self.active_sessions = {}

        # RPG-specific commands
        self.rpg_commands = {
            'create': self._cmd_create_character,
            'status': self._cmd_character_status,
            'inventory': self._cmd_inventory,
            'equip': self._cmd_equip,
            'travel': self._cmd_travel,
            'talk': self._cmd_talk,
            'quest': self._cmd_quest,
            'explore': self._cmd_explore,
            'save': self._cmd_save_game,
            'load': self._cmd_load_game
        }

        # Load existing characters
        self._load_characters()

    def _load_characters(self):
        """Load saved characters from memory"""
        if self.memory_system:
            # Look for character data in memory
            characters_data = self.memory_system.recall_memory(
                query="rpg_character",
                memory_type="rpg_data"
            )

            for memory in characters_data:
                if 'character_data' in memory.get('metadata', {}):
                    char_data = memory['metadata']['character_data']
                    character = RPGCharacter.from_dict(char_data)
                    self.characters[character.name.lower()] = character

    def process_rpg_command(self, user_id: str, command: str, args: List[str] = None) -> str:
        """Process RPG-specific commands"""
        if not args:
            args = []

        command = command.lower()

        if command in self.rpg_commands:
            try:
                return self.rpg_commands[command](user_id, args)
            except Exception as e:
                return f"❌ RPG command error: {e}"
        else:
            # Try to handle as general RPG interaction
            return self._handle_rpg_interaction(user_id, command, args)

    def _cmd_create_character(self, user_id: str, args: List[str]) -> str:
        """Create a new RPG character"""
        if not args:
            return "❌ Usage: create <character_name> [class]"

        char_name = args[0]
        char_class = args[1] if len(args) > 1 else "Adventurer"

        if char_name.lower() in self.characters:
            return f"❌ Character '{char_name}' already exists!"

        # Create new character
        character = RPGCharacter(char_name, char_class)
        self.characters[char_name.lower()] = character

        # Save to memory
        if self.memory_system:
            self.memory_system.store_memory(
                f"Created RPG character: {char_name} ({char_class})",
                memory_type="rpg_data",
                metadata={
                    'character_data': character.to_dict(),
                    'action': 'character_creation'
                },
                importance=0.9
            )

        return f"🎉 Character '{char_name}' created as a {char_class}!\n{character.get_status()}"

    def _cmd_character_status(self, user_id: str, args: List[str]) -> str:
        """Show character status"""
        char_name = args[0] if args else self._get_user_character(user_id)

        if not char_name:
            return "❌ No character specified. Use 'status <character_name>' or create a character first."

        character = self.characters.get(char_name.lower())
        if not character:
            return f"❌ Character '{char_name}' not found!"

        return character.get_status()

    def _cmd_inventory(self, user_id: str, args: List[str]) -> str:
        """Show character inventory"""
        char_name = args[0] if args else self._get_user_character(user_id)
        character = self.characters.get(char_name.lower())

        if not character:
            return f"❌ Character '{char_name}' not found!"

        if not character.inventory:
            return f"🎒 {character.name}'s inventory is empty."

        inventory_list = "\n".join([f"• {item['name']}" for item in character.inventory])
        return f"🎒 {character.name}'s Inventory:\n{inventory_list}"

    def _cmd_equip(self, user_id: str, args: List[str]) -> str:
        """Equip an item"""
        if len(args) < 2:
            return "❌ Usage: equip <character_name> <item_name>"

        char_name = args[0]
        item_name = " ".join(args[1:])

        character = self.characters.get(char_name.lower())
        if not character:
            return f"❌ Character '{char_name}' not found!"

        return character.equip_item(item_name)

    def _cmd_travel(self, user_id: str, args: List[str]) -> str:
        """Travel to a location"""
        if not args:
            return "❌ Usage: travel <location_id>"

        location_id = args[0].lower()
        return self.world.travel_to(location_id)

    def _cmd_talk(self, user_id: str, args: List[str]) -> str:
        """Talk to an NPC"""
        if not args:
            return "❌ Usage: talk <npc_name>"

        npc_name = args[0].lower()
        npc_info = self.world.get_npc_info(npc_name)

        if not npc_info:
            return f"❌ NPC '{npc_name}' not found!"

        dialogue = npc_info.get('dialogue', "Hello there!")
        return f"💬 {npc_info['name']}: \"{dialogue}\""

    def _cmd_quest(self, user_id: str, args: List[str]) -> str:
        """Handle quest-related commands"""
        if not args:
            return "❌ Usage: quest <list|accept|complete> [quest_id]"

        action = args[0].lower()

        if action == "list":
            quests = list(self.world.quests.keys())
            quest_list = "\n".join([f"• {q} - {self.world.quests[q]['name']}" for q in quests])
            return f"📋 Available Quests:\n{quest_list}"
        elif action == "info" and len(args) > 1:
            quest_id = args[1]
            quest_info = self.world.get_quest_info(quest_id)
            if quest_info:
                return f"""
🎯 **{quest_info['name']}**
{quest_info['description']}

🎯 Objectives:
{chr(10).join(f"• {obj}" for obj in quest_info['objectives'])}

💎 Rewards:
{chr(10).join(f"• {k}: {v}" for k, v in quest_info['rewards'].items())}

⚡ Difficulty: {quest_info['difficulty']}
                """.strip()
            else:
                return f"❌ Quest '{quest_id}' not found!"
        else:
            return "❌ Unknown quest action. Use: list, info <quest_id>"

    def _cmd_explore(self, user_id: str, args: List[str]) -> str:
        """Explore current location"""
        location_info = self.world.get_location_info(self.world.current_location)

        if not location_info:
            return "❌ Current location information not available."

        npcs = location_info.get('npcs', [])
        connections = location_info.get('connections', [])

        response = f"""
🏞️ **{location_info['name']}**
{location_info['description']}

👥 NPCs here: {', '.join(npcs) if npcs else 'None'}
🗺️ Connected to: {', '.join(connections) if connections else 'Nowhere'}
        """.strip()

        # Random exploration events
        if random.random() < 0.3:  # 30% chance
            events = [
                "You find some interesting tracks on the ground.",
                "A gentle breeze carries the scent of wildflowers.",
                "You hear distant sounds that make you curious.",
                "You discover some useful herbs growing nearby.",
                "The environment seems unusually quiet today."
            ]
            response += f"\n\n🌟 {random.choice(events)}"

        return response

    def _cmd_save_game(self, user_id: str, args: List[str]) -> str:
        """Save game progress"""
        if self.memory_system:
            # Save all character data
            for char_name, character in self.characters.items():
                self.memory_system.store_memory(
                    f"RPG Save - Character: {char_name}",
                    memory_type="rpg_save",
                    metadata={
                        'character_data': character.to_dict(),
                        'world_state': {
                            'current_location': self.world.current_location
                        }
                    },
                    importance=1.0
                )

            return "💾 Game progress saved successfully!"
        else:
            return "❌ No memory system available for saving."

    def _cmd_load_game(self, user_id: str, args: List[str]) -> str:
        """Load game progress"""
        if self.memory_system:
            # This would load the most recent save
            save_data = self.memory_system.recall_memory(
                query="RPG Save",
                memory_type="rpg_save",
                limit=1
            )

            if save_data:
                return "🎮 Game progress loaded successfully!"
            else:
                return "❌ No save data found."
        else:
            return "❌ No memory system available for loading."

    def _handle_rpg_interaction(self, user_id: str, command: str, args: List[str]) -> str:
        """Handle general RPG interactions"""
        # Try to interpret as natural language RPG commands
        full_command = f"{command} {' '.join(args)}".lower()

        # Simple NLP-like responses
        if any(word in full_command for word in ['hello', 'hi', 'greetings']):
            return "👋 Greetings, adventurer! How can I help you on your journey?"
        elif any(word in full_command for word in ['help', 'assist', 'aid']):
            return "🛡️ I can help you with: creating characters, exploring, quests, combat, and more. Try 'help' for specific commands!"
        elif any(word in full_command for word in ['fight', 'battle', 'combat']):
            return "⚔️ Combat system coming soon! For now, focus on exploration and quests."
        elif any(word in full_command for word in ['magic', 'spell', 'cast']):
            return "🔮 Magic system is being developed. Your character's intelligence affects magical abilities!"
        else:
            return f"🤔 I don't understand '{full_command}'. Try using specific RPG commands like 'explore', 'quest list', or 'create <name>'."

    def _get_user_character(self, user_id: str) -> Optional[str]:
        """Get the user's active character"""
        # For now, return the first character (in a real system, this would be user-specific)
        if self.characters:
            return list(self.characters.keys())[0]
        return None

    def get_rpg_status(self) -> str:
        """Get overall RPG system status"""
        return f"""
🎮 **RPG System Status**
📊 Characters: {len(self.characters)}
🏞️ Locations: {len(self.world.locations)}
👥 NPCs: {len(self.world.npcs)}
🎯 Quests: {len(self.world.quests)}
📍 Current Location: {self.world.locations[self.world.current_location]['name']}

Active Characters:
{chr(10).join(f"• {char.name} (Lv.{char.level} {char.character_class})" for char in self.characters.values())}
        """.strip()
