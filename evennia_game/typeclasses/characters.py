"""
Characters

Characters are (by default) Objects setup to be puppeted by Accounts.
They are what you "see" in game. The Character class in this module
is setup to be the "default" character type created by the default
creation commands.

Enhanced with AI-powered NPCs for Project Chimera.

"""

import sys
import os
import random
from typing import Dict, Any, Optional, List
from evennia.objects.objects import DefaultCharacter
from evennia.utils import logger
from evennia.scripts.scripts import DefaultScript

from .objects import ObjectParent

# Add chimera_core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class Character(ObjectParent, DefaultCharacter):
    """
    The Character just re-implements some of the Object's methods and hooks
    to represent a Character entity in-game.

    See mygame/typeclasses/objects.py for a list of
    properties and methods available on all Object child classes like this.

    """

    pass


class LLMCharacter(Character):
    """
    An AI-powered NPC character that uses LLMs for dialogue and behavior.
    
    This character type can engage in dynamic conversations, remember
    interactions, and exhibit personality-driven behaviors.
    """
    
    def at_object_creation(self):
        """Called when the character is first created."""
        super().at_object_creation()
        
        # AI character attributes
        self.db.ai_powered = True
        self.db.personality = "neutral"
        self.db.background = ""
        self.db.dialogue_style = "conversational"
        self.db.goals = []
        self.db.memories = []
        self.db.conversation_history = []
        
        # Behavior settings
        self.db.response_chance = 0.7  # Chance to respond to speech
        self.db.proactive_chance = 0.1  # Chance to initiate conversation
        self.db.max_memory_items = 50
        self.db.last_interaction = None
        
        # Generation metadata
        self.db.generation_prompt = ""
        self.db.generation_metadata = {}
        
        logger.log_info(f"LLM Character created: {self.key}")
    
    def at_say(self, speaker, message, **kwargs):
        """Called when someone speaks in the same room."""
        super().at_say(speaker, message, **kwargs)
        
        # Don't respond to self
        if speaker == self:
            return
        
        # Record the conversation
        self._record_conversation(speaker, message)
        
        # Maybe respond based on personality and chance
        if random.random() < self.db.response_chance:
            self._generate_response(speaker, message)
    
    def at_object_receive(self, moved_obj, source_location, **kwargs):
        """Called when someone enters the room."""
        super().at_object_receive(moved_obj, source_location, **kwargs)
        
        # Greet players occasionally
        if (hasattr(moved_obj, 'has_account') and moved_obj.has_account and 
            random.random() < self.db.proactive_chance):
            self._generate_greeting(moved_obj)
    
    def _record_conversation(self, speaker, message):
        """Record a conversation for memory and context."""
        conversation_entry = {
            "speaker": speaker.key,
            "message": message,
            "timestamp": self.db.time if hasattr(self.db, 'time') else 0,
            "location": self.location.key if self.location else "unknown"
        }
        
        # Add to conversation history
        if not hasattr(self.db, 'conversation_history'):
            self.db.conversation_history = []
        
        history = self.db.conversation_history
        history.append(conversation_entry)
        
        # Keep only recent conversations
        if len(history) > 20:
            history = history[-20:]
        
        self.db.conversation_history = history
        
        # Add to memories if significant
        if self._is_significant_interaction(speaker, message):
            self._add_memory(f"{speaker.key} said: {message}")
    
    def _is_significant_interaction(self, speaker, message):
        """Determine if an interaction is worth remembering."""
        # Simple heuristics - could be enhanced with AI analysis
        significant_keywords = ["quest", "help", "important", "remember", "tell", "story"]
        message_lower = message.lower()
        
        # Long messages are more likely to be significant
        if len(message) > 50:
            return True
        
        # Messages with significant keywords
        if any(keyword in message_lower for keyword in significant_keywords):
            return True
        
        # Direct address to this character
        if self.key.lower() in message_lower:
            return True
        
        return False
    
    def _add_memory(self, memory_text: str):
        """Add a memory to the character's long-term memory."""
        if not hasattr(self.db, 'memories'):
            self.db.memories = []
        
        memory_entry = {
            "content": memory_text,
            "timestamp": self.db.time if hasattr(self.db, 'time') else 0,
            "importance": 1.0
        }
        
        memories = self.db.memories
        memories.append(memory_entry)
        
        # Keep only the most recent memories
        max_memories = getattr(self.db, 'max_memory_items', 50)
        if len(memories) > max_memories:
            # Sort by importance and timestamp, keep the best
            memories.sort(key=lambda x: (x['importance'], x['timestamp']), reverse=True)
            memories = memories[:max_memories]
        
        self.db.memories = memories
        
        # Also store in the knowledge base for RAG
        self._store_memory_in_knowledge_base(memory_text)
    
    def _store_memory_in_knowledge_base(self, memory_text: str):
        """Store memory in the global knowledge base."""
        ai_manager = self._get_ai_manager()
        if not ai_manager and hasattr(ai_manager, 'knowledge_manager'):
            # Submit async task to store memory
            async def _store():
                async with ai_manager.knowledge_manager as km:
                    return await km.update_character_memory(
                        character_name=self.key,
                        memory_update={
                            "subject": "conversation",
                            "content": memory_text,
                            "memory_type": "character_memory",
                            "importance": 1.0,
                            "associated_entities": [],
                            "emotional_context": None
                        }
                    )
            
            ai_manager.task_manager.submit_async_task(_store(), f"store_memory_{self.key}")
    
    def _generate_response(self, speaker, message):
        """Generate an AI response to someone's speech."""
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            return
        
        # Build context for the response
        context = self._build_dialogue_context(speaker, message)
        
        # Create response prompt
        prompt = f"{speaker.key} said to {self.key}: \"{message}\"\n\nGenerate an appropriate response."
        
        # Submit async dialogue generation
        ai_manager.generate_dialogue_async(
            character_obj=self,
            prompt=prompt,
            context=context,
            callback=self._handle_dialogue_response
        )
    
    def _generate_greeting(self, entering_character):
        """Generate a greeting for someone entering the room."""
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            return
        
        context = self._build_dialogue_context(entering_character, "")
        context["interaction_type"] = "greeting"
        
        prompt = f"Generate a greeting for {entering_character.key} who just entered the room."
        
        ai_manager.generate_dialogue_async(
            character_obj=self,
            prompt=prompt,
            context=context,
            callback=self._handle_dialogue_response
        )
    
    def _build_dialogue_context(self, other_character, message):
        """Build context information for dialogue generation."""
        return {
            "character_name": self.key,
            "personality": getattr(self.db, 'personality', 'neutral'),
            "background": getattr(self.db, 'background', ''),
            "dialogue_style": getattr(self.db, 'dialogue_style', 'conversational'),
            "goals": getattr(self.db, 'goals', []),
            "location": self.location.key if self.location else "unknown",
            "other_character": other_character.key,
            "current_message": message,
            "recent_conversations": getattr(self.db, 'conversation_history', [])[-5:],
            "relevant_memories": self._get_relevant_memories(other_character, message)
        }
    
    def _get_relevant_memories(self, other_character, message, max_memories=3):
        """Get memories relevant to the current conversation."""
        memories = getattr(self.db, 'memories', [])
        
        # Simple relevance scoring based on character name and keywords
        relevant_memories = []
        
        for memory in memories:
            relevance_score = 0
            memory_content = memory['content'].lower()
            
            # Higher score if memory involves the other character
            if other_character.key.lower() in memory_content:
                relevance_score += 2
            
            # Score based on keyword overlap
            message_words = set(message.lower().split())
            memory_words = set(memory_content.split())
            overlap = len(message_words.intersection(memory_words))
            relevance_score += overlap * 0.1
            
            if relevance_score > 0:
                relevant_memories.append((memory, relevance_score))
        
        # Sort by relevance and return top memories
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem[0]['content'] for mem in relevant_memories[:max_memories]]
    
    def _handle_dialogue_response(self, result, error=None):
        """Handle the result of dialogue generation."""
        if error:
            logger.log_err(f"Dialogue generation failed for {self.key}: {error}")
            return
        
        if result and isinstance(result, dict):
            response_text = result.get("text", "")
            actions = result.get("actions", [])
            
            if response_text:
                # Speak the response
                self.execute_cmd(f'say {response_text}')
                
                # Record our own response
                self._record_conversation(self, response_text)
                
                # Execute any actions
                for action in actions:
                    self._execute_action(action)
    
    def _execute_action(self, action_description: str):
        """Execute a character action based on AI description."""
        # Simple action parsing - could be enhanced
        action_lower = action_description.lower()
        
        if "smile" in action_lower or "grin" in action_lower:
            self.location.msg_contents(f"{self.key} smiles.", exclude=[self])
        elif "nod" in action_lower:
            self.location.msg_contents(f"{self.key} nods.", exclude=[self])
        elif "frown" in action_lower:
            self.location.msg_contents(f"{self.key} frowns.", exclude=[self])
        elif "laugh" in action_lower:
            self.location.msg_contents(f"{self.key} laughs.", exclude=[self])
        else:
            # Generic action
            self.location.msg_contents(f"{self.key} {action_description}.", exclude=[self])
    
    def _get_ai_manager(self):
        """Get the AI manager script."""
        from evennia import search_script
        ai_managers = search_script("ai_manager")
        return ai_managers[0] if ai_managers else None
    
    def update_from_generation_data(self, generation_data: Dict[str, Any]):
        """
        Update character properties from AI generation data.
        
        Args:
            generation_data: Dictionary containing generated character data
        """
        # Update basic properties
        if "name" in generation_data:
            self.key = generation_data["name"]
        
        if "description" in generation_data:
            self.db.desc = generation_data["description"]
        
        # Update AI-specific properties
        if "personality" in generation_data:
            self.db.personality = generation_data["personality"]
        
        if "background" in generation_data:
            self.db.background = generation_data["background"]
        
        if "dialogue_style" in generation_data:
            self.db.dialogue_style = generation_data["dialogue_style"]
        
        if "goals" in generation_data:
            self.db.goals = generation_data["goals"]
        
        # Store generation metadata
        self.db.generation_metadata = generation_data
        
        logger.log_info(f"Updated LLM character: {self.key}")
    
    def get_ai_context(self) -> Dict[str, Any]:
        """
        Get context information about this character for AI operations.
        
        Returns:
            Dictionary containing character context
        """
        return {
            "character_name": self.key,
            "description": self.db.desc or "",
            "personality": getattr(self.db, 'personality', 'neutral'),
            "background": getattr(self.db, 'background', ''),
            "dialogue_style": getattr(self.db, 'dialogue_style', 'conversational'),
            "goals": getattr(self.db, 'goals', []),
            "recent_memories": getattr(self.db, 'memories', [])[-10:],
            "conversation_history": getattr(self.db, 'conversation_history', [])[-10:],
            "location": self.location.key if self.location else "unknown",
            "generation_metadata": getattr(self.db, 'generation_metadata', {})
        }
    
    def initiate_conversation(self, target_character, topic: Optional[str] = None):
        """
        Initiate a conversation with another character.
        
        Args:
            target_character: Character to talk to
            topic: Optional conversation topic
        """
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            return
        
        context = self._build_dialogue_context(target_character, "")
        context["interaction_type"] = "initiate_conversation"
        context["topic"] = topic
        
        if topic:
            prompt = f"Initiate a conversation with {target_character.key} about {topic}."
        else:
            prompt = f"Initiate a casual conversation with {target_character.key}."
        
        ai_manager.generate_dialogue_async(
            character_obj=self,
            prompt=prompt,
            context=context,
            callback=self._handle_dialogue_response
        )
