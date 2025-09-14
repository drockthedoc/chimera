"""
AI Agents for Project Chimera.

This module implements specialized AI agents for different aspects of
world generation and management, each with their own expertise and prompts.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel

from .client import OllamaClient, OllamaConfig
from ..schemas.game_objects import RoomSchema, CharacterSchema, ItemSchema, WorldEventSchema
from ..schemas.ai_responses import (
    DialogueResponse, GenerationResponse, AnalysisResponse, 
    ConfidenceLevel, AgentRequest
)


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all AI agents in the Chimera system.
    
    Each agent specializes in a particular domain and provides
    structured responses based on their expertise.
    """
    
    def __init__(self, client: OllamaClient, agent_name: str):
        self.client = client
        self.agent_name = agent_name
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role and expertise."""
        pass
    
    async def process_request(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Process a request and return a structured response.
        
        Args:
            request: The agent request containing prompt and context
            
        Returns:
            Structured response dictionary
        """
        try:
            # Build the full prompt with context
            full_prompt = self._build_prompt(request)
            
            # Generate response
            response_text = await self.client.generate_text(
                prompt=full_prompt,
                system_prompt=self.system_prompt,
                temperature=request.parameters.get("temperature", 0.7),
                max_tokens=request.parameters.get("max_tokens", 1000)
            )
            
            # Parse and validate response
            return await self._parse_response(response_text, request)
            
        except Exception as e:
            logger.error(f"Agent {self.agent_name} failed to process request: {e}")
            return {
                "error": str(e),
                "agent": self.agent_name,
                "confidence": ConfidenceLevel.LOW
            }
    
    def _build_prompt(self, request: AgentRequest) -> str:
        """Build the full prompt including context and instructions."""
        prompt_parts = [request.prompt]
        
        if request.context:
            context_str = "\n".join([f"{k}: {v}" for k, v in request.context.items()])
            prompt_parts.insert(0, f"Context:\n{context_str}\n")
        
        return "\n".join(prompt_parts)
    
    @abstractmethod
    async def _parse_response(self, response_text: str, request: AgentRequest) -> Dict[str, Any]:
        """Parse the raw response text into a structured format."""
        pass


class DialogueAgent(BaseAgent):
    """
    Agent specialized in generating character dialogue and interactions.
    
    This agent understands character personalities, relationships, and
    contextual factors to generate appropriate dialogue responses.
    """
    
    def __init__(self, client: OllamaClient):
        super().__init__(client, "DialogueAgent")
    
    def _get_system_prompt(self) -> str:
        return """You are a master storyteller and dialogue writer for a fantasy MUD game. 
Your expertise is in creating authentic, engaging character dialogue that reflects:

1. Character personality and background
2. Current emotional state and relationships
3. World lore and setting consistency
4. Natural speech patterns and mannerisms

Always respond with dialogue that feels natural and advances the narrative.
Consider the character's goals, fears, and motivations in every response.
Keep responses concise but meaningful, typically 1-3 sentences unless the situation calls for more.

Format your response as natural dialogue, optionally including brief action descriptions in parentheses."""
    
    async def _parse_response(self, response_text: str, request: AgentRequest) -> Dict[str, Any]:
        """Parse dialogue response."""
        # Extract character name from context
        character_name = request.context.get("character_name", "Unknown")
        
        # Simple parsing - in a full implementation, this could be more sophisticated
        dialogue_text = response_text.strip()
        
        # Extract actions (text in parentheses)
        actions = []
        import re
        action_matches = re.findall(r'\(([^)]+)\)', dialogue_text)
        actions.extend(action_matches)
        
        # Remove actions from dialogue text
        clean_dialogue = re.sub(r'\([^)]+\)', '', dialogue_text).strip()
        
        return {
            "response_type": "dialogue",
            "character_name": character_name,
            "text": clean_dialogue,
            "actions": actions,
            "emotion": request.context.get("emotion"),
            "confidence": ConfidenceLevel.HIGH,
            "context": request.context
        }


class GeographyAgent(BaseAgent):
    """
    Agent specialized in generating rooms, locations, and geographical features.
    
    This agent understands spatial relationships, environmental consistency,
    and creates immersive location descriptions.
    """
    
    def __init__(self, client: OllamaClient):
        super().__init__(client, "GeographyAgent")
    
    def _get_system_prompt(self) -> str:
        return """You are a master world-builder specializing in geography and locations for a fantasy MUD game.
Your expertise includes:

1. Creating vivid, immersive room descriptions
2. Establishing logical spatial relationships and exits
3. Maintaining environmental consistency
4. Incorporating appropriate atmosphere and mood
5. Suggesting suitable items, NPCs, and interactive elements

When generating locations, consider:
- Climate and terrain type
- Cultural influences and inhabitants
- Historical significance
- Practical purposes and functions
- Sensory details (sights, sounds, smells)

Always ensure your creations fit logically within the existing world structure."""
    
    async def _parse_response(self, response_text: str, request: AgentRequest) -> Dict[str, Any]:
        """Parse geography/room generation response."""
        try:
            # Attempt to parse as JSON if structured generation was requested
            if request.schema_name == "RoomSchema":
                # Try to parse as structured JSON using the RoomSchema. If
                # parsing fails, fall back to a basic structure so the caller
                # still receives useful data.
                try:
                    data = json.loads(response_text)
                    room = RoomSchema(**data)
                    return {
                        "response_type": "generation",
                        "content_type": "room",
                        "generated_data": room.dict(),
                        "confidence": ConfidenceLevel.HIGH,
                        "reasoning": "Structured room data validated against schema",
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse structured room response: {e}")
                    return {
                        "response_type": "generation",
                        "content_type": "room",
                        "generated_data": {
                            "name": request.context.get("room_name", "Generated Room"),
                            "description": response_text.strip(),
                            "exits": [],
                            "tags": ["generated"],
                            "atmosphere": request.context.get("atmosphere", "neutral"),
                        },
                        "confidence": ConfidenceLevel.MEDIUM,
                        "reasoning": "Basic room generation from text description",
                    }
            else:
                return {
                    "response_type": "generation",
                    "content_type": "location_description",
                    "generated_data": {"description": response_text.strip()},
                    "confidence": ConfidenceLevel.HIGH
                }
                
        except Exception as e:
            logger.error(f"Failed to parse geography response: {e}")
            return {
                "response_type": "error",
                "error": str(e),
                "raw_response": response_text
            }


class CharacterAgent(BaseAgent):
    """
    Agent specialized in generating NPCs, their personalities, and behaviors.
    
    This agent creates compelling characters with consistent personalities,
    backgrounds, and motivations that fit within the world.
    """
    
    def __init__(self, client: OllamaClient):
        super().__init__(client, "CharacterAgent")
    
    def _get_system_prompt(self) -> str:
        return """You are a master character creator for a fantasy MUD game, specializing in NPCs.
Your expertise includes:

1. Developing unique, memorable personalities
2. Creating consistent character backgrounds and motivations
3. Establishing relationships and social dynamics
4. Defining character goals, fears, and quirks
5. Ensuring characters fit naturally within the world setting

When creating characters, consider:
- Their role in the community/world
- Personal history and formative experiences
- Speech patterns and mannerisms
- Skills, abilities, and knowledge
- Relationships with other characters
- Internal conflicts and growth potential

Create characters that feel alive and have depth beyond their immediate function."""
    
    async def _parse_response(self, response_text: str, request: AgentRequest) -> Dict[str, Any]:
        """Parse character generation response."""
        try:
            if request.schema_name == "CharacterSchema":
                # Attempt structured JSON parsing and validation using the
                # CharacterSchema. If parsing fails, return a basic structure.
                try:
                    data = json.loads(response_text)
                    character = CharacterSchema(**data)
                    return {
                        "response_type": "generation",
                        "content_type": "character",
                        "generated_data": character.dict(),
                        "confidence": ConfidenceLevel.HIGH,
                        "reasoning": "Structured character data validated against schema",
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse structured character response: {e}")
                    return {
                        "response_type": "generation",
                        "content_type": "character",
                        "generated_data": {
                            "name": request.context.get("character_name", "Generated Character"),
                            "description": response_text.strip(),
                            "personality": "neutral",
                            "background": response_text.strip(),
                            "dialogue_style": "conversational",
                        },
                        "confidence": ConfidenceLevel.MEDIUM,
                        "reasoning": "Basic character generation from text description",
                    }
            else:
                return {
                    "response_type": "generation",
                    "content_type": "character_description",
                    "generated_data": {"description": response_text.strip()},
                    "confidence": ConfidenceLevel.HIGH
                }
                
        except Exception as e:
            logger.error(f"Failed to parse character response: {e}")
            return {
                "response_type": "error",
                "error": str(e),
                "raw_response": response_text
            }


class EventAgent(BaseAgent):
    """
    Agent specialized in generating world events, quests, and dynamic content.
    
    This agent creates engaging events that can drive narrative and
    provide dynamic content for players to discover and interact with.
    """
    
    def __init__(self, client: OllamaClient):
        super().__init__(client, "EventAgent")
    
    def _get_system_prompt(self) -> str:
        return """You are a master storyteller and event designer for a fantasy MUD game.
Your expertise includes:

1. Creating engaging world events and storylines
2. Designing meaningful quests and objectives
3. Establishing cause-and-effect relationships
4. Balancing challenge and reward
5. Maintaining narrative consistency and pacing

When creating events, consider:
- Player agency and meaningful choices
- World state and ongoing storylines
- Appropriate difficulty and scope
- Long-term consequences and implications
- Opportunities for character development
- Integration with existing world elements

Create events that feel organic to the world and provide compelling gameplay experiences."""
    
    async def _parse_response(self, response_text: str, request: AgentRequest) -> Dict[str, Any]:
        """Parse event generation response."""
        try:
            if request.schema_name == "WorldEventSchema":
                # Parse and validate the response using WorldEventSchema. If
                # parsing fails, provide a minimal fallback structure.
                try:
                    data = json.loads(response_text)
                    event = WorldEventSchema(**data)
                    return {
                        "response_type": "generation",
                        "content_type": "world_event",
                        "generated_data": event.dict(),
                        "confidence": ConfidenceLevel.HIGH,
                        "reasoning": "Structured event data validated against schema",
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse structured event response: {e}")
                    return {
                        "response_type": "generation",
                        "content_type": "world_event",
                        "generated_data": {
                            "name": request.context.get("event_name", "Generated Event"),
                            "description": response_text.strip(),
                            "trigger_conditions": [],
                            "effects": {},
                            "repeatable": False,
                            "global_event": False,
                        },
                        "confidence": ConfidenceLevel.MEDIUM,
                        "reasoning": "Basic event generation from text description",
                    }
            else:
                return {
                    "response_type": "generation",
                    "content_type": "event_description",
                    "generated_data": {"description": response_text.strip()},
                    "confidence": ConfidenceLevel.HIGH
                }
                
        except Exception as e:
            logger.error(f"Failed to parse event response: {e}")
            return {
                "response_type": "error",
                "error": str(e),
                "raw_response": response_text
            }


class AgentManager:
    """
    Manager class for coordinating multiple AI agents.
    
    This class provides a unified interface for invoking different agents
    and can coordinate multi-agent responses when needed.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
        self.agents = {
            "dialogue": DialogueAgent(client),
            "geography": GeographyAgent(client),
            "character": CharacterAgent(client),
            "event": EventAgent(client)
        }
    
    async def invoke_agent(self, agent_type: str, request: AgentRequest) -> Dict[str, Any]:
        """
        Invoke a specific agent with a request.
        
        Args:
            agent_type: Type of agent to invoke
            request: The request to process
            
        Returns:
            Agent response
        """
        if agent_type not in self.agents:
            return {
                "error": f"Unknown agent type: {agent_type}",
                "available_agents": list(self.agents.keys())
            }
        
        agent = self.agents[agent_type]
        return await agent.process_request(request)
    
    async def multi_agent_consultation(
        self, 
        primary_agent: str, 
        request: AgentRequest,
        consulting_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get input from multiple agents on a request.
        
        Args:
            primary_agent: The primary agent to handle the request
            request: The request to process
            consulting_agents: Additional agents to consult
            
        Returns:
            Multi-agent response with consensus information
        """
        consulting_agents = consulting_agents or []

        # Get primary agent response
        primary_response = await self.invoke_agent(primary_agent, request)

        # Collect input from consulting agents
        consultation_results = {}
        for agent_name in consulting_agents:
            if agent_name == primary_agent or agent_name not in self.agents:
                continue
            consultation_results[agent_name] = await self.invoke_agent(
                agent_name, request
            )

        if consultation_results:
            primary_response["consultations"] = consultation_results

        return primary_response


# Convenience functions for common agent operations
async def invoke_dialogue_agent(
    character_name: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """Convenience function to invoke the dialogue agent."""
    if client is None:
        client = OllamaClient()
    
    agent = DialogueAgent(client)
    request = AgentRequest(
        agent_type="dialogue",
        prompt=prompt,
        context=context or {"character_name": character_name}
    )
    
    return await agent.process_request(request)


async def invoke_geography_agent(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    schema_name: Optional[str] = None,
    client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """Convenience function to invoke the geography agent."""
    if client is None:
        client = OllamaClient()
    
    agent = GeographyAgent(client)
    request = AgentRequest(
        agent_type="geography",
        prompt=prompt,
        context=context or {},
        schema_name=schema_name
    )
    
    return await agent.process_request(request)