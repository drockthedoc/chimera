"""
AI Core module for Project Chimera

This module provides the interface to the local LLM server (Ollama)
and implements various AI agents for different aspects of world generation.
"""

from .client import OllamaClient
from .agents import DialogueAgent, GeographyAgent, CharacterAgent, EventAgent

__all__ = [
    "OllamaClient",
    "DialogueAgent", 
    "GeographyAgent",
    "CharacterAgent",
    "EventAgent",
]