"""
Structured generation utilities for Project Chimera.

This module provides guaranteed JSON output from LLMs using schema validation
and constraint-based generation techniques.
"""

import json
import logging
from typing import Dict, Any, Type, Optional, Union
from pydantic import BaseModel, ValidationError
import asyncio
import re

from ..ai_core.client import OllamaClient
from ..schemas.game_objects import RoomSchema, CharacterSchema, ItemSchema


logger = logging.getLogger(__name__)


class JSONValidator:
    """
    Validator for ensuring JSON output conforms to Pydantic schemas.
    """
    
    @staticmethod
    def validate_json(json_str: str, schema_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Validate JSON string against a Pydantic schema.
        
        Args:
            json_str: JSON string to validate
            schema_class: Pydantic model class to validate against
            
        Returns:
            Validated data as dictionary
            
        Raises:
            ValidationError: If validation fails
            ValueError: If JSON is invalid
        """
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate against schema
            validated_instance = schema_class(**data)
            
            # Return as dictionary
            return validated_instance.dict()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except ValidationError as e:
            raise ValidationError(f"Schema validation failed: {e}")
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """
        Extract JSON from text that may contain other content.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string or None if not found
        """
        # Look for JSON blocks marked with ```json
        json_block_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Look for JSON objects in the text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        # Try to parse each potential JSON object
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        return None


class StructuredGenerator:
    """
    Generator that ensures structured JSON output from LLMs.
    
    This class uses various techniques to guarantee valid JSON output
    that conforms to specified Pydantic schemas.
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client or OllamaClient()
        self.validator = JSONValidator()
        
        # Schema examples for few-shot prompting
        self.schema_examples = {
            "RoomSchema": {
                "name": "The Ancient Library",
                "description": "A vast library filled with dusty tomes and ancient scrolls. Tall shelves stretch toward a vaulted ceiling, and the air smells of old parchment and leather bindings.",
                "exits": [
                    {"direction": "north", "description": "A wooden door leads to the main hall"},
                    {"direction": "east", "description": "An archway opens to the reading room"}
                ],
                "tags": ["library", "ancient", "scholarly"],
                "atmosphere": "quiet and scholarly",
                "items": ["ancient tome", "reading desk", "oil lamp"],
                "npcs": ["elderly librarian"]
            },
            "CharacterSchema": {
                "name": "Eldara the Wise",
                "description": "An elderly woman with silver hair and kind eyes, wearing simple robes",
                "personality": "wise",
                "background": "Once a powerful mage, now serves as the village's keeper of knowledge",
                "dialogue_style": "speaks slowly and thoughtfully, often in riddles",
                "goals": ["preserve ancient knowledge", "guide young adventurers"],
                "skills": ["ancient lore", "divination", "herb lore"],
                "inventory": ["crystal orb", "spell components", "ancient journal"]
            }
        }
    
    async def generate_structured_content(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        max_attempts: int = 3,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured content that conforms to a Pydantic schema.
        
        Args:
            prompt: Generation prompt
            schema_class: Pydantic schema class
            max_attempts: Maximum retry attempts
            temperature: Generation temperature (lower = more deterministic)
            system_prompt: Optional system prompt override
            
        Returns:
            Validated structured data
            
        Raises:
            Exception: If generation fails after all attempts
        """
        schema_name = schema_class.__name__
        
        # Build structured prompt
        structured_prompt = self._build_structured_prompt(prompt, schema_class, system_prompt)
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Generating {schema_name} (attempt {attempt + 1}/{max_attempts})")
                
                # Generate response
                response = await self.ollama_client.generate_text(
                    prompt=structured_prompt,
                    temperature=temperature,
                    max_tokens=1000
                )
                
                # Extract JSON from response
                json_str = self.validator.extract_json_from_text(response)
                if not json_str:
                    # If no JSON found, try to use the entire response
                    json_str = response.strip()
                
                # Validate against schema
                validated_data = self.validator.validate_json(json_str, schema_class)
                
                logger.info(f"Successfully generated valid {schema_name}")
                return validated_data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    logger.error(f"Failed to generate valid {schema_name} after {max_attempts} attempts")
                    raise Exception(f"Structured generation failed: {e}")
                
                # Adjust prompt for retry
                structured_prompt = self._build_retry_prompt(structured_prompt, str(e), schema_class)
        
        # This should never be reached due to the raise in the loop
        raise Exception("Unexpected error in structured generation")
    
    def _build_structured_prompt(
        self, 
        prompt: str, 
        schema_class: Type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> str:
        """Build a prompt that encourages structured JSON output."""
        schema_name = schema_class.__name__
        
        # Get schema information
        schema_info = self._get_schema_info(schema_class)
        example = self.schema_examples.get(schema_name, {})
        
        # Build the structured prompt
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        prompt_parts.extend([
            f"Generate a valid JSON object that conforms to the {schema_name} schema.",
            f"",
            f"Schema requirements:",
            schema_info,
            f"",
            f"Example {schema_name}:",
            json.dumps(example, indent=2) if example else "No example available",
            f"",
            f"Request: {prompt}",
            f"",
            f"Respond with ONLY a valid JSON object that matches the {schema_name} schema:",
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_retry_prompt(
        self, 
        original_prompt: str, 
        error_message: str, 
        schema_class: Type[BaseModel]
    ) -> str:
        """Build a retry prompt that addresses the previous error."""
        schema_name = schema_class.__name__
        
        retry_prompt = f"{original_prompt}\n\nPREVIOUS ATTEMPT FAILED with error: {error_message}\n\nPlease generate a valid JSON object that strictly follows the {schema_name} schema. Ensure all required fields are present and have the correct data types."
        
        return retry_prompt
    
    def _get_schema_info(self, schema_class: Type[BaseModel]) -> str:
        """Get human-readable schema information."""
        try:
            schema = schema_class.schema()
            
            info_parts = []
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "unknown")
                description = field_info.get("description", "")
                is_required = field_name in required
                
                requirement = "REQUIRED" if is_required else "optional"
                info_parts.append(f"- {field_name} ({field_type}, {requirement}): {description}")
            
            return "\n".join(info_parts)
            
        except Exception as e:
            logger.warning(f"Failed to extract schema info: {e}")
            return f"Schema: {schema_class.__name__}"
    
    async def generate_room(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for generating rooms."""
        return await self.generate_structured_content(
            prompt=prompt,
            schema_class=RoomSchema,
            **kwargs
        )
    
    async def generate_character(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for generating characters."""
        return await self.generate_structured_content(
            prompt=prompt,
            schema_class=CharacterSchema,
            **kwargs
        )
    
    async def generate_item(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for generating items."""
        return await self.generate_structured_content(
            prompt=prompt,
            schema_class=ItemSchema,
            **kwargs
        )


class ConstrainedGenerator:
    """
    Alternative generator using constraint-based techniques.
    
    This class would integrate with libraries like 'outlines' or 'guidance'
    to provide even stronger guarantees about output structure.
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client or OllamaClient()

        # Attempt to import an external constraint library (like `outlines`).
        # If unavailable, we'll fall back to our structured generator which
        # already performs schema validation via Pydantic.
        try:  # pragma: no cover - optional dependency
            from outlines import models, generate  # type: ignore

            self._outlines_models = models
            self._outlines_generate = generate
            logger.info("ConstrainedGenerator initialized with outlines library")
        except Exception:  # outlines is optional and may not be installed
            self._outlines_models = None
            self._outlines_generate = None
            logger.info(
                "ConstrainedGenerator initialized without external constraint library;"
                " using Pydantic validation fallback"
            )
    
    async def generate_with_constraints(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content with hard constraints on output format.
        
        This method would use libraries like 'outlines' to guarantee
        that the LLM output conforms exactly to the specified schema.
        
        Args:
            prompt: Generation prompt
            schema_class: Pydantic schema class
            **kwargs: Additional generation parameters
            
        Returns:
            Guaranteed valid structured data
        """
        # If we have an external constraint library available, try to use it.
        if self._outlines_models and self._outlines_generate:  # pragma: no cover - optional path
            try:
                schema = schema_class.schema()
                model = self._outlines_models.ollama(
                    self.ollama_client.config.default_model,
                    base_url=self.ollama_client.config.base_url,
                )
                generator = self._outlines_generate.json(model, schema)
                result = await generator(prompt, **kwargs)
                if isinstance(result, str):
                    result = json.loads(result)
                validated = schema_class(**result)
                return validated.dict()
            except Exception as e:
                logger.warning(
                    "External constraint generation failed (%s); falling back to structured generation",
                    e,
                )

        # Fall back to the structured generator which enforces schema via Pydantic
        structured_gen = StructuredGenerator(self.ollama_client)
        return await structured_gen.generate_structured_content(prompt, schema_class, **kwargs)


# Convenience functions
async def generate_room_json(prompt: str, client: Optional[OllamaClient] = None) -> Dict[str, Any]:
    """Generate a room using structured generation."""
    generator = StructuredGenerator(client)
    return await generator.generate_room(prompt)


async def generate_character_json(prompt: str, client: Optional[OllamaClient] = None) -> Dict[str, Any]:
    """Generate a character using structured generation."""
    generator = StructuredGenerator(client)
    return await generator.generate_character(prompt)


async def generate_item_json(prompt: str, client: Optional[OllamaClient] = None) -> Dict[str, Any]:
    """Generate an item using structured generation."""
    generator = StructuredGenerator(client)
    return await generator.generate_item(prompt)