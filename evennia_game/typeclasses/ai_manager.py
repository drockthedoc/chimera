"""
AI Manager Script for Project Chimera.

This global script handles all asynchronous communication with the AI Core
and manages AI-driven world generation and character behaviors.
"""

import logging
from typing import Dict, Any, Optional, List
from evennia import DefaultScript
from evennia.utils import logger

# Import Chimera core modules
from chimera_core.ai_core.client import OllamaClient, OllamaConfig
from chimera_core.ai_core.agents import AgentManager, AgentRequest
from chimera_core.knowledge_base.manager import KnowledgeManager
from chimera_core.utils.structured_generation import StructuredGenerator
from chimera_core.utils.async_helpers import submit_background_task, get_task_manager
from chimera_core.schemas.game_objects import RoomSchema, CharacterSchema
from chimera_core.schemas.ai_responses import DialogueResponse, GenerationResponse


class AIManagerScript(DefaultScript):
    """
    Global script that manages all AI operations for the Chimera world engine.
    
    This script runs persistently and handles:
    - Communication with Ollama LLM server
    - World knowledge management via vector database
    - Structured content generation
    - Character AI behaviors
    - Background world events
    """
    
    def at_script_creation(self):
        """Called when the script is first created."""
        self.key = "ai_manager"
        self.desc = "Chimera AI Manager - handles all LLM operations"
        self.interval = 60  # Check every minute for background tasks
        self.persistent = True
        self.start_delay = True
        
        # Initialize AI components
        self.ollama_client = None
        self.agent_manager = None
        self.knowledge_manager = None
        self.structured_generator = None
        
        # Configuration
        self.config = {
            "ollama_url": "http://localhost:11434",
            "ollama_model": "llama3",
            "embedding_model": "nomic-embed-text",
            "infinity_host": "localhost",
            "infinity_port": 23817,
            "max_concurrent_tasks": 4,
            "enable_background_generation": True,
            "log_level": "INFO"
        }
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        
        logger.log_info("AI Manager Script created")
    
    def at_start(self):
        """Called when the script starts running."""
        logger.log_info("AI Manager Script starting...")
        
        try:
            # Initialize AI clients
            self._initialize_ai_components()
            
            # Test connectivity
            self._test_connections()
            
            logger.log_info("AI Manager Script started successfully")
            
        except Exception as e:
            logger.log_err(f"Failed to start AI Manager: {e}")
            self.stop()
    
    def at_repeat(self):
        """Called every interval (60 seconds)."""
        # Check for completed tasks
        self._process_completed_tasks()
        
        # Perform background maintenance
        self._background_maintenance()
        
        # Generate background content if enabled
        if self.config.get("enable_background_generation", True):
            self._generate_background_content()
    
    def _initialize_ai_components(self):
        """Initialize all AI components."""
        try:
            # Initialize Ollama client
            ollama_config = OllamaConfig(
                base_url=self.config["ollama_url"],
                default_model=self.config["ollama_model"],
                embedding_model=self.config["embedding_model"]
            )
            self.ollama_client = OllamaClient(ollama_config)
            
            # Initialize agent manager
            self.agent_manager = AgentManager(self.ollama_client)
            
            # Initialize knowledge manager
            self.knowledge_manager = KnowledgeManager(ollama_client=self.ollama_client)
            
            # Initialize structured generator
            self.structured_generator = StructuredGenerator(self.ollama_client)
            
            logger.log_info("AI components initialized successfully")
            
        except Exception as e:
            logger.log_err(f"Failed to initialize AI components: {e}")
            raise
    
    def _test_connections(self):
        """Test connections to external services."""
        def test_ollama():
            """Test Ollama connection."""
            async def _test():
                return await self.ollama_client.check_health()
            return _test()
        
        def test_infinity():
            """Test Infinity connection."""
            async def _test():
                async with self.knowledge_manager as km:
                    return await km.infinity_client.health_check()
            return _test()
        
        # Submit connection tests as background tasks
        ollama_task = submit_background_task(test_ollama(), "test_ollama")
        infinity_task = submit_background_task(test_infinity(), "test_infinity")
        
        logger.log_info("Connection tests submitted")
    
    def _process_completed_tasks(self):
        """Process completed background tasks."""
        task_manager = get_task_manager()
        completed_task_ids = []
        
        for task_id, task_info in self.active_tasks.items():
            if task_manager.is_task_done(task_id):
                try:
                    result = task_manager.get_task_result(task_id, timeout=0.1)
                    self.completed_tasks[task_id] = {
                        "result": result,
                        "task_info": task_info,
                        "completed_at": self.db.time
                    }
                    completed_task_ids.append(task_id)
                    
                    # Handle specific task types
                    self._handle_completed_task(task_id, task_info, result)
                    
                except Exception as e:
                    logger.log_err(f"Error processing completed task {task_id}: {e}")
                    completed_task_ids.append(task_id)
        
        # Remove completed tasks from active list
        for task_id in completed_task_ids:
            self.active_tasks.pop(task_id, None)
    
    def _handle_completed_task(self, task_id: str, task_info: Dict[str, Any], result: Any):
        """Handle a completed task based on its type."""
        task_type = task_info.get("type", "unknown")
        
        if task_type == "generate_room":
            self._handle_room_generation_complete(task_id, task_info, result)
        elif task_type == "generate_character":
            self._handle_character_generation_complete(task_id, task_info, result)
        elif task_type == "dialogue_response":
            self._handle_dialogue_complete(task_id, task_info, result)
        elif task_type == "background_event":
            self._handle_background_event(task_id, task_info, result)
        elif task_type == "test_ollama":
            logger.log_info(f"Ollama connection test: {'OK' if result else 'FAILED'}")
        elif task_type == "test_infinity":
            logger.log_info(f"Infinity connection test: {'OK' if result else 'FAILED'}")
    
    def _handle_room_generation_complete(self, task_id: str, task_info: Dict[str, Any], result: Any):
        """Handle completed room generation."""
        if result and isinstance(result, dict):
            logger.log_info(f"Room generation completed: {result.get('name', 'Unknown')}")
            
            # Notify the requesting object if specified
            requester = task_info.get("requester")
            if requester:
                requester.msg(f"Room generation completed: {result.get('name', 'Unknown')}")
        else:
            logger.log_err(f"Room generation failed for task {task_id}")
    
    def _handle_character_generation_complete(self, task_id: str, task_info: Dict[str, Any], result: Any):
        """Handle completed character generation."""
        if result and isinstance(result, dict):
            logger.log_info(f"Character generation completed: {result.get('name', 'Unknown')}")
            
            # Notify the requesting object if specified
            requester = task_info.get("requester")
            if requester:
                requester.msg(f"Character generation completed: {result.get('name', 'Unknown')}")
        else:
            logger.log_err(f"Character generation failed for task {task_id}")
    
    def _handle_dialogue_complete(self, task_id: str, task_info: Dict[str, Any], result: Any):
        """Handle completed dialogue generation."""
        if result and isinstance(result, dict):
            character_name = result.get("character_name", "Unknown")
            dialogue_text = result.get("text", "...")
            
            # Find the character object and make them speak
            character = task_info.get("character_obj")
            if character:
                character.location.msg_contents(f"{character_name} says, \"{dialogue_text}\"")
            
            logger.log_info(f"Dialogue generated for {character_name}")
        else:
            logger.log_err(f"Dialogue generation failed for task {task_id}")

    def _handle_background_event(self, task_id: str, task_info: Dict[str, Any], result: Any):
        """Handle completed background event generation."""
        if result and isinstance(result, dict):
            description = result.get("description") or result.get("generated_data", {}).get(
                "description", ""
            )
            logger.log_info(f"Background event: {description}")
        else:
            logger.log_err(f"Background event generation failed for task {task_id}")
    
    def _background_maintenance(self):
        """Perform background maintenance tasks."""
        # Clean up old completed tasks (keep only last 100)
        if len(self.completed_tasks) > 100:
            # Remove oldest tasks
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].get("completed_at", 0)
            )
            
            for task_id, _ in sorted_tasks[:-100]:
                self.completed_tasks.pop(task_id, None)
    
    def _generate_background_content(self):
        """Generate background content to enrich the world."""
        # Only generate if we don't have too many active tasks
        if len(self.active_tasks) >= self.config.get("max_concurrent_tasks", 4):
            return

        # Generate a simple background world event using the event agent.
        async def _generate_event():
            agent_request = AgentRequest(
                agent_type="event",
                prompt="Generate a brief background event happening in the world.",
                schema_name="WorldEventSchema",
            )
            return await self.agent_manager.invoke_agent("event", agent_request)

        task_id = submit_background_task(
            _generate_event(),
            task_name=f"background_event_{len(self.active_tasks)}",
        )

        self.active_tasks[task_id] = {
            "type": "background_event",
            "started_at": self.db.time,
        }
    
    # Public API methods for other game objects to use
    
    def generate_room_async(
        self, 
        prompt: str, 
        requester=None,
        callback=None
    ) -> str:
        """
        Generate a room asynchronously.
        
        Args:
            prompt: Description of the room to generate
            requester: Object that requested the generation (for notifications)
            callback: Optional callback function
            
        Returns:
            Task ID for tracking
        """
        async def _generate():
            return await self.structured_generator.generate_room(prompt)
        
        task_id = submit_background_task(
            _generate(),
            task_name=f"generate_room_{len(self.active_tasks)}",
            callback=callback
        )
        
        self.active_tasks[task_id] = {
            "type": "generate_room",
            "prompt": prompt,
            "requester": requester,
            "started_at": self.db.time
        }
        
        return task_id
    
    def generate_character_async(
        self, 
        prompt: str, 
        requester=None,
        callback=None
    ) -> str:
        """
        Generate a character asynchronously.
        
        Args:
            prompt: Description of the character to generate
            requester: Object that requested the generation
            callback: Optional callback function
            
        Returns:
            Task ID for tracking
        """
        async def _generate():
            return await self.structured_generator.generate_character(prompt)
        
        task_id = submit_background_task(
            _generate(),
            task_name=f"generate_character_{len(self.active_tasks)}",
            callback=callback
        )
        
        self.active_tasks[task_id] = {
            "type": "generate_character",
            "prompt": prompt,
            "requester": requester,
            "started_at": self.db.time
        }
        
        return task_id
    
    def generate_dialogue_async(
        self, 
        character_obj,
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        callback=None
    ) -> str:
        """
        Generate character dialogue asynchronously.
        
        Args:
            character_obj: The character object
            prompt: Dialogue prompt
            context: Additional context information
            callback: Optional callback function
            
        Returns:
            Task ID for tracking
        """
        async def _generate():
            agent_request = AgentRequest(
                agent_type="dialogue",
                prompt=prompt,
                context=context or {"character_name": character_obj.key}
            )
            return await self.agent_manager.invoke_agent("dialogue", agent_request)
        
        task_id = submit_background_task(
            _generate(),
            task_name=f"dialogue_{character_obj.key}_{len(self.active_tasks)}",
            callback=callback
        )
        
        self.active_tasks[task_id] = {
            "type": "dialogue_response",
            "character_obj": character_obj,
            "prompt": prompt,
            "context": context,
            "started_at": self.db.time
        }
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status information
        """
        if task_id in self.active_tasks:
            is_done = get_task_manager().is_task_done(task_id)
            return {
                "status": "completed" if is_done else "running",
                "task_info": self.active_tasks[task_id]
            }
        elif task_id in self.completed_tasks:
            return {
                "status": "completed",
                "result": self.completed_tasks[task_id]
            }
        else:
            return {"status": "not_found"}
    
    def get_active_tasks_count(self) -> int:
        """Get the number of active tasks."""
        return len(self.active_tasks)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "config": self.config,
            "uptime": getattr(self.db, 'time', 0)
        }