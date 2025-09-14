"""
AI-powered admin commands for Project Chimera.

These commands allow administrators to generate content using the AI system,
manage the knowledge base, and control AI behaviors.
"""

import sys
import os
from typing import Dict, Any, Optional
from evennia import Command, default_cmds
from evennia.utils import logger
from evennia.utils.evtable import EvTable

# Add chimera_core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class CmdLLMCreateRoom(Command):
    """
    Create a room using AI generation.
    
    Usage:
        +llm_create_room <description>
        +llm_create_room/here <description>
        
    Examples:
        +llm_create_room a mystical forest clearing with ancient stone circles
        +llm_create_room/here an abandoned wizard's tower filled with magical artifacts
        
    Switches:
        /here - Create the room and immediately move to it
        
    This command uses the AI system to generate a detailed room based on your
    description. The AI will create appropriate descriptions, exits, and atmosphere.
    """
    
    key = "+llm_create_room"
    aliases = ["llm_room", "ai_room"]
    locks = "cmd:perm(Builder)"
    help_category = "AI Commands"
    
    def func(self):
        """Execute the command."""
        if not self.args:
            self.caller.msg("Usage: +llm_create_room <description>")
            return
        
        # Get the AI manager
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            self.caller.msg("AI Manager is not available. Make sure it's running.")
            return
        
        # Check if we have too many active tasks
        if ai_manager.get_active_tasks_count() >= 4:
            self.caller.msg("AI system is busy. Please try again in a moment.")
            return
        
        description = self.args.strip()
        move_to_room = "here" in self.switches
        
        self.caller.msg(f"Generating room: {description}")
        self.caller.msg("This may take a moment...")
        
        # Submit room generation task
        task_id = ai_manager.generate_room_async(
            prompt=description,
            requester=self.caller,
            callback=lambda result, error=None: self._handle_room_creation(
                result, error, move_to_room, description
            )
        )
        
        # Store task info for tracking
        if not hasattr(self.caller.db, 'ai_tasks'):
            self.caller.db.ai_tasks = {}
        
        self.caller.db.ai_tasks[task_id] = {
            "type": "room_creation",
            "description": description,
            "move_to_room": move_to_room
        }
        
        self.caller.msg(f"Room generation task submitted (ID: {task_id})")
    
    def _handle_room_creation(self, result, error, move_to_room, description):
        """Handle the completion of room generation."""
        if error:
            self.caller.msg(f"Room generation failed: {error}")
            return
        
        if not result or not isinstance(result, dict):
            self.caller.msg("Room generation failed: Invalid result")
            return
        
        try:
            # Import here to avoid circular imports
            from evennia import create_object
            from ..typeclasses.rooms import GeneratedRoom
            
            # Create the room object
            room = create_object(
                GeneratedRoom,
                key=result.get("name", "Generated Room"),
                location=None  # Rooms have no location
            )
            
            # Update room with generated data
            room.update_from_generation_data(result)
            
            # Create exits if specified
            exits_data = result.get("exits", [])
            if exits_data and self.caller.location:
                self._create_exits(room, self.caller.location, exits_data)
            
            # Notify the caller
            self.caller.msg(f"Room '{room.key}' created successfully!")
            self.caller.msg(f"Description: {room.db.desc}")
            
            # Move to room if requested
            if move_to_room:
                self.caller.move_to(room)
                self.caller.msg("You have been moved to the new room.")
            
            logger.log_info(f"AI-generated room created: {room.key} by {self.caller.key}")
            
        except Exception as e:
            self.caller.msg(f"Failed to create room object: {e}")
            logger.log_err(f"Room creation failed: {e}")
    
    def _create_exits(self, new_room, current_room, exits_data):
        """Create exits between rooms."""
        from evennia import create_object
        from ..typeclasses.exits import Exit
        
        for exit_info in exits_data[:2]:  # Limit to 2 exits to avoid complexity
            direction = exit_info.get("direction", "north")
            exit_desc = exit_info.get("description", f"An exit leading {direction}")
            
            # Create exit from current room to new room
            exit_there = create_object(
                Exit,
                key=direction,
                location=current_room,
                destination=new_room
            )
            exit_there.db.desc = exit_desc
            
            # Create return exit
            opposite_directions = {
                "north": "south", "south": "north",
                "east": "west", "west": "east",
                "up": "down", "down": "up",
                "northeast": "southwest", "southwest": "northeast",
                "northwest": "southeast", "southeast": "northwest"
            }
            
            opposite = opposite_directions.get(direction, "back")
            exit_back = create_object(
                Exit,
                key=opposite,
                location=new_room,
                destination=current_room
            )
            exit_back.db.desc = f"An exit leading {opposite}"
            
            self.caller.msg(f"Created exits: {direction} <-> {opposite}")
    
    def _get_ai_manager(self):
        """Get the AI manager script."""
        from evennia import search_script
        ai_managers = search_script("ai_manager")
        return ai_managers[0] if ai_managers else None


class CmdLLMCreateCharacter(Command):
    """
    Create an AI-powered NPC character.
    
    Usage:
        +llm_create_character <description>
        +llm_create_character/here <description>
        
    Examples:
        +llm_create_character a wise old wizard who knows ancient secrets
        +llm_create_character/here a friendly tavern keeper with many stories
        
    Switches:
        /here - Create the character in your current location
        
    This command uses the AI system to generate a detailed NPC character
    with personality, background, and AI-driven dialogue capabilities.
    """
    
    key = "+llm_create_character"
    aliases = ["llm_char", "ai_char", "llm_npc"]
    locks = "cmd:perm(Builder)"
    help_category = "AI Commands"
    
    def func(self):
        """Execute the command."""
        if not self.args:
            self.caller.msg("Usage: +llm_create_character <description>")
            return
        
        # Get the AI manager
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            self.caller.msg("AI Manager is not available. Make sure it's running.")
            return
        
        # Check if we have too many active tasks
        if ai_manager.get_active_tasks_count() >= 4:
            self.caller.msg("AI system is busy. Please try again in a moment.")
            return
        
        description = self.args.strip()
        create_here = "here" in self.switches
        
        self.caller.msg(f"Generating character: {description}")
        self.caller.msg("This may take a moment...")
        
        # Submit character generation task
        task_id = ai_manager.generate_character_async(
            prompt=description,
            requester=self.caller,
            callback=lambda result, error=None: self._handle_character_creation(
                result, error, create_here, description
            )
        )
        
        # Store task info for tracking
        if not hasattr(self.caller.db, 'ai_tasks'):
            self.caller.db.ai_tasks = {}
        
        self.caller.db.ai_tasks[task_id] = {
            "type": "character_creation",
            "description": description,
            "create_here": create_here
        }
        
        self.caller.msg(f"Character generation task submitted (ID: {task_id})")
    
    def _handle_character_creation(self, result, error, create_here, description):
        """Handle the completion of character generation."""
        if error:
            self.caller.msg(f"Character generation failed: {error}")
            return
        
        if not result or not isinstance(result, dict):
            self.caller.msg("Character generation failed: Invalid result")
            return
        
        try:
            # Import here to avoid circular imports
            from evennia import create_object
            from ..typeclasses.characters import LLMCharacter
            
            # Determine location
            location = self.caller.location if create_here else None
            
            # Create the character object
            character = create_object(
                LLMCharacter,
                key=result.get("name", "Generated Character"),
                location=location
            )
            
            # Update character with generated data
            character.update_from_generation_data(result)
            
            # Notify the caller
            self.caller.msg(f"Character '{character.key}' created successfully!")
            self.caller.msg(f"Description: {character.db.desc}")
            self.caller.msg(f"Personality: {character.db.personality}")
            
            if create_here:
                self.caller.msg(f"{character.key} has been created in your current location.")
            
            logger.log_info(f"AI-generated character created: {character.key} by {self.caller.key}")
            
        except Exception as e:
            self.caller.msg(f"Failed to create character object: {e}")
            logger.log_err(f"Character creation failed: {e}")
    
    def _get_ai_manager(self):
        """Get the AI manager script."""
        from evennia import search_script
        ai_managers = search_script("ai_manager")
        return ai_managers[0] if ai_managers else None


class CmdAIStatus(Command):
    """
    Check the status of the AI system.
    
    Usage:
        +ai_status
        +ai_status/tasks
        +ai_status/config
        
    Switches:
        /tasks - Show detailed information about active and completed tasks
        /config - Show AI system configuration
        
    This command displays information about the AI system's current state,
    including active tasks, system health, and configuration.
    """
    
    key = "+ai_status"
    aliases = ["ai_stat", "llm_status"]
    locks = "cmd:perm(Builder)"
    help_category = "AI Commands"
    
    def func(self):
        """Execute the command."""
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            self.caller.msg("AI Manager is not available.")
            return
        
        if "tasks" in self.switches:
            self._show_tasks(ai_manager)
        elif "config" in self.switches:
            self._show_config(ai_manager)
        else:
            self._show_status(ai_manager)
    
    def _show_status(self, ai_manager):
        """Show general AI system status."""
        status = ai_manager.get_system_status()
        
        table = EvTable("Property", "Value", border="cells")
        table.add_row("Active Tasks", status.get("active_tasks", 0))
        table.add_row("Completed Tasks", status.get("completed_tasks", 0))
        table.add_row("Uptime", f"{status.get('uptime', 0)} seconds")
        
        self.caller.msg("AI System Status:")
        self.caller.msg(str(table))
    
    def _show_tasks(self, ai_manager):
        """Show detailed task information."""
        # Show active tasks
        active_tasks = ai_manager.active_tasks
        if active_tasks:
            table = EvTable("Task ID", "Type", "Started", border="cells")
            for task_id, task_info in active_tasks.items():
                table.add_row(
                    task_id[:8] + "...",  # Truncate ID
                    task_info.get("type", "unknown"),
                    str(task_info.get("started_at", "unknown"))
                )
            
            self.caller.msg("Active Tasks:")
            self.caller.msg(str(table))
        else:
            self.caller.msg("No active tasks.")
        
        # Show recent completed tasks
        completed_tasks = ai_manager.completed_tasks
        if completed_tasks:
            recent_completed = list(completed_tasks.items())[-5:]  # Last 5
            table = EvTable("Task ID", "Type", "Completed", "Status", border="cells")
            
            for task_id, task_data in recent_completed:
                task_info = task_data.get("task_info", {})
                result = task_data.get("result")
                status = "Success" if result else "Failed"
                
                table.add_row(
                    task_id[:8] + "...",
                    task_info.get("type", "unknown"),
                    str(task_data.get("completed_at", "unknown")),
                    status
                )
            
            self.caller.msg("\nRecent Completed Tasks:")
            self.caller.msg(str(table))
    
    def _show_config(self, ai_manager):
        """Show AI system configuration."""
        config = ai_manager.config
        
        table = EvTable("Setting", "Value", border="cells")
        for key, value in config.items():
            table.add_row(key, str(value))
        
        self.caller.msg("AI System Configuration:")
        self.caller.msg(str(table))
    
    def _get_ai_manager(self):
        """Get the AI manager script."""
        from evennia import search_script
        ai_managers = search_script("ai_manager")
        return ai_managers[0] if ai_managers else None


class CmdAIControl(Command):
    """
    Control the AI system.
    
    Usage:
        +ai_control start
        +ai_control stop
        +ai_control restart
        +ai_control config <setting> <value>
        
    Examples:
        +ai_control start
        +ai_control config enable_background_generation False
        +ai_control restart
        
    This command allows administrators to control the AI system,
    modify configuration, and restart services.
    """
    
    key = "+ai_control"
    aliases = ["ai_ctrl"]
    locks = "cmd:perm(Admin)"
    help_category = "AI Commands"
    
    def func(self):
        """Execute the command."""
        if not self.args:
            self.caller.msg("Usage: +ai_control <start|stop|restart|config>")
            return
        
        args = self.args.split()
        command = args[0].lower()
        
        if command == "start":
            self._start_ai_system()
        elif command == "stop":
            self._stop_ai_system()
        elif command == "restart":
            self._restart_ai_system()
        elif command == "config" and len(args) >= 3:
            setting = args[1]
            value = " ".join(args[2:])
            self._update_config(setting, value)
        else:
            self.caller.msg("Invalid command. Use: start, stop, restart, or config <setting> <value>")
    
    def _start_ai_system(self):
        """Start the AI system."""
        from evennia import create_script
        from ..typeclasses.ai_manager import AIManagerScript
        
        # Check if already running
        ai_manager = self._get_ai_manager()
        if ai_manager:
            self.caller.msg("AI system is already running.")
            return
        
        # Create and start the AI manager script
        try:
            script = create_script(AIManagerScript, key="ai_manager")
            self.caller.msg("AI system started successfully.")
            logger.log_info(f"AI system started by {self.caller.key}")
        except Exception as e:
            self.caller.msg(f"Failed to start AI system: {e}")
            logger.log_err(f"Failed to start AI system: {e}")
    
    def _stop_ai_system(self):
        """Stop the AI system."""
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            self.caller.msg("AI system is not running.")
            return
        
        try:
            ai_manager.stop()
            self.caller.msg("AI system stopped.")
            logger.log_info(f"AI system stopped by {self.caller.key}")
        except Exception as e:
            self.caller.msg(f"Failed to stop AI system: {e}")
            logger.log_err(f"Failed to stop AI system: {e}")
    
    def _restart_ai_system(self):
        """Restart the AI system."""
        self.caller.msg("Restarting AI system...")
        self._stop_ai_system()
        self._start_ai_system()
    
    def _update_config(self, setting, value):
        """Update AI system configuration."""
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            self.caller.msg("AI system is not running.")
            return
        
        # Convert value to appropriate type
        try:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
        except ValueError:
            pass  # Keep as string
        
        # Update configuration
        if hasattr(ai_manager, 'config') and setting in ai_manager.config:
            old_value = ai_manager.config[setting]
            ai_manager.config[setting] = value
            self.caller.msg(f"Updated {setting}: {old_value} -> {value}")
            logger.log_info(f"AI config updated by {self.caller.key}: {setting} = {value}")
        else:
            self.caller.msg(f"Unknown configuration setting: {setting}")
    
    def _get_ai_manager(self):
        """Get the AI manager script."""
        from evennia import search_script
        ai_managers = search_script("ai_manager")
        return ai_managers[0] if ai_managers else None


class CmdMyTasks(Command):
    """
    View your AI generation tasks.
    
    Usage:
        +my_tasks
        +my_tasks/clear
        
    Switches:
        /clear - Clear your completed task history
        
    This command shows the status of AI generation tasks you've submitted.
    """
    
    key = "+my_tasks"
    aliases = ["my_ai_tasks"]
    locks = "cmd:all()"
    help_category = "AI Commands"
    
    def func(self):
        """Execute the command."""
        if "clear" in self.switches:
            self.caller.db.ai_tasks = {}
            self.caller.msg("Task history cleared.")
            return
        
        if not hasattr(self.caller.db, 'ai_tasks') or not self.caller.db.ai_tasks:
            self.caller.msg("You have no AI tasks.")
            return
        
        ai_manager = self._get_ai_manager()
        if not ai_manager:
            self.caller.msg("AI Manager is not available.")
            return
        
        table = EvTable("Task ID", "Type", "Description", "Status", border="cells")
        
        for task_id, task_info in self.caller.db.ai_tasks.items():
            status_info = ai_manager.get_task_status(task_id)
            status = status_info.get("status", "unknown")
            
            table.add_row(
                task_id[:8] + "...",
                task_info.get("type", "unknown"),
                task_info.get("description", "")[:30] + "...",
                status.title()
            )
        
        self.caller.msg("Your AI Tasks:")
        self.caller.msg(str(table))
    
    def _get_ai_manager(self):
        """Get the AI manager script."""
        from evennia import search_script
        ai_managers = search_script("ai_manager")
        return ai_managers[0] if ai_managers else None