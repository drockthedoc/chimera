"""
Logging configuration for Project Chimera.

This module provides centralized logging setup for the entire system.
"""

import logging
import logging.handlers
import os
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for Project Chimera.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for external libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Create chimera-specific logger
    chimera_logger = logging.getLogger("chimera")
    chimera_logger.setLevel(numeric_level)
    
    logging.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the chimera prefix.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return logging.getLogger(f"chimera.{name}")


class ChimeraLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log messages.
    """
    
    def process(self, msg, kwargs):
        """Add context information to log messages."""
        context = self.extra
        
        # Add context information to the message
        context_parts = []
        if "character" in context:
            context_parts.append(f"char={context['character']}")
        if "location" in context:
            context_parts.append(f"loc={context['location']}")
        if "task_id" in context:
            context_parts.append(f"task={context['task_id']}")
        
        if context_parts:
            msg = f"[{', '.join(context_parts)}] {msg}"
        
        return msg, kwargs


def get_context_logger(name: str, **context) -> ChimeraLoggerAdapter:
    """
    Get a logger with context information.
    
    Args:
        name: Logger name
        **context: Context information to include in logs
        
    Returns:
        Logger adapter with context
    """
    logger = get_logger(name)
    return ChimeraLoggerAdapter(logger, context)