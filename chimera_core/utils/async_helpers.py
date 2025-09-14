"""
Async helper utilities for Project Chimera.

This module provides utilities for running async operations in Evennia's
synchronous environment and managing async tasks.
"""

import asyncio
import threading
import logging
from typing import Any, Callable, Coroutine, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import functools


logger = logging.getLogger(__name__)


def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Run an async coroutine in a separate thread with its own event loop.
    
    This is useful for running async operations from Evennia's synchronous
    context without blocking the main game loop.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    # Run in a thread pool to avoid blocking
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result()


def async_to_sync(func: Callable[..., Coroutine]) -> Callable:
    """
    Decorator to convert an async function to sync by running it in a thread.
    
    Args:
        func: Async function to convert
        
    Returns:
        Synchronous wrapper function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)
        return run_async_in_thread(coro)
    
    return wrapper


class AsyncTaskManager:
    """
    Manager for handling async tasks in Evennia's synchronous environment.
    
    This class provides a way to queue and execute async operations
    without blocking the main game loop.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Any] = {}
        self._task_counter = 0
    
    def submit_async_task(
        self, 
        coro: Coroutine, 
        task_name: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit an async task to be executed in a separate thread.
        
        Args:
            coro: The coroutine to execute
            task_name: Optional name for the task
            callback: Optional callback to call when task completes
            
        Returns:
            Task ID for tracking
        """
        self._task_counter += 1
        task_id = task_name or f"task_{self._task_counter}"
        
        def run_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                if callback:
                    callback(result)
                return result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                if callback:
                    callback(None, error=e)
                raise
            finally:
                loop.close()
                # Remove from active tasks
                self.active_tasks.pop(task_id, None)
        
        # Submit to thread pool
        future = self.executor.submit(run_task)
        self.active_tasks[task_id] = future
        
        logger.info(f"Submitted async task: {task_id}")
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a task (blocking).
        
        Args:
            task_id: ID of the task
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            KeyError: If task ID not found
            TimeoutError: If timeout exceeded
        """
        if task_id not in self.active_tasks:
            raise KeyError(f"Task {task_id} not found")
        
        future = self.active_tasks[task_id]
        return future.result(timeout=timeout)
    
    def is_task_done(self, task_id: str) -> bool:
        """
        Check if a task is completed.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if task is done, False otherwise
        """
        if task_id not in self.active_tasks:
            return True  # Task not found, assume done
        
        future = self.active_tasks[task_id]
        return future.done()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it hasn't started yet.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.active_tasks:
            return False
        
        future = self.active_tasks[task_id]
        cancelled = future.cancel()
        
        if cancelled:
            self.active_tasks.pop(task_id, None)
            logger.info(f"Cancelled task: {task_id}")
        
        return cancelled
    
    def get_active_tasks(self) -> Dict[str, bool]:
        """
        Get status of all active tasks.
        
        Returns:
            Dictionary mapping task IDs to completion status
        """
        return {task_id: future.done() for task_id, future in self.active_tasks.items()}
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the task manager.
        
        Args:
            wait: Whether to wait for active tasks to complete
        """
        logger.info("Shutting down AsyncTaskManager")
        self.executor.shutdown(wait=wait)
        self.active_tasks.clear()


# Global task manager instance
_global_task_manager: Optional[AsyncTaskManager] = None


def get_task_manager() -> AsyncTaskManager:
    """Get the global task manager instance."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = AsyncTaskManager()
    return _global_task_manager


def submit_background_task(
    coro: Coroutine, 
    task_name: Optional[str] = None,
    callback: Optional[Callable] = None
) -> str:
    """
    Convenience function to submit a background task.
    
    Args:
        coro: The coroutine to execute
        task_name: Optional name for the task
        callback: Optional callback to call when task completes
        
    Returns:
        Task ID for tracking
    """
    manager = get_task_manager()
    return manager.submit_async_task(coro, task_name, callback)


class AsyncContextManager:
    """
    Context manager for handling async resources in sync code.
    
    This helps manage async clients and resources that need proper
    cleanup in Evennia's synchronous environment.
    """
    
    def __init__(self, async_context_manager):
        self.async_context_manager = async_context_manager
        self.resource = None
    
    def __enter__(self):
        """Enter the context manager."""
        async def _enter():
            self.resource = await self.async_context_manager.__aenter__()
            return self.resource
        
        return run_async_in_thread(_enter())
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        async def _exit():
            if self.resource:
                await self.async_context_manager.__aexit__(exc_type, exc_val, exc_tb)
        
        run_async_in_thread(_exit())


def sync_context(async_context_manager):
    """
    Convert an async context manager to a sync one.
    
    Args:
        async_context_manager: The async context manager to convert
        
    Returns:
        Synchronous context manager
    """
    return AsyncContextManager(async_context_manager)