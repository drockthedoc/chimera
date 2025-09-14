"""
Ollama client for Project Chimera AI Core.

This module provides the interface to communicate with a local Ollama server,
handling both text generation and embedding requests.
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import logging

from ..schemas.ai_responses import ErrorResponse, ConfidenceLevel


logger = logging.getLogger(__name__)


class OllamaConfig:
    """Configuration for Ollama client."""
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3",
        embedding_model: str = "nomic-embed-text",
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Provides methods for text generation, embeddings, and structured output.
    All methods are async to prevent blocking the Evennia game loop.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure we have an active session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
    
    async def _make_request(
        self, 
        endpoint: str, 
        data: Dict[str, Any],
        retries: int = 0
    ) -> Dict[str, Any]:
        """
        Make a request to the Ollama API with retry logic.
        
        Args:
            endpoint: API endpoint to call
            data: Request payload
            retries: Current retry count
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If request fails after all retries
        """
        await self._ensure_session()
        
        url = f"{self.config.base_url}/api/{endpoint}"
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
        except Exception as e:
            if retries < self.config.max_retries:
                logger.warning(f"Request failed, retrying ({retries + 1}/{self.config.max_retries}): {e}")
                await asyncio.sleep(2 ** retries)  # Exponential backoff
                return await self._make_request(endpoint, data, retries + 1)
            else:
                logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
                raise
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The input prompt
            model: Model to use (defaults to config default)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        model = model or self.config.default_model
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        if max_tokens:
            data["options"]["num_predict"] = max_tokens
        
        try:
            response = await self._make_request("generate", data)
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate chat response using Ollama chat API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to config default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        model = model or self.config.default_model
        
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            data["options"]["num_predict"] = max_tokens
        
        try:
            response = await self._make_request("chat", data)
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise
    
    async def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text using Ollama.
        
        Args:
            text: Text or list of texts to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector(s)
        """
        model = model or self.config.embedding_model
        
        # Handle single string input
        if isinstance(text, str):
            data = {
                "model": model,
                "prompt": text
            }
            
            try:
                response = await self._make_request("embeddings", data)
                return response.get("embedding", [])
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise
        
        # Handle list of strings
        else:
            embeddings = []
            for t in text:
                data = {
                    "model": model,
                    "prompt": t
                }
                
                try:
                    response = await self._make_request("embeddings", data)
                    embeddings.append(response.get("embedding", []))
                except Exception as e:
                    logger.error(f"Embedding generation failed for text '{t[:50]}...': {e}")
                    embeddings.append([])  # Add empty embedding on failure
            
            return embeddings
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.
        
        Returns:
            List of model information dictionaries
        """
        try:
            await self._ensure_session()
            url = f"{self.config.base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list models: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def check_health(self) -> bool:
        """
        Check if Ollama server is healthy and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            await self._ensure_session()
            url = f"{self.config.base_url}/api/tags"
            
            async with self.session.get(url) as response:
                return response.status == 200
                
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        data = {"name": model_name}
        
        try:
            response = await self._make_request("pull", data)
            return response.get("status") == "success"
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False


# Convenience functions for common operations
async def quick_generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Quick text generation without managing client lifecycle.
    
    Args:
        prompt: Input prompt
        system_prompt: Optional system prompt
        model: Model to use
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    async with OllamaClient() as client:
        return await client.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature
        )


async def quick_embed(text: Union[str, List[str]], model: Optional[str] = None) -> Union[List[float], List[List[float]]]:
    """
    Quick embedding generation without managing client lifecycle.
    
    Args:
        text: Text to embed
        model: Embedding model to use
        
    Returns:
        Embedding vector(s)
    """
    async with OllamaClient() as client:
        return await client.generate_embeddings(text=text, model=model)