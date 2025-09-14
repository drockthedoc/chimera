"""
Knowledge Base module for Project Chimera

This module provides the interface to the Infinity vector database,
implementing RAG (Retrieval-Augmented Generation) functionality for
maintaining world consistency and long-term memory.
"""

from .client import InfinityClient, InfinityConfig
from .manager import KnowledgeManager
from .schemas import DocumentSchema, QueryResult, EmbeddingDocument

__all__ = [
    "InfinityClient",
    "InfinityConfig", 
    "KnowledgeManager",
    "DocumentSchema",
    "QueryResult",
    "EmbeddingDocument",
]