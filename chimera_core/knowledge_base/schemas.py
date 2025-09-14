"""
Schemas for the Knowledge Base system in Project Chimera.

These schemas define the structure for documents, queries, and results
in the vector database system.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Types of documents stored in the knowledge base."""
    LORE = "lore"
    CHARACTER_MEMORY = "character_memory"
    WORLD_STATE = "world_state"
    LOCATION_INFO = "location_info"
    EVENT_HISTORY = "event_history"
    DIALOGUE_HISTORY = "dialogue_history"
    RULE = "rule"
    FACT = "fact"


class DocumentSchema(BaseModel):
    """Schema for documents stored in the knowledge base."""
    id: Optional[str] = Field(None, description="Unique document identifier")
    content: str = Field(..., min_length=1, description="The document content/text")
    document_type: DocumentType = Field(..., description="Type of document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source: Optional[str] = Field(None, description="Source of the document")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    importance: float = Field(1.0, ge=0.0, le=10.0, description="Importance score (0-10)")
    
    # Location-specific fields
    location: Optional[str] = Field(None, description="Associated location")
    character: Optional[str] = Field(None, description="Associated character")
    
    # Temporal fields
    game_time: Optional[str] = Field(None, description="In-game time reference")
    expires_at: Optional[datetime] = Field(None, description="When this document expires")


class EmbeddingDocument(BaseModel):
    """Document with embedding vector for storage in vector database."""
    document: DocumentSchema = Field(..., description="The original document")
    embedding: List[float] = Field(..., description="Vector embedding of the content")
    collection: str = Field(..., description="Collection name in the database")


class QueryResult(BaseModel):
    """Result from a knowledge base query."""
    document: DocumentSchema = Field(..., description="The retrieved document")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Rank in the result set")


class SearchQuery(BaseModel):
    """Schema for search queries to the knowledge base."""
    query_text: str = Field(..., min_length=1, description="The search query text")
    document_types: Optional[List[DocumentType]] = Field(None, description="Filter by document types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    location: Optional[str] = Field(None, description="Filter by location")
    character: Optional[str] = Field(None, description="Filter by character")
    time_range: Optional[Dict[str, datetime]] = Field(None, description="Filter by time range")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    include_expired: bool = Field(False, description="Include expired documents")


class KnowledgeContext(BaseModel):
    """Context information for RAG operations."""
    current_location: Optional[str] = Field(None, description="Current location context")
    active_characters: List[str] = Field(default_factory=list, description="Currently active characters")
    recent_events: List[str] = Field(default_factory=list, description="Recent events for context")
    game_time: Optional[str] = Field(None, description="Current in-game time")
    player_context: Dict[str, Any] = Field(default_factory=dict, description="Player-specific context")


class RAGResponse(BaseModel):
    """Response from a RAG (Retrieval-Augmented Generation) operation."""
    generated_content: str = Field(..., description="The generated content")
    source_documents: List[QueryResult] = Field(..., description="Documents used for generation")
    context_used: KnowledgeContext = Field(..., description="Context information used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the response")
    reasoning: Optional[str] = Field(None, description="Explanation of the reasoning process")


class MemoryUpdate(BaseModel):
    """Schema for updating character or world memory."""
    subject: str = Field(..., description="What the memory is about")
    content: str = Field(..., description="The memory content")
    memory_type: DocumentType = Field(..., description="Type of memory")
    importance: float = Field(1.0, ge=0.0, le=10.0, description="Importance of this memory")
    associated_entities: List[str] = Field(default_factory=list, description="Related characters/locations")
    emotional_context: Optional[str] = Field(None, description="Emotional context of the memory")


class ConsistencyCheck(BaseModel):
    """Schema for world consistency checking."""
    statement: str = Field(..., description="Statement to check for consistency")
    context: KnowledgeContext = Field(..., description="Context for the check")
    check_type: str = Field("general", description="Type of consistency check")


class ConsistencyResult(BaseModel):
    """Result of a consistency check."""
    is_consistent: bool = Field(..., description="Whether the statement is consistent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the result")
    conflicting_documents: List[QueryResult] = Field(default_factory=list, description="Documents that conflict")
    supporting_documents: List[QueryResult] = Field(default_factory=list, description="Documents that support")
    explanation: str = Field(..., description="Explanation of the consistency check")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for resolving conflicts")