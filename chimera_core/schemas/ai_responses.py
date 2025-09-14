"""
Pydantic schemas for AI system responses in Project Chimera.

These schemas define the structure for various types of responses
from the AI agents, ensuring consistent and validated output.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ResponseType(str, Enum):
    """Types of AI responses."""
    DIALOGUE = "dialogue"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    ERROR = "error"
    SUCCESS = "success"


class ConfidenceLevel(str, Enum):
    """Confidence levels for AI responses."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DialogueResponse(BaseModel):
    """Schema for dialogue responses from NPCs."""
    character_name: str = Field(..., description="Name of the speaking character")
    text: str = Field(..., min_length=1, description="The dialogue text")
    emotion: Optional[str] = Field(None, description="Emotional state of the character")
    actions: List[str] = Field(default_factory=list, description="Physical actions accompanying dialogue")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.MEDIUM, description="AI confidence in response")


class GenerationResponse(BaseModel):
    """Schema for content generation responses."""
    content_type: str = Field(..., description="Type of content generated (room, character, item, etc.)")
    generated_data: Dict[str, Any] = Field(..., description="The generated content data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the generation")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.MEDIUM, description="AI confidence in generation")
    consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="World consistency score")


class AnalysisResponse(BaseModel):
    """Schema for analysis responses from AI agents."""
    analysis_type: str = Field(..., description="Type of analysis performed")
    findings: List[str] = Field(..., description="Key findings from the analysis")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    data: Dict[str, Any] = Field(default_factory=dict, description="Supporting data")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.MEDIUM, description="AI confidence in analysis")


class ErrorResponse(BaseModel):
    """Schema for error responses from AI system."""
    error_type: str = Field(..., description="Type of error encountered")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: List[str] = Field(default_factory=list, description="Suggested solutions")


class AgentRequest(BaseModel):
    """Schema for requests to AI agents."""
    agent_type: str = Field(..., description="Type of agent to invoke")
    prompt: str = Field(..., description="The prompt or request")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")
    schema_name: Optional[str] = Field(None, description="Expected response schema")


class MultiAgentResponse(BaseModel):
    """Schema for responses from multiple AI agents."""
    primary_response: Union[DialogueResponse, GenerationResponse, AnalysisResponse] = Field(..., description="Primary agent response")
    supporting_responses: List[Union[DialogueResponse, GenerationResponse, AnalysisResponse]] = Field(default_factory=list, description="Supporting agent responses")
    consensus_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Agreement between agents")
    final_recommendation: Optional[str] = Field(None, description="Final recommendation based on all agents")


class WorldStateQuery(BaseModel):
    """Schema for querying world state information."""
    query_type: str = Field(..., description="Type of world state query")
    location: Optional[str] = Field(None, description="Specific location to query")
    character: Optional[str] = Field(None, description="Specific character to query")
    time_range: Optional[Dict[str, Any]] = Field(None, description="Time range for the query")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional query filters")


class WorldStateResponse(BaseModel):
    """Schema for world state query responses."""
    query_id: str = Field(..., description="Unique identifier for the query")
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    total_count: int = Field(..., ge=0, description="Total number of results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")
    timestamp: str = Field(..., description="Timestamp of the query")