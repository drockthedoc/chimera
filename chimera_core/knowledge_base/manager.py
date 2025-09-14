"""
Knowledge Manager for Project Chimera.

This module provides high-level operations for managing world knowledge,
implementing RAG patterns, and maintaining consistency across the game world.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from .client import InfinityClient, InfinityConfig
from .schemas import (
    DocumentSchema, SearchQuery, QueryResult, KnowledgeContext,
    RAGResponse, MemoryUpdate, ConsistencyCheck, ConsistencyResult,
    DocumentType
)
from ..ai_core.client import OllamaClient
from ..ai_core.agents import AgentManager, AgentRequest


logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    High-level manager for world knowledge and RAG operations.
    
    This class provides the main interface for storing, retrieving, and
    reasoning about world knowledge using the vector database and AI agents.
    """
    
    def __init__(
        self, 
        infinity_client: Optional[InfinityClient] = None,
        ollama_client: Optional[OllamaClient] = None,
        config: Optional[InfinityConfig] = None
    ):
        self.ollama_client = ollama_client or OllamaClient()
        self.infinity_client = infinity_client or InfinityClient(config, self.ollama_client)
        self.agent_manager = AgentManager(self.ollama_client)
        
        # Collections for different types of knowledge
        self.collections = {
            "world_lore": "world_lore",
            "character_memories": "character_memories", 
            "location_info": "location_info",
            "event_history": "event_history",
            "dialogue_history": "dialogue_history",
            "rules_and_facts": "rules_and_facts"
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.infinity_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.infinity_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge base by creating necessary collections.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create database
            await self.infinity_client.create_database()
            
            # Create all collections
            for collection_name in self.collections.values():
                await self.infinity_client.create_collection(collection_name)
            
            logger.info("Knowledge base initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    async def store_knowledge(
        self, 
        content: str, 
        document_type: DocumentType,
        metadata: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
        character: Optional[str] = None,
        importance: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Store a piece of knowledge in the appropriate collection.
        
        Args:
            content: The knowledge content
            document_type: Type of document
            metadata: Additional metadata
            location: Associated location
            character: Associated character
            importance: Importance score (0-10)
            tags: Tags for categorization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create document
            document = DocumentSchema(
                content=content,
                document_type=document_type,
                metadata=metadata or {},
                location=location,
                character=character,
                importance=importance,
                tags=tags or []
            )
            
            # Determine collection based on document type
            collection_name = self._get_collection_for_type(document_type)
            
            # Store document
            success = await self.infinity_client.store_document(document, collection_name)
            
            if success:
                logger.info(f"Stored {document_type} knowledge: {content[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            return False
    
    async def retrieve_relevant_knowledge(
        self, 
        query: str, 
        context: Optional[KnowledgeContext] = None,
        max_results: int = 10,
        document_types: Optional[List[DocumentType]] = None
    ) -> List[QueryResult]:
        """
        Retrieve knowledge relevant to a query using semantic search.
        
        Args:
            query: The search query
            context: Context information for filtering
            max_results: Maximum number of results
            document_types: Filter by document types
            
        Returns:
            List of relevant documents
        """
        try:
            # Build search query
            search_query = SearchQuery(
                query_text=query,
                max_results=max_results,
                document_types=document_types,
                location=context.current_location if context else None,
                min_score=0.3  # Minimum relevance threshold
            )
            
            # Search across relevant collections
            all_results = []
            
            if document_types:
                # Search specific collections
                for doc_type in document_types:
                    collection = self._get_collection_for_type(doc_type)
                    results = await self.infinity_client.search_documents(search_query, collection)
                    all_results.extend(results)
            else:
                # Search all collections
                for collection in self.collections.values():
                    results = await self.infinity_client.search_documents(search_query, collection)
                    all_results.extend(results)
            
            # Sort by score and limit results
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
    async def generate_with_context(
        self, 
        prompt: str, 
        context: Optional[KnowledgeContext] = None,
        agent_type: str = "geography",
        max_context_docs: int = 5
    ) -> RAGResponse:
        """
        Generate content using RAG (Retrieval-Augmented Generation).
        
        Args:
            prompt: The generation prompt
            context: Context information
            agent_type: Type of agent to use for generation
            max_context_docs: Maximum context documents to retrieve
            
        Returns:
            RAG response with generated content and sources
        """
        try:
            # Retrieve relevant context
            relevant_docs = await self.retrieve_relevant_knowledge(
                query=prompt,
                context=context,
                max_results=max_context_docs
            )
            
            # Build context-enhanced prompt
            context_text = self._build_context_text(relevant_docs, context)
            enhanced_prompt = f"{context_text}\n\nRequest: {prompt}"
            
            # Generate response using appropriate agent
            agent_request = AgentRequest(
                agent_type=agent_type,
                prompt=enhanced_prompt,
                context={
                    "original_prompt": prompt,
                    "context_docs_count": len(relevant_docs),
                    "current_location": context.current_location if context else None
                }
            )
            
            agent_response = await self.agent_manager.invoke_agent(agent_type, agent_request)
            
            # Calculate confidence based on context quality and agent confidence
            context_quality = min(1.0, len(relevant_docs) / max_context_docs)
            agent_confidence = 0.7  # Default confidence from agent
            overall_confidence = (context_quality + agent_confidence) / 2
            
            return RAGResponse(
                generated_content=agent_response.get("generated_data", {}).get("description", ""),
                source_documents=relevant_docs,
                context_used=context or KnowledgeContext(),
                confidence=overall_confidence,
                reasoning=f"Generated using {len(relevant_docs)} context documents"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate with context: {e}")
            return RAGResponse(
                generated_content="Error generating content",
                source_documents=[],
                context_used=context or KnowledgeContext(),
                confidence=0.0,
                reasoning=f"Generation failed: {str(e)}"
            )
    
    async def update_character_memory(
        self, 
        character_name: str, 
        memory_update: MemoryUpdate
    ) -> bool:
        """
        Update a character's memory with new information.
        
        Args:
            character_name: Name of the character
            memory_update: Memory update information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create memory document
            memory_content = f"{character_name} remembers: {memory_update.content}"
            
            document = DocumentSchema(
                content=memory_content,
                document_type=DocumentType.CHARACTER_MEMORY,
                character=character_name,
                importance=memory_update.importance,
                metadata={
                    "subject": memory_update.subject,
                    "emotional_context": memory_update.emotional_context,
                    "associated_entities": memory_update.associated_entities
                },
                tags=[character_name, "memory"] + [entity for entity in memory_update.associated_entities]
            )
            
            # Store in character memories collection
            return await self.infinity_client.store_document(
                document, 
                self.collections["character_memories"]
            )
            
        except Exception as e:
            logger.error(f"Failed to update character memory: {e}")
            return False
    
    async def check_consistency(self, check: ConsistencyCheck) -> ConsistencyResult:
        """
        Check if a statement is consistent with existing world knowledge.
        
        Args:
            check: Consistency check parameters
            
        Returns:
            Consistency check result
        """
        try:
            # Retrieve relevant documents
            relevant_docs = await self.retrieve_relevant_knowledge(
                query=check.statement,
                context=check.context,
                max_results=20
            )
            
            # Analyze for conflicts and support
            conflicting_docs = []
            supporting_docs = []
            
            # Use AI to analyze consistency
            analysis_prompt = f"""
            Analyze the following statement for consistency with the provided context:
            
            Statement: {check.statement}
            
            Context documents:
            {self._build_context_text(relevant_docs, check.context)}
            
            Determine if the statement conflicts with or is supported by the context.
            """
            
            agent_request = AgentRequest(
                agent_type="event",  # Use event agent for analysis
                prompt=analysis_prompt,
                context={"check_type": check.check_type}
            )
            
            analysis_response = await self.agent_manager.invoke_agent("event", agent_request)
            
            # Simple heuristic: high-scoring documents are more likely to be relevant
            for doc in relevant_docs:
                if doc.score > 0.8:
                    supporting_docs.append(doc)
                elif doc.score > 0.6:
                    # Could be conflicting - would need more sophisticated analysis
                    pass
            
            is_consistent = len(conflicting_docs) == 0
            confidence = min(1.0, len(supporting_docs) / 3)  # More support = higher confidence
            
            return ConsistencyResult(
                is_consistent=is_consistent,
                confidence=confidence,
                conflicting_documents=conflicting_docs,
                supporting_documents=supporting_docs,
                explanation=analysis_response.get("generated_data", {}).get("description", "Analysis completed"),
                suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to check consistency: {e}")
            return ConsistencyResult(
                is_consistent=False,
                confidence=0.0,
                conflicting_documents=[],
                supporting_documents=[],
                explanation=f"Consistency check failed: {str(e)}",
                suggestions=["Retry the consistency check", "Verify knowledge base connectivity"]
            )
    
    async def get_character_context(self, character_name: str, max_memories: int = 10) -> List[QueryResult]:
        """
        Get recent memories and context for a character.
        
        Args:
            character_name: Name of the character
            max_memories: Maximum number of memories to retrieve
            
        Returns:
            List of character memories and related context
        """
        search_query = SearchQuery(
            query_text=f"memories of {character_name}",
            character=character_name,
            document_types=[DocumentType.CHARACTER_MEMORY, DocumentType.DIALOGUE_HISTORY],
            max_results=max_memories,
            min_score=0.1
        )
        
        return await self.infinity_client.search_documents(
            search_query, 
            self.collections["character_memories"]
        )
    
    async def get_location_context(self, location_name: str, max_docs: int = 15) -> List[QueryResult]:
        """
        Get context information about a location.
        
        Args:
            location_name: Name of the location
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            List of location-related documents
        """
        search_query = SearchQuery(
            query_text=f"information about {location_name}",
            location=location_name,
            document_types=[DocumentType.LOCATION_INFO, DocumentType.EVENT_HISTORY, DocumentType.LORE],
            max_results=max_docs,
            min_score=0.2
        )
        
        return await self.infinity_client.search_documents(
            search_query,
            self.collections["location_info"]
        )
    
    def _get_collection_for_type(self, document_type: DocumentType) -> str:
        """Get the appropriate collection name for a document type."""
        type_to_collection = {
            DocumentType.LORE: self.collections["world_lore"],
            DocumentType.CHARACTER_MEMORY: self.collections["character_memories"],
            DocumentType.LOCATION_INFO: self.collections["location_info"],
            DocumentType.EVENT_HISTORY: self.collections["event_history"],
            DocumentType.DIALOGUE_HISTORY: self.collections["dialogue_history"],
            DocumentType.RULE: self.collections["rules_and_facts"],
            DocumentType.FACT: self.collections["rules_and_facts"],
            DocumentType.WORLD_STATE: self.collections["world_lore"]
        }
        
        return type_to_collection.get(document_type, self.collections["rules_and_facts"])
    
    def _build_context_text(
        self, 
        documents: List[QueryResult], 
        context: Optional[KnowledgeContext] = None
    ) -> str:
        """Build context text from retrieved documents."""
        if not documents:
            return "No relevant context found."
        
        context_parts = ["Relevant context:"]
        
        for i, doc_result in enumerate(documents[:10], 1):  # Limit to top 10
            doc = doc_result.document
            context_parts.append(f"{i}. {doc.content} (relevance: {doc_result.score:.2f})")
        
        if context:
            if context.current_location:
                context_parts.append(f"\nCurrent location: {context.current_location}")
            if context.active_characters:
                context_parts.append(f"Active characters: {', '.join(context.active_characters)}")
            if context.game_time:
                context_parts.append(f"Game time: {context.game_time}")
        
        return "\n".join(context_parts)
    
    async def cleanup_expired_documents(self) -> int:
        """
        Clean up expired documents from all collections.

        Returns:
            Number of documents cleaned up
        """
        removed_total = 0
        now_iso = datetime.utcnow().isoformat()

        for name, collection in self.collections.items():
            try:
                delete_data = {
                    "database_name": self.infinity_client.config.database_name,
                    "collection_name": collection,
                    "filter": {"expires_at": {"$lte": now_iso}},
                }

                # Use the Infinity API to delete all expired documents in one call.
                response = await self.infinity_client._make_request(
                    "DELETE", "documents", delete_data
                )

                # Different backends may return different keys for the number of
                # deleted documents, so handle common possibilities.
                removed = (
                    response.get("deleted", 0)
                    or response.get("num_deleted", 0)
                    or len(response.get("deleted_ids", []))
                )
                removed_total += removed
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup expired documents in collection '{collection}': {e}"
                )

        logger.info(f"Removed {removed_total} expired documents from knowledge base")
        return removed_total
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        stats = {}
        
        try:
            for name, collection in self.collections.items():
                collection_stats = await self.infinity_client.get_collection_stats(collection)
                stats[name] = collection_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {"error": str(e)}