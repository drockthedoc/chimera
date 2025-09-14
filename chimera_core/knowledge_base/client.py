"""
Infinity vector database client for Project Chimera.

This module provides the interface to communicate with an Infinity vector database
instance, handling document storage, embedding generation, and semantic search.
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

from .schemas import DocumentSchema, EmbeddingDocument, QueryResult, SearchQuery
from ..ai_core.client import OllamaClient


logger = logging.getLogger(__name__)


class InfinityConfig:
    """Configuration for Infinity database client."""
    def __init__(
        self,
        host: str = "localhost",
        port: int = 23817,
        database_name: str = "chimera_world",
        default_collection: str = "world_knowledge",
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.host = host
        self.port = port
        self.database_name = database_name
        self.default_collection = default_collection
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = f"http://{host}:{port}"


class InfinityClient:
    """
    Client for interacting with Infinity vector database.
    
    Provides methods for document storage, embedding generation,
    and semantic search operations.
    """
    
    def __init__(self, config: Optional[InfinityConfig] = None, ollama_client: Optional[OllamaClient] = None):
        self.config = config or InfinityConfig()
        self.ollama_client = ollama_client or OllamaClient()
        self.session: Optional[aiohttp.ClientSession] = None
        self._collections_created = set()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        await self.ollama_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        await self.ollama_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _ensure_session(self):
        """Ensure we have an active session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
    
    async def _make_request(
        self, 
        method: str,
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        retries: int = 0
    ) -> Dict[str, Any]:
        """
        Make a request to the Infinity API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            data: Request payload
            retries: Current retry count
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If request fails after all retries
        """
        await self._ensure_session()
        
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url, params=data) as response:
                    return await self._handle_response(response)
            else:
                async with self.session.request(method, url, json=data) as response:
                    return await self._handle_response(response)
                    
        except Exception as e:
            if retries < self.config.max_retries:
                logger.warning(f"Request failed, retrying ({retries + 1}/{self.config.max_retries}): {e}")
                await asyncio.sleep(2 ** retries)  # Exponential backoff
                return await self._make_request(method, endpoint, data, retries + 1)
            else:
                logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
                raise
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response and extract data."""
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise Exception(f"Infinity API error {response.status}: {error_text}")
    
    async def create_database(self) -> bool:
        """
        Create the database if it doesn't exist.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {"database_name": self.config.database_name}
            await self._make_request("POST", "databases", data)
            logger.info(f"Database '{self.config.database_name}' created successfully")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Database '{self.config.database_name}' already exists")
                return True
            else:
                logger.error(f"Failed to create database: {e}")
                return False
    
    async def create_collection(self, collection_name: str, dimension: int = 768) -> bool:
        """
        Create a collection for storing embeddings.
        
        Args:
            collection_name: Name of the collection
            dimension: Dimension of the embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if collection_name in self._collections_created:
            return True
            
        try:
            data = {
                "database_name": self.config.database_name,
                "collection_name": collection_name,
                "fields": [
                    {"name": "id", "type": "varchar"},
                    {"name": "content", "type": "varchar"},
                    {"name": "metadata", "type": "varchar"},
                    {"name": "embedding", "type": f"vector,{dimension},float"}
                ]
            }
            await self._make_request("POST", "collections", data)
            self._collections_created.add(collection_name)
            logger.info(f"Collection '{collection_name}' created successfully")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                self._collections_created.add(collection_name)
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            else:
                logger.error(f"Failed to create collection: {e}")
                return False
    
    async def store_document(
        self, 
        document: DocumentSchema, 
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Store a document in the vector database.
        
        Args:
            document: Document to store
            collection_name: Collection to store in
            
        Returns:
            True if successful, False otherwise
        """
        collection_name = collection_name or self.config.default_collection
        
        try:
            # Ensure collection exists
            await self.create_collection(collection_name)
            
            # Generate embedding for the document content
            embedding = await self.ollama_client.generate_embeddings(document.content)
            
            # Prepare document data
            doc_data = {
                "id": document.id or f"{document.document_type}_{datetime.utcnow().isoformat()}",
                "content": document.content,
                "metadata": json.dumps({
                    "document_type": document.document_type,
                    "tags": document.tags,
                    "source": document.source,
                    "timestamp": document.timestamp.isoformat(),
                    "importance": document.importance,
                    "location": document.location,
                    "character": document.character,
                    "game_time": document.game_time,
                    "expires_at": document.expires_at.isoformat() if document.expires_at else None,
                    **document.metadata
                }),
                "embedding": embedding
            }
            
            # Insert document
            insert_data = {
                "database_name": self.config.database_name,
                "collection_name": collection_name,
                "documents": [doc_data]
            }
            
            await self._make_request("POST", "documents", insert_data)
            logger.info(f"Document stored successfully in collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
    
    async def search_documents(
        self, 
        query: SearchQuery, 
        collection_name: Optional[str] = None
    ) -> List[QueryResult]:
        """
        Search for documents using semantic similarity.
        
        Args:
            query: Search query parameters
            collection_name: Collection to search in
            
        Returns:
            List of query results
        """
        collection_name = collection_name or self.config.default_collection
        
        try:
            # Generate embedding for the query
            query_embedding = await self.ollama_client.generate_embeddings(query.query_text)
            
            # Build search request
            search_data = {
                "database_name": self.config.database_name,
                "collection_name": collection_name,
                "query_vector": query_embedding,
                "top_k": query.max_results,
                "filter": self._build_filter(query)
            }
            
            # Perform search
            response = await self._make_request("POST", "search", search_data)
            
            # Parse results
            results = []
            for i, result in enumerate(response.get("results", [])):
                # Parse metadata
                metadata = json.loads(result.get("metadata", "{}"))
                
                # Reconstruct document
                document = DocumentSchema(
                    id=result.get("id"),
                    content=result.get("content"),
                    document_type=metadata.get("document_type", "fact"),
                    metadata={k: v for k, v in metadata.items() if k not in [
                        "document_type", "tags", "source", "timestamp", "importance",
                        "location", "character", "game_time", "expires_at"
                    ]},
                    tags=metadata.get("tags", []),
                    source=metadata.get("source"),
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
                    importance=metadata.get("importance", 1.0),
                    location=metadata.get("location"),
                    character=metadata.get("character"),
                    game_time=metadata.get("game_time"),
                    expires_at=datetime.fromisoformat(metadata["expires_at"]) if metadata.get("expires_at") else None
                )
                
                # Create query result
                query_result = QueryResult(
                    document=document,
                    score=result.get("score", 0.0),
                    rank=i + 1
                )
                
                # Apply score filter
                if query_result.score >= query.min_score:
                    results.append(query_result)
            
            logger.info(f"Found {len(results)} documents matching query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def _build_filter(self, query: SearchQuery) -> Dict[str, Any]:
        """Build filter conditions for the search query."""
        filters = {}
        
        if query.document_types:
            filters["document_type"] = {"$in": [dt.value for dt in query.document_types]}
        
        if query.tags:
            filters["tags"] = {"$in": query.tags}
        
        if query.location:
            filters["location"] = query.location
        
        if query.character:
            filters["character"] = query.character
        
        if query.time_range:
            time_filter = {}
            if "start" in query.time_range:
                time_filter["$gte"] = query.time_range["start"].isoformat()
            if "end" in query.time_range:
                time_filter["$lte"] = query.time_range["end"].isoformat()
            if time_filter:
                filters["timestamp"] = time_filter
        
        if not query.include_expired:
            filters["$or"] = [
                {"expires_at": {"$exists": False}},
                {"expires_at": None},
                {"expires_at": {"$gt": datetime.utcnow().isoformat()}}
            ]
        
        return filters
    
    async def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> bool:
        """
        Delete a document from the database.
        
        Args:
            document_id: ID of the document to delete
            collection_name: Collection containing the document
            
        Returns:
            True if successful, False otherwise
        """
        collection_name = collection_name or self.config.default_collection
        
        try:
            delete_data = {
                "database_name": self.config.database_name,
                "collection_name": collection_name,
                "filter": {"id": document_id}
            }
            
            await self._make_request("DELETE", "documents", delete_data)
            logger.info(f"Document '{document_id}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def update_document(
        self, 
        document_id: str, 
        updated_document: DocumentSchema,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            document_id: ID of the document to update
            updated_document: Updated document data
            collection_name: Collection containing the document
            
        Returns:
            True if successful, False otherwise
        """
        # For simplicity, we'll delete and re-insert
        # In a production system, you might want a more efficient update mechanism
        collection_name = collection_name or self.config.default_collection
        
        try:
            # Delete old document
            await self.delete_document(document_id, collection_name)
            
            # Insert updated document
            updated_document.id = document_id
            return await self.store_document(updated_document, collection_name)
            
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False
    
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection to get stats for
            
        Returns:
            Collection statistics
        """
        collection_name = collection_name or self.config.default_collection
        
        try:
            stats_data = {
                "database_name": self.config.database_name,
                "collection_name": collection_name
            }
            
            response = await self._make_request("GET", "collections/stats", stats_data)
            return response
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """
        Check if the Infinity database is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            await self._make_request("GET", "health")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False