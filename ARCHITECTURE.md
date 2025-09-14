# Project Chimera - Architecture Documentation

## Overview

Project Chimera is an autonomous LLM-powered world engine built on the Evennia MUD framework. It implements a three-pillar architecture that combines traditional MUD gameplay with modern AI capabilities to create dynamic, evolving game worlds.

## Three-Pillar Architecture

### 1. Game Server (Evennia)
- **Purpose**: Core game mechanics, player management, world persistence
- **Technology**: Evennia MUD framework (Python/Django)
- **Responsibilities**:
  - Player authentication and session management
  - Game object persistence and state management
  - Command processing and game mechanics
  - Real-time multiplayer communication
  - Web interface and admin tools

### 2. AI Core (Ollama)
- **Purpose**: Language model inference and content generation
- **Technology**: Ollama local LLM server
- **Responsibilities**:
  - Natural language processing and generation
  - Character dialogue and personality simulation
  - Content generation (rooms, NPCs, items, events)
  - Contextual response generation
  - Embedding generation for semantic search

### 3. World Knowledge Base (Infinity)
- **Purpose**: Long-term memory and world consistency
- **Technology**: Infinity vector database
- **Responsibilities**:
  - Storing and retrieving world lore and facts
  - Character memory and relationship tracking
  - Semantic search and retrieval-augmented generation (RAG)
  - World state consistency checking
  - Historical event tracking

## System Components

### Core Framework (`chimera_core/`)

#### AI Core (`ai_core/`)
- **`client.py`**: Ollama HTTP client with async support
- **`agents.py`**: Specialized AI agents for different domains
  - `DialogueAgent`: Character conversation and responses
  - `GeographyAgent`: Room and location generation
  - `CharacterAgent`: NPC creation and personality
  - `EventAgent`: World events and quest generation
- **`manager.py`**: Agent coordination and multi-agent workflows

#### Knowledge Base (`knowledge_base/`)
- **`client.py`**: Infinity vector database client
- **`manager.py`**: High-level knowledge operations and RAG
- **`schemas.py`**: Data structures for documents and queries

#### Schemas (`schemas/`)
- **`game_objects.py`**: Pydantic models for game entities
- **`ai_responses.py`**: Structured AI response formats

#### Utilities (`utils/`)
- **`structured_generation.py`**: JSON schema validation and constraint-based generation
- **`async_helpers.py`**: Async/sync bridge for Evennia integration
- **`logging_config.py`**: Centralized logging configuration

### Evennia Integration (`evennia_game/`)

#### Typeclasses (`typeclasses/`)
- **`ai_manager.py`**: Global AI coordination script
- **`rooms.py`**: Enhanced room classes with AI capabilities
- **`characters.py`**: AI-powered NPC characters
- **`objects.py`**: Smart objects with AI behaviors

#### Commands (`commands/`)
- **`ai_commands.py`**: Admin commands for AI content generation
  - `+llm_create_room`: Generate rooms with AI
  - `+llm_create_character`: Create AI-powered NPCs
  - `+ai_status`: Monitor AI system health
  - `+ai_control`: Start/stop/configure AI services

## Data Flow

### 1. Content Generation Flow
```
Player Command → AI Manager → Agent Selection → LLM Generation → 
Schema Validation → Object Creation → World Integration
```

### 2. Character Interaction Flow
```
Player Speech → LLMCharacter → Context Building → Knowledge Retrieval → 
Dialogue Generation → Response Validation → Character Action
```

### 3. Knowledge Management Flow
```
Game Event → Knowledge Extraction → Embedding Generation → 
Vector Storage → Retrieval → Context Enhancement
```

## AI Agent Specialization

### DialogueAgent
- **Domain**: Character conversations and social interactions
- **Capabilities**:
  - Context-aware dialogue generation
  - Personality-consistent responses
  - Emotional state modeling
  - Memory integration

### GeographyAgent
- **Domain**: Spatial environments and locations
- **Capabilities**:
  - Room description generation
  - Exit and connection planning
  - Atmospheric consistency
  - Environmental storytelling

### CharacterAgent
- **Domain**: NPC creation and development
- **Capabilities**:
  - Personality generation
  - Background story creation
  - Goal and motivation assignment
  - Relationship modeling

### EventAgent
- **Domain**: Dynamic world events and narratives
- **Capabilities**:
  - Quest generation
  - World event creation
  - Cause-and-effect modeling
  - Narrative arc development

## Memory and Context Management

### Short-term Memory
- **Location**: In-game object attributes (`db` namespace)
- **Scope**: Individual characters, rooms, and objects
- **Duration**: Session-based or until explicitly cleared
- **Use Cases**: Recent conversations, immediate context

### Long-term Memory
- **Location**: Infinity vector database
- **Scope**: Global world knowledge
- **Duration**: Persistent across sessions
- **Use Cases**: World lore, character relationships, historical events

### Context Building
1. **Immediate Context**: Current location, present characters, recent actions
2. **Personal Context**: Character memories, relationships, goals
3. **World Context**: Relevant lore, ongoing events, world state
4. **Semantic Context**: Retrieved similar situations and responses

## Async Architecture

### Problem
Evennia is primarily synchronous, but AI operations are inherently async and potentially long-running.

### Solution
- **AsyncTaskManager**: Manages async operations in separate threads
- **Background Processing**: AI operations run without blocking game loop
- **Callback System**: Results delivered asynchronously to requesting objects
- **Task Tracking**: Monitor and manage long-running operations

### Implementation
```python
# Sync Evennia code
task_id = ai_manager.generate_room_async(prompt, callback=self.handle_result)

# Async execution in background thread
async def generate_room(prompt):
    result = await ollama_client.generate(prompt)
    return structured_generator.validate(result)
```

## Structured Generation

### Challenge
LLMs can produce inconsistent or invalid JSON output.

### Solution
- **Schema Validation**: Pydantic models ensure data consistency
- **Retry Logic**: Multiple attempts with error feedback
- **Example-based Prompting**: Few-shot examples guide generation
- **Constraint-based Generation**: Future integration with libraries like `outlines`

### Process
1. **Prompt Construction**: Include schema requirements and examples
2. **Generation**: LLM produces structured output
3. **Validation**: Pydantic schema validation
4. **Retry**: If invalid, retry with error feedback
5. **Integration**: Valid data integrated into game objects

## Scalability Considerations

### Horizontal Scaling
- **Multiple Evennia Instances**: Load balancer distributes players
- **Ollama Clustering**: Multiple LLM servers for parallel processing
- **Database Sharding**: Partition knowledge base by domain or region

### Vertical Scaling
- **Model Optimization**: Use quantized or smaller models for speed
- **Caching**: Cache frequent queries and responses
- **Batch Processing**: Group similar operations for efficiency

### Resource Management
- **Task Queuing**: Limit concurrent AI operations
- **Memory Management**: Monitor and clean up old data
- **Connection Pooling**: Reuse database and HTTP connections

## Security and Safety

### AI Safety
- **Content Filtering**: Validate generated content for appropriateness
- **Rate Limiting**: Prevent AI system abuse
- **Fallback Mechanisms**: Graceful degradation when AI unavailable
- **Human Oversight**: Admin tools for monitoring and intervention

### Data Security
- **Input Sanitization**: Clean user inputs before AI processing
- **Output Validation**: Ensure AI outputs are safe for game integration
- **Access Control**: Restrict AI admin commands to authorized users
- **Audit Logging**: Track all AI operations for security review

## Performance Optimization

### Caching Strategies
- **Response Caching**: Cache common AI responses
- **Embedding Caching**: Reuse embeddings for similar content
- **Context Caching**: Cache frequently accessed world knowledge

### Model Optimization
- **Model Selection**: Choose appropriate model size for use case
- **Quantization**: Use quantized models for faster inference
- **Batch Processing**: Process multiple requests together

### Database Optimization
- **Index Optimization**: Proper indexing for vector searches
- **Connection Pooling**: Reuse database connections
- **Query Optimization**: Efficient vector similarity searches

## Monitoring and Observability

### Metrics
- **AI Performance**: Response times, success rates, error rates
- **System Resources**: CPU, memory, disk usage
- **Game Metrics**: Player engagement, content generation rates
- **Database Performance**: Query times, storage usage

### Logging
- **Structured Logging**: JSON-formatted logs for analysis
- **Log Levels**: Appropriate verbosity for different components
- **Centralized Logging**: Aggregate logs from all components
- **Error Tracking**: Detailed error reporting and alerting

### Health Checks
- **Service Health**: Monitor Ollama, Infinity, and Evennia status
- **AI System Health**: Check agent responsiveness and accuracy
- **Database Health**: Monitor vector database performance
- **Integration Health**: Verify component communication

## Future Enhancements

### Advanced AI Features
- **Multi-modal AI**: Image and audio generation capabilities
- **Advanced RAG**: More sophisticated retrieval and ranking
- **Reinforcement Learning**: NPCs that learn from player interactions
- **Collaborative Filtering**: Player preference-based content generation

### Technical Improvements
- **Constraint-based Generation**: Guaranteed valid outputs
- **Distributed Processing**: Microservices architecture
- **Real-time Streaming**: Streaming AI responses for better UX
- **Advanced Caching**: Intelligent cache invalidation and warming

### Game Features
- **Dynamic World Events**: AI-driven world evolution
- **Procedural Quests**: Infinite quest generation
- **Adaptive Difficulty**: AI-adjusted challenge levels
- **Social AI**: Complex NPC relationships and politics

## Development Guidelines

### Code Organization
- **Separation of Concerns**: Clear boundaries between components
- **Async/Sync Boundaries**: Proper handling of async operations
- **Error Handling**: Comprehensive error handling and recovery
- **Testing**: Unit tests for all components

### AI Integration Patterns
- **Agent Pattern**: Specialized agents for different domains
- **Context Pattern**: Consistent context building across agents
- **Validation Pattern**: Schema validation for all AI outputs
- **Callback Pattern**: Async result handling

### Performance Patterns
- **Lazy Loading**: Load AI resources only when needed
- **Connection Pooling**: Reuse expensive connections
- **Batch Processing**: Group operations for efficiency
- **Caching**: Cache at multiple levels for performance