# Project Chimera - Autonomous LLM-Powered World Engine

Project Chimera is an ambitious system that creates a persistent, text-based multiplayer world using the Evennia MUD engine, powered by a locally-hosted LLM that acts as an autonomous world-builder.

## Architecture

The system follows a decoupled, three-pillar architecture:

1. **Game Server (Evennia)**: The authoritative state manager and single source of truth for all game objects
2. **AI Core (Local LLM Server)**: The generative and reasoning engine via Ollama
3. **World Knowledge Base (Vector Database)**: Long-term memory using Infinity vector database for RAG

## Technology Stack

- **Game Engine**: Evennia
- **LLM Serving**: Ollama (Llama 3 and other models)
- **Vector Database**: Infinity (infiniflow/infinity)
- **Structured Generation**: outlines or LM Format Enforcer
- **Schema Definition**: Pydantic

## Project Structure

```
chimera/
├── evennia_game/          # Main Evennia game directory
├── chimera_core/          # Core framework modules
│   ├── ai_core/          # AI Core client and agents
│   ├── knowledge_base/   # Vector database interface
│   ├── schemas/          # Pydantic data models
│   └── utils/            # Utility functions
├── requirements.txt      # Python dependencies
└── setup.py             # Package setup
```

## Setup Instructions

1. Install Evennia: `pip install evennia`
2. Install Ollama and start the server
3. Install Infinity vector database
4. Install project dependencies: `pip install -r requirements.txt`
5. Initialize the Evennia game: `evennia migrate`
6. Start the game server: `evennia start`

## Current Status

✅ **Core Implementation Complete** ✅

### Completed Components
- [x] Project architecture and structure
- [x] Evennia game server initialization
- [x] Core schema definitions (Pydantic models)
- [x] AI Core implementation (Ollama integration)
- [x] Knowledge Base implementation (Infinity vector DB)
- [x] Evennia Typeclasses (LLMCharacter, GeneratedRoom)
- [x] Admin commands for content generation
- [x] Structured JSON generation with validation
- [x] Async task management for Evennia integration
- [x] Comprehensive documentation and setup guides

### Ready for Testing
The core framework is now complete and ready for testing. All major components have been implemented:

- **AI Agents**: Specialized agents for dialogue, geography, characters, and events
- **Knowledge Management**: RAG-powered world consistency and memory
- **Structured Generation**: Schema-validated content creation
- **Evennia Integration**: AI Manager script and enhanced typeclasses
- **Admin Commands**: Full suite of AI-powered content generation tools

## Features

- **Autonomous World Generation**: AI creates rooms, NPCs, and content dynamically
- **Intelligent NPCs**: Characters with memory, personality, and contextual dialogue
- **RAG-Powered Consistency**: Vector database ensures world coherence
- **Structured Generation**: Schema-validated JSON output from LLMs
- **Multi-Agent Framework**: Specialized AI agents for different domains
- **Admin Commands**: Easy-to-use commands for content generation
- **Async Integration**: Non-blocking AI operations in Evennia
- **Extensible Architecture**: Modular design for easy enhancement

## Documentation

- **[SETUP.md](SETUP.md)**: Complete installation and configuration guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system architecture documentation
- **[requirements.txt](requirements.txt)**: All project dependencies

## License

See LICENSE.md for details.