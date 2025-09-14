# Project Chimera - Setup Guide

This guide will help you set up Project Chimera, an autonomous LLM-powered world engine built on Evennia.

## Prerequisites

- Python 3.9 or higher
- Git
- At least 8GB RAM (16GB recommended for optimal performance)
- 10GB free disk space

## System Dependencies

### 1. Install Ollama (Local LLM Server)

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download and install from: https://ollama.ai/download

**Start Ollama and pull required models:**
```bash
# Start Ollama service
ollama serve

# In another terminal, pull the required models
ollama pull llama3          # Main language model
ollama pull nomic-embed-text # Embedding model for vector search
```

### 2. Install Infinity Vector Database

**Option A: Docker (Recommended)**
```bash
# Pull and run Infinity container
docker pull infiniflow/infinity:latest
docker run -d --name infinity-db -p 23817:23817 infiniflow/infinity:latest
```

**Option B: Build from Source**
```bash
git clone https://github.com/infiniflow/infinity.git
cd infinity
# Follow build instructions in their README
```

## Project Setup

### 1. Clone and Install Project Chimera

```bash
# Clone the repository
git clone <repository-url> chimera
cd chimera

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Evennia

```bash
# Navigate to the Evennia game directory
cd evennia_game

# Initialize the database
evennia migrate

# Create a superuser account
evennia createsuperuser

# Collect static files
evennia collectstatic
```

### 3. Start the AI Manager

The AI Manager script needs to be created and started within Evennia:

```bash
# Start Evennia server
evennia start

# Connect to the game (in another terminal or web browser)
evennia connect

# In the game, as a superuser, start the AI system:
+ai_control start
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Infinity Database Configuration
INFINITY_HOST=localhost
INFINITY_PORT=23817
INFINITY_DATABASE=chimera_world

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/chimera.log

# AI System Configuration
MAX_CONCURRENT_TASKS=4
ENABLE_BACKGROUND_GENERATION=true
```

### Evennia Settings

The main configuration is in `evennia_game/server/conf/settings.py`. Key settings:

```python
# Add to INSTALLED_APPS if needed
INSTALLED_APPS += [
    'chimera_core',
]

# Configure logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs/chimera.log',
        },
    },
    'loggers': {
        'chimera': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

## Testing the Installation

### 1. Verify Services

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check Infinity is running
curl http://localhost:23817/health

# Check Evennia is running
evennia status
```

### 2. Test AI Commands

Connect to your Evennia game and try these commands:

```
# Check AI system status
+ai_status

# Create an AI-generated room
+llm_create_room a mystical forest clearing with ancient stone circles

# Create an AI-powered NPC
+llm_create_character/here a wise old wizard who knows ancient secrets

# Check your tasks
+my_tasks
```

## Troubleshooting

### Common Issues

**1. Ollama Connection Failed**
- Ensure Ollama is running: `ollama serve`
- Check if models are downloaded: `ollama list`
- Verify URL in configuration

**2. Infinity Database Connection Failed**
- Check if Infinity container is running: `docker ps`
- Verify port 23817 is accessible
- Check Infinity logs: `docker logs infinity-db`

**3. AI Manager Won't Start**
- Check Evennia logs: `evennia log`
- Verify Python path includes chimera_core
- Ensure all dependencies are installed

**4. Memory Issues**
- Reduce concurrent tasks in configuration
- Use smaller language models
- Increase system RAM or swap space

### Performance Optimization

**1. Model Selection**
- Use smaller models for faster responses: `ollama pull llama3:8b`
- Consider quantized models for lower memory usage

**2. Database Optimization**
- Regularly clean up old embeddings
- Monitor vector database size
- Use appropriate embedding dimensions

**3. System Resources**
- Monitor CPU and memory usage
- Adjust max_concurrent_tasks based on system capacity
- Use SSD storage for better I/O performance

## Development Setup

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest --cov=chimera_core tests/
```

### Code Quality

```bash
# Format code
black chimera_core/

# Lint code
flake8 chimera_core/

# Type checking (if using mypy)
mypy chimera_core/
```

## Production Deployment

### Security Considerations

1. **Change default passwords** for all services
2. **Use HTTPS** for web interfaces
3. **Configure firewalls** to restrict access to necessary ports only
4. **Regular backups** of game database and knowledge base
5. **Monitor logs** for suspicious activity

### Scaling

1. **Horizontal scaling**: Run multiple Evennia instances behind a load balancer
2. **Database scaling**: Use database clustering for Infinity
3. **Model serving**: Use multiple Ollama instances for load distribution
4. **Caching**: Implement Redis for session and response caching

### Monitoring

1. **System metrics**: CPU, memory, disk usage
2. **Application metrics**: Response times, error rates, task queue length
3. **AI metrics**: Model inference times, embedding generation rates
4. **Game metrics**: Player count, content generation rates

## Support and Documentation

- **Project Documentation**: See README.md for architecture overview
- **Evennia Documentation**: https://www.evennia.com/docs/
- **Ollama Documentation**: https://ollama.ai/docs/
- **Infinity Documentation**: https://github.com/infiniflow/infinity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.