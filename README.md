# RustLLM

A Rust-based local LLM chat tool similar to Ollama.

## Features

- Run local LLMs directly on your machine
- Interactive CLI chat interface
- REST API server for integration with other applications
- Model management (download, list, delete)
- Support for GGUF format models

## System Design

### Architecture Overview

RustLLM follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    RustLLM System                           │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface                    │  REST API Server        │
│  ┌─────────────────┐             │  ┌─────────────────┐    │
│  │ Interactive     │             │  │ HTTP Endpoints  │    │
│  │ Chat Session    │             │  │ - /api/models   │    │
│  │ - User Input    │             │  │ - /api/chat     │    │
│  │ - Model Output  │             │  │ - /api/health   │    │
│  │ - Commands      │             │  └─────────────────┘    │
│  └─────────────────┘             │                         │
├─────────────────────────────────────────────────────────────┤
│                     Core Engine                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Model Manager   │  │ Inference Engine│  │ Chat Context │ │
│  │ - Load/Unload   │  │ - GGUF Support  │  │ - Message    │ │
│  │ - Download      │  │ - Temperature   │  │   History    │ │
│  │ - Validation    │  │ - Token Limits  │  │ - System     │ │
│  │ - Caching       │  │ - Generation    │  │   Prompts    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Storage Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Model Storage   │  │ Configuration   │                  │
│  │ ~/.rustllm/     │  │ - Model Registry│                  │
│  │ ├── models/     │  │ - Download URLs │                  │
│  │ └── cache/      │  │ - Checksums     │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Flow Diagram

```
User Request
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ CLI/API     │───▶│ Request      │───▶│ Model       │
│ Interface   │    │ Router       │    │ Manager     │
└─────────────┘    └──────────────┘    └─────────────┘
     │                    │                    │
     │                    ▼                    ▼
     │             ┌──────────────┐    ┌─────────────┐
     │             │ Input        │    │ Model       │
     │             │ Validation   │    │ Loading     │
     │             └──────────────┘    └─────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Response    │◀───│ Inference    │◀───│ Context     │
│ Formatting  │    │ Engine       │    │ Building    │
└─────────────┘    └──────────────┘    └─────────────┘
     │
     ▼
  User Output
```

### Data Flow

#### Chat Request Flow
```
1. User Input
   ├── CLI: Interactive prompt
   └── API: HTTP POST /api/chat

2. Request Processing
   ├── Parse message content
   ├── Validate model name
   ├── Load model (if not cached)
   └── Build chat context

3. Inference
   ├── Format prompt with context
   ├── Apply temperature/token settings
   ├── Generate response tokens
   └── Decode to text

4. Response
   ├── CLI: Stream to terminal
   └── API: JSON response with usage stats
```

#### Model Management Flow
```
1. Model Download
   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
   │ Request     │───▶│ Registry     │───▶│ Download    │
   │ Model Name  │    │ Lookup       │    │ Manager     │
   └─────────────┘    └──────────────┘    └─────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┼──────┐
                              ▼                   ▼      │
                     ┌──────────────┐    ┌─────────────┐ │
                     │ URL & Hash   │    │ HTTP        │ │
                     │ Validation   │    │ Download    │ │
                     └──────────────┘    └─────────────┘ │
                              │                   │      │
                              └───────────────────┼──────┘
                                                  ▼
                                         ┌─────────────┐
                                         │ File        │
                                         │ Storage     │
                                         │ + Checksum  │
                                         └─────────────┘
```

### Module Architecture

#### Core Modules
```
src/
├── main.rs              # Application entry point
├── cli/                 # Command-line interface
│   ├── mod.rs          # CLI orchestration
│   └── model_commands.rs # Model management commands
├── server/              # REST API server
│   └── mod.rs          # HTTP handlers and routing
├── model/               # Model management and inference
│   ├── mod.rs          # Model operations
│   ├── download.rs     # Model downloading logic
│   └── inference.rs    # GGUF inference engine
└── utils/               # Shared utilities
    └── mod.rs          # Helper functions
```

#### Dependency Graph
```
┌─────────────┐    ┌─────────────┐
│   main.rs   │───▶│   cli/      │
└─────────────┘    └─────────────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│  server/    │    │   model/    │
└─────────────┘    └─────────────┘
       │                  │
       └──────────────────┘
                  │
                  ▼
           ┌─────────────┐
           │   utils/    │
           └─────────────┘
```

### Performance Considerations

#### Model Caching Strategy
```
Memory Management:
┌─────────────────────────────────────┐
│ Model Cache (LRU)                   │
├─────────────────────────────────────┤
│ Hot Models    │ Memory Usage: High  │
│ (Recently     │ Access Time: ~1ms   │
│  Used)        │                     │
├─────────────────────────────────────┤
│ Warm Models   │ Memory Usage: Med   │
│ (Cached but   │ Access Time: ~100ms │
│  Idle)        │                     │
├─────────────────────────────────────┤
│ Cold Models   │ Memory Usage: Low   │
│ (On Disk)     │ Access Time: ~5s    │
└─────────────────────────────────────┘
```

#### Concurrency Model
```
┌─────────────────────────────────────┐
│ Server Architecture                 │
├─────────────────────────────────────┤
│ Axum HTTP Server                    │
│ ├── Request Handler (async)         │
│ ├── Model Pool (Arc<Mutex<Model>>) │
│ └── Connection Pool                 │
├─────────────────────────────────────┤
│ Each request gets:                  │
│ ├── Isolated chat context          │
│ ├── Shared model instance          │
│ └── Independent response stream     │
└─────────────────────────────────────┘
```

### Security & Reliability

#### Error Handling Strategy
```
Error Propagation:
User Request
     │
     ▼ (Result<T, Error>)
Request Validation
     │
     ▼ (Result<T, ModelError>)
Model Loading
     │
     ▼ (Result<T, InferenceError>)
Inference Engine
     │
     ▼ (Result<Response, Error>)
Response Formatting
     │
     ▼
User Response (Success/Error)
```

#### Model Validation
```
Download Process:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ SHA256      │───▶│ File Size    │───▶│ GGUF        │
│ Checksum    │    │ Validation   │    │ Format      │
│ Verify      │    │              │    │ Validation  │
└─────────────┘    └──────────────┘    └─────────────┘
       │                  │                    │
       └──── Fail ────────┼────── Fail ────────┘
                          │
                          ▼
                 ┌─────────────┐
                 │ Model Ready │
                 │ for Use     │
                 └─────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/muthuspark/rustllm.git
cd rustllm

# Build the project
cargo build --release

# Run the binary
./target/release/rustllm --help
```

## Usage

### Chat with a model

```bash
# Start a chat session with a model
rustllm chat --model llama2-7b
```

### Download a model

```bash
# Download a pre-configured model
rustllm model pull llama2-7b
```

### List available models

```bash
# List all downloaded models
rustllm model list
```

### Start the API server

```bash
# Start the API server on localhost:8000
rustllm serve
```

## Available Models

- llama2-7b (Llama 2 7B quantized to 4-bit)
- mistral-7b (Mistral 7B quantized to 4-bit)
- phi-2 (Phi-2 quantized to 4-bit)
- neural-chat-7b (Neural Chat 7B v3.1 quantized to 4-bit)

## API Usage

The RustLLM server provides a REST API compatible with OpenAI's chat completions format. Start the server with:

```bash
rustllm serve --port 8000
```

### API Endpoints

#### Health Check
Check if the server is running:

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "success": true,
  "data": "OK"
}
```

#### List Models
Get all available models:

```bash
curl http://localhost:8000/api/models
```

Response:
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "llama2-7b.Q4_K_M.gguf",
        "size_bytes": 4368438272,
        "last_modified": "2024-01-15T10:30:00Z"
      }
    ]
  }
}
```

#### Get Model Info
Get information about a specific model:

```bash
curl http://localhost:8000/api/models/llama2-7b
```

Response:
```json
{
  "success": true,
  "data": {
    "name": "llama2-7b.Q4_K_M.gguf",
    "size_bytes": 4368438272,
    "last_modified": "2024-01-15T10:30:00Z"
  }
}
```

#### Download Model
Download a model from the registry:

```bash
curl -X POST http://localhost:8000/api/models/llama2-7b \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

Response:
```json
{
  "success": true,
  "data": "Model llama2-7b downloaded successfully"
}
```

#### Delete Model
Remove a model from local storage:

```bash
curl -X DELETE http://localhost:8000/api/models/llama2-7b
```

Response:
```json
{
  "success": true,
  "data": "Model llama2-7b deleted successfully"
}
```

#### Chat Completion
Generate a chat response (OpenAI-compatible format):

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "message": {
      "role": "assistant",
      "content": "The capital of France is Paris. It is located in the north-central part of the country and is known for its rich history, culture, and iconic landmarks like the Eiffel Tower."
    },
    "usage": {
      "prompt_tokens": 25,
      "completion_tokens": 32,
      "total_tokens": 57
    }
  }
}
```

#### Multi-turn Conversation
Continue a conversation by including previous messages:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      },
      {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      {
        "role": "user",
        "content": "What are some famous landmarks there?"
      }
    ],
    "temperature": 0.7
  }'
```

#### Streaming Chat (Not Yet Implemented)
Stream responses in real-time:

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a story"
      }
    ],
    "stream": true
  }'
```

### Python Example

Here's how to use the API with Python:

```python
import requests
import json

# Base URL for the API
base_url = "http://localhost:8000/api"

# Check if server is running
health = requests.get(f"{base_url}/health")
print("Server status:", health.json())

# List available models
models = requests.get(f"{base_url}/models")
print("Available models:", models.json())

# Chat with a model
chat_request = {
    "model": "llama2-7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 200
}

response = requests.post(f"{base_url}/chat", json=chat_request)
result = response.json()

if result["success"]:
    print("Assistant:", result["data"]["message"]["content"])
    print("Token usage:", result["data"]["usage"])
else:
    print("Error:", result["error"])
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const baseURL = 'http://localhost:8000/api';

async function chatWithModel() {
  try {
    // Chat request
    const response = await axios.post(`${baseURL}/chat`, {
      model: 'llama2-7b',
      messages: [
        { role: 'user', content: 'Write a haiku about programming' }
      ],
      temperature: 0.8,
      max_tokens: 50
    });

    console.log('Response:', response.data.data.message.content);
    console.log('Tokens used:', response.data.data.usage.total_tokens);
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

chatWithModel();
```

### Error Handling

All API endpoints return a consistent error format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (model doesn't exist)
- `500` - Internal Server Error
- `501` - Not Implemented (streaming endpoints)

## Configuration

Models are stored in `~/.rustllm/models` by default. You can specify a custom path with the `--models-path` option.

## License

[MIT License](LICENSE)