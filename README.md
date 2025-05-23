# RustLLM

A Rust-based local LLM chat tool similar to Ollama.

## Features

- Run local LLMs directly on your machine
- Interactive CLI chat interface
- REST API server for integration with other applications
- Model management (download, list, delete)
- Support for GGUF format models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rustllm.git
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

## API Endpoints

- `GET /api/models` - List available models
- `GET /api/models/:model_name` - Get model info
- `POST /api/models/:model_name` - Download a model
- `DELETE /api/models/:model_name` - Delete a model
- `POST /api/chat` - Generate a chat completion
- `POST /api/chat/stream` - Stream a chat completion
- `GET /api/health` - Health check

## Configuration

Models are stored in `~/.rustllm/models` by default. You can specify a custom path with the `--models-path` option.

## License

[MIT License](LICENSE)