# AI Chat CLI

A command-line interface for chatting with AI/LLMs from Hugging Face. Works on Arch Linux, Linux Mint, and other Linux distributions.

## Features

- Download models directly from Hugging Face
- Interactive chat sessions with conversation history
- One-shot question/answer mode
- Support for any Hugging Face causal language model
- Automatic model caching
- CUDA GPU acceleration support (if available)

## Quick Start

### Installation

1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd ai-chat
   ```

2. Run the installation script:
   ```bash
   ./install.sh
   ```

   Or manually install dependencies:
   ```bash
   pip install huggingface_hub transformers torch accelerate
   ```

3. Make sure `~/.local/bin` is in your PATH, or run the script directly:
   ```bash
   python3 ai-chat.py --help
   ```

### Usage

#### Download a Model

```bash
# Using model ID
ai-chat -Df "microsoft/phi-2"

# Using full Hugging Face URL
ai-chat -Df "https://huggingface.co/microsoft/phi-2"
```

#### Start Chatting

```bash
# Interactive chat session
ai-chat -c "microsoft/phi-2"

# Single question/answer
ai-chat -c "microsoft/phi-2" -m "What is artificial intelligence?"
```

#### List Downloaded Models

```bash
ai-chat -l
```

## Command Reference

| Command | Description |
|---------|-------------|
| `-Df <URL_OR_ID>` | Download a model from Hugging Face |
| `-c <MODEL_ID>` | Chat with a model (interactive mode) |
| `-m <TEXT>` | Send a single message (use with -c) |
| `-l` | List all downloaded models |
| `--help` | Show help message |

## Interactive Mode Commands

While in interactive chat mode, you can use these commands:

- `/quit`, `/exit`, `/q` - Exit the chat
- `/model <id>` - Load a different model
- `/clear` - Clear conversation history

## Examples

### Example 1: Download and Chat with Phi-2

```bash
# Download the model
ai-chat -Df "microsoft/phi-2"

# Start chatting
ai-chat -c "microsoft/phi-2"
```

### Example 2: One-Shot Question

```bash
ai-chat -c "microsoft/phi-2" -m "Explain quantum computing in simple terms"
```

### Example 3: Using Full URL

```bash
ai-chat -Df "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
ai-chat -c "Qwen/Qwen2.5-0.5B-Instruct"
```

## Recommended Models

Here are some lightweight models that work well for CLI chatting:

- **microsoft/phi-2** - Small but capable (2.7B parameters)
- **Qwen/Qwen2.5-0.5B-Instruct** - Very small, fast inference
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - Lightweight chat model
- **google/gemma-2b** - Google's lightweight model

## Requirements

- Python 3.8+
- pip
- Required Python packages:
  - huggingface_hub
  - transformers
  - torch
  - accelerate

## System Requirements

- **Minimum**: 4GB RAM (for smaller models < 1B parameters)
- **Recommended**: 8GB+ RAM (for models 1-3B parameters)
- **GPU**: Optional but recommended for faster inference (CUDA support automatic)

## Troubleshooting

### Out of Memory Error

Try using a smaller model or ensure you have enough RAM. You can also try running with CPU-only mode by setting:
```bash
export CUDA_VISIBLE_DEVICES=""
```

### Model Download Fails

Check your internet connection and try again. The tool supports resume downloads, so it will continue from where it left off.

### Slow Inference

Consider using a smaller model or enabling GPU acceleration if available.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
