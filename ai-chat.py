#!/usr/bin/env python3
"""
AI/LLM CLI Chat Tool for Arch Linux and Linux Mint

A command-line interface for chatting with AI/LLMs from Hugging Face.
Supports downloading models and interactive chat sessions.

Usage:
    ai-chat -Df "https://huggingface.co/model_name"  # Download a model
    ai-chat -c "model_name"                          # Chat with a model
    ai-chat                                          # Interactive mode
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, list_models
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install dependencies: pip install huggingface_hub transformers torch accelerate")
    sys.exit(1)


class AIChatCLI:
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "ai-chat"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_model = None
        self.tokenizer = None
        self.model = None
        self.generator = None

    def download_model(self, model_id: str) -> bool:
        """Download a model from Hugging Face."""
        print(f"Downloading model: {model_id}")
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                resume_download=True
            )
            print(f"✓ Model downloaded successfully to: {local_path}")
            return True
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            return False

    def load_model(self, model_id: str) -> bool:
        """Load a model for chatting."""
        print(f"Loading model: {model_id}")
        try:
            model_path = self.cache_dir / model_id.replace("/", "--")
            
            if not model_path.exists():
                print(f"Model not found locally. Attempting to download...")
                if not self.download_model(model_id):
                    return False
            
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            print("Loading model (this may take a while)...")
            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                device_map=device_map,
                torch_dtype=dtype if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            self.current_model = model_id
            print(f"✓ Model loaded successfully: {model_id}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False

    def chat(self, message: str, conversation_history: list = None) -> str:
        """Generate a response to a message."""
        if not self.generator:
            return "Error: No model loaded. Use -Df to download and -c to load a model first."
        
        try:
            if conversation_history:
                prompt = "\n".join(conversation_history) + "\nUser: " + message
            else:
                prompt = f"User: {message}"
            
            prompt += "\nAssistant:"
            
            response = self.generator(prompt)[0]['generated_text']
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"

    def interactive_chat(self, model_id: str = None):
        """Start an interactive chat session."""
        if model_id:
            if not self.load_model(model_id):
                return
        elif not self.current_model:
            print("No model loaded. Use -Df to download a model first.")
            print("Example: ai-chat -Df \"microsoft/phi-2\"")
            return
        
        print("\n" + "="*60)
        print("AI Chat Interface")
        print("="*60)
        print(f"Current model: {self.current_model}")
        print("Commands:")
        print("  /quit, /exit, /q - Exit the chat")
        print("  /model <id>     - Load a different model")
        print("  /clear          - Clear conversation history")
        print("="*60 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("Goodbye!")
                    break
                
                if user_input.startswith('/model '):
                    new_model = user_input[7:].strip()
                    if self.load_model(new_model):
                        conversation_history = []
                        print(f"Switched to model: {new_model}")
                    continue
                
                if user_input == '/clear':
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                
                response = self.chat(user_input, conversation_history)
                print(f"\nAssistant: {response}\n")
                
                conversation_history.append(f"User: {user_input}")
                conversation_history.append(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
                continue
            except EOFError:
                print("\nGoodbye!")
                break

    def one_shot_chat(self, model_id: str, message: str):
        """Single question-answer interaction."""
        if not self.load_model(model_id):
            return
        
        response = self.chat(message)
        print(f"\n{response}\n")


def parse_hf_url(url: str) -> str:
    """Extract model ID from Hugging Face URL."""
    if url.startswith("https://huggingface.co/"):
        return url.replace("https://huggingface.co/", "").split("/")[0]
    return url


def main():
    parser = argparse.ArgumentParser(
        description="AI/LLM CLI Chat Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -Df "microsoft/phi-2"                    # Download a model
  %(prog)s -Df "https://huggingface.co/meta-llama/Llama-2-7b"  # Download from URL
  %(prog)s -c "microsoft/phi-2"                     # Chat with a model
  %(prog)s -c "microsoft/phi-2" -m "What is AI?"   # One-shot question
  %(prog)s                                          # Interactive mode
        """
    )
    
    parser.add_argument(
        '-Df', '--download',
        metavar='URL_OR_ID',
        help='Download a model from Hugging Face (URL or model ID)'
    )
    
    parser.add_argument(
        '-c', '--chat',
        metavar='MODEL_ID',
        help='Chat with a model (loads existing or downloads if needed)'
    )
    
    parser.add_argument(
        '-m', '--message',
        metavar='TEXT',
        help='Send a single message (use with -c)'
    )
    
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List downloaded models'
    )
    
    args = parser.parse_args()
    
    cli = AIChatCLI()
    
    if args.download:
        model_id = parse_hf_url(args.download)
        cli.download_model(model_id)
    
    elif args.list:
        print("Downloaded models:")
        if cli.cache_dir.exists():
            for item in cli.cache_dir.iterdir():
                if item.is_dir():
                    model_name = item.name.replace("--", "/")
                    print(f"  - {model_name}")
        else:
            print("  No models downloaded yet.")
    
    elif args.chat:
        if args.message:
            cli.one_shot_chat(args.chat, args.message)
        else:
            cli.interactive_chat(args.chat)
    
    else:
        # No arguments - show help or start interactive if model exists
        print("AI Chat CLI - Chat with AI/LLMs from the command line")
        print("\nQuick Start:")
        print("  1. Download a model: ai-chat -Df \"microsoft/phi-2\"")
        print("  2. Start chatting:  ai-chat -c \"microsoft/phi-2\"")
        print("\nOr see all options: ai-chat --help")


if __name__ == "__main__":
    main()
