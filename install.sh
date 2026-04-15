#!/bin/bash
#
# AI-CHAT-PRO Installation Script
# Supports: Arch Linux, Linux Mint (Ubuntu-based)
# Features: Auto-detect distro, install deps, setup PATH
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/bin"
VENV_DIR="$HOME/.ai-chat-pro-venv"

echo "=========================================="
echo "  AI-CHAT-PRO Installer"
echo "=========================================="

# Detect Distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    echo "[INFO] Detected OS: $PRETTY_NAME"
else
    echo "[ERROR] Cannot detect distribution. Exiting."
    exit 1
fi

# Install System Dependencies
install_deps_arch() {
    echo "[INFO] Installing Arch Linux dependencies..."
    sudo pacman -Sy --noconfirm python python-pip cuda-toolkit cudnn nccl
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[WARN] NVIDIA drivers may not be installed. Please install nvidia-utils."
    fi
}

install_deps_mint() {
    echo "[INFO] Installing Linux Mint/Ubuntu dependencies..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
    # CUDA installation on Ubuntu/Mint is complex, usually done via runfile or deb
    echo "[INFO] For CUDA support on Mint, ensure you have NVIDIA drivers installed:"
    echo "       sudo ubuntu-drivers autoinstall"
}

case $DISTRO in
    arch|manjaro|endeavouros)
        install_deps_arch
        ;;
    linuxmint|ubuntu|debian|pop)
        install_deps_mint
        ;;
    *)
        echo "[WARN] Unknown distro '$DISTRO'. Attempting generic install."
        ;;
esac

# Setup Virtual Environment
echo ""
echo "[INFO] Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install Python Dependencies
echo ""
echo "[INFO] Installing Python packages (this may take a while)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
pip install torch torchvision torchaudio  # Fallback to CPU if CUDA fails

pip install transformers accelerate sentencepiece protobuf
pip install huggingface_hub
pip install pynvml  # For detailed GPU monitoring

# Create bin directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Copy script and make executable
echo ""
echo "[INFO] Installing ai-chat-pro to $INSTALL_DIR..."
cp "$SCRIPT_DIR/ai-chat-pro.py" "$INSTALL_DIR/ai-chat-pro.py"
chmod +x "$INSTALL_DIR/ai-chat-pro.py"

# Create wrapper script to activate venv automatically
cat > "$INSTALL_DIR/ai-chat-pro" << 'WRAPPER'
#!/bin/bash
VENV_DIR="$HOME/.ai-chat-pro-venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi
exec python3 "$HOME/.local/bin/ai-chat-pro.py" "$@"
WRAPPER

chmod +x "$INSTALL_DIR/ai-chat-pro"

# Add to PATH if not already present
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "[INFO] Adding $HOME/.local/bin to PATH..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
    echo "[ACTION] Please run: source ~/.bashrc  (or restart your terminal)"
fi

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Usage Examples:"
echo "  ai-chat-pro --info                    # Check GPU/CUDA status"
echo "  ai-chat-pro -s mistral                # Search for models"
echo "  ai-chat-pro -rl                       # List recommended models"
echo "  ai-chat-pro -rw                       # Download weekly recommended"
echo "  ai-chat-pro -r 2                      # Download model by ID"
echo "  ai-chat-pro -Df microsoft/phi-2       # Download specific model"
echo "  ai-chat-pro -c microsoft/phi-2        # Start chat"
echo "  ai-chat-pro -c phi-2 -m \"Hello\"       # One-shot question"
echo ""
echo "Interactive Commands:"
echo "  /stats, /config, /benchmark, /export, /system, /help"
echo ""
echo "Note: First run will download model weights (~GBs)."
echo "=========================================="
