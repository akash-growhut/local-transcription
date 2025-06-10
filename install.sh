#!/bin/bash

# Real-Time Whisper Transcription Setup Script
# Sets up Python environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Real-Time Whisper Transcription System Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Not in a virtual environment. Creating one..."
    
    # Create virtual environment
    python3 -m venv whisper-realtime-env
    echo "✅ Virtual environment created: whisper-realtime-env"
    
    # Activate virtual environment
    source whisper-realtime-env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (for better compatibility)
echo ""
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS - installing PyTorch with Metal support"
    pip install torch torchaudio
elif command -v nvidia-smi &> /dev/null; then
    # CUDA available
    echo "CUDA detected - installing PyTorch with CUDA support"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # CPU only
    echo "Installing PyTorch (CPU version)"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install system-specific audio dependencies
echo ""
echo "Installing system audio dependencies..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux - checking for audio libraries..."
    
    # Check if we can install system packages
    if command -v apt-get &> /dev/null; then
        echo "Installing Linux audio dependencies with apt..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
    elif command -v yum &> /dev/null; then
        echo "Installing Linux audio dependencies with yum..."
        sudo yum install -y portaudio-devel
    else
        echo "⚠️  Could not determine package manager. You may need to install portaudio manually."
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - checking for Homebrew..."
    
    if command -v brew &> /dev/null; then
        echo "Installing macOS audio dependencies with Homebrew..."
        brew install portaudio
    else
        echo "⚠️  Homebrew not found. Please install portaudio manually:"
        echo "   brew install portaudio"
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download Whisper model (this will cache it for faster startup)
echo ""
echo "Pre-downloading Whisper model (this may take a few minutes)..."
python3 -c "
import whisper
print('Downloading Whisper large model...')
model = whisper.load_model('large')
print('✅ Whisper model downloaded and cached!')
"

# Test audio system
echo ""
echo "Testing audio system..."
if python3 -c "import sounddevice as sd; print('✅ Audio system OK')" 2>/dev/null; then
    echo "✅ Audio dependencies installed successfully"
else
    echo "⚠️  Audio system test failed. You may need to install additional dependencies."
fi

# Test webrtcvad
echo ""
echo "Testing Voice Activity Detection..."
if python3 -c "import webrtcvad; print('✅ VAD system OK')" 2>/dev/null; then
    echo "✅ VAD dependencies installed successfully"
else
    echo "⚠️  VAD system test failed."
fi

# Create startup script
echo ""
echo "Creating startup script..."
cat > run.sh << 'EOF'
#!/bin/bash

# Activate virtual environment if it exists
if [ -d "whisper-realtime-env" ]; then
    source whisper-realtime-env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run the application
echo "🚀 Starting Real-Time Whisper Transcription..."
python3 main.py
EOF

chmod +x run.sh
echo "✅ Created run.sh startup script"

# Installation complete
echo ""
echo "=================================================="
echo "🎉 INSTALLATION COMPLETE!"
echo "=================================================="
echo ""
echo "To get started:"
echo "1. Test your audio system:  python3 audio_test.py"
echo "2. Run the application:     python3 main.py"
echo "   or use the shortcut:     ./run.sh"
echo ""
echo "The web interface will be available at: http://localhost:7860"
echo ""
echo "Hardware recommendations:"
echo "- Use a good quality microphone for best results"
echo "- Ensure sufficient RAM (8GB+) for the Whisper large model"
echo "- GPU acceleration will improve performance but is not required"
echo ""
echo "Happy transcribing! 🎤✨" 