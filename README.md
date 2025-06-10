# 🎤 Real-Time Audio Transcription System

A sophisticated real-time transcription system that captures audio from your microphone, processes it through OpenAI's Whisper model, and displays live transcription with automatic English translation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Whisper](https://img.shields.io/badge/Whisper-Large-green.svg)
![Real-time](https://img.shields.io/badge/Real--time-Audio-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎙️ **Real-time microphone capture** - Continuous audio streaming with 200ms chunks
- 🧠 **AI-powered transcription** - Uses OpenAI Whisper Large model for high accuracy
- 🌍 **Multi-language support** - Auto-detects languages and translates to English
- 🔊 **Voice Activity Detection** - Intelligently filters out silence and background noise
- 🖥️ **Modern web interface** - Clean, responsive UI built with Gradio
- ⚡ **Low latency processing** - Typically 2-5 seconds from speech to text
- 📊 **Live statistics** - Real-time monitoring of transcription performance
- 🎯 **High accuracy** - Optimized for conversational speech in multiple languages

## 🏗️ Architecture

```
Microphone → Audio Stream → 200ms Chunks → Buffer (2sec) → VAD → Whisper → Display
     ↓              ↓            ↓            ↓         ↓        ↓         ↓
  Real-time    sounddevice   Accumulate    Silence   large    English   Web UI
   Capture                   in queue     Detection  Model   Translation
```

### Core Components

| Component           | Technology    | Purpose                         |
| ------------------- | ------------- | ------------------------------- |
| **Audio Capture**   | sounddevice   | Microphone input streaming      |
| **Audio Buffering** | Python queue  | Chunk accumulation & management |
| **VAD**             | webrtcvad     | Voice activity detection        |
| **ASR**             | Whisper Large | Speech-to-text transcription    |
| **Translation**     | Whisper       | Automatic language translation  |
| **UI**              | Gradio        | Real-time web interface         |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or later
- Microphone (built-in or USB)
- 8GB+ RAM (recommended for Whisper Large)
- Optional: NVIDIA GPU for faster processing

### Installation

1. **Clone and setup:**

```bash
git clone https://github.com/akash-growhut/local-transcription.git
cd local-transcription
chmod +x install.sh
./install.sh
```

2. **Test your audio system:**

```bash
python3 audio_test.py
```

3. **Run the application:**

```bash
# Quick start (medium model, faster startup)
python3 quick_start.py

# Full system (large model, best accuracy)
python3 main.py

# Or use the shortcut:
./run.sh
```

4. **Open your browser:**
   - Quick version: `http://localhost:7861`
   - Full version: `http://localhost:7860`

> **📥 Model Download**: On first run, Whisper models will be automatically downloaded (~800MB-3GB). This may take 2-10 minutes depending on your internet speed. Subsequent runs will be much faster!

## 🎯 Usage

### Web Interface

1. **Start Recording**: Click the red "🔴 Start Recording" button
2. **Speak**: Talk clearly into your microphone
3. **Watch**: Live transcription appears in real-time
4. **Stop**: Click "⏹️ Stop Recording" when done
5. **Clear**: Use "🗑️ Clear History" to reset transcription

### Features in Detail

- **Format**: `[Timestamp] [Language] Transcribed text`
- **Languages**: Auto-detects and shows detected language
- **Translation**: All text automatically translated to English
- **History**: Shows last 15 transcription segments
- **Statistics**: Live stats showing recording status, languages detected, etc.

## ⚙️ Configuration

### Audio Settings (in `main.py`)

```python
# Audio configuration
SAMPLE_RATE = 16000      # Whisper optimal rate
CHUNK_DURATION = 0.2     # 200ms audio chunks
BUFFER_DURATION = 2.0    # 2-second processing buffer
CHANNELS = 1             # Mono audio
```

### Whisper Settings

```python
# Whisper configuration
MODEL_SIZE = "large"     # Options: tiny, base, small, medium, large
TASK = "translate"       # Always translate to English
TEMPERATURE = 0.0        # Deterministic output
```

### VAD Settings

```python
# Voice Activity Detection
VAD_AGGRESSIVENESS = 2   # 0-3, higher = more aggressive
SPEECH_THRESHOLD = 0.3   # Minimum speech ratio to process
```

## 🔧 Technical Details

### Performance Metrics

| Metric                | Target         | Maximum        |
| --------------------- | -------------- | -------------- |
| Audio Capture Latency | <50ms          | <100ms         |
| Buffer Accumulation   | 2 seconds      | 3 seconds      |
| Whisper Processing    | <2 seconds     | <5 seconds     |
| **Total Latency**     | **<3 seconds** | **<6 seconds** |

### Accuracy Targets

- **English Speech**: >90% accuracy
- **Other Languages**: >85% accuracy
- **Language Detection**: >95% accuracy
- **Translation Quality**: Conversational level

### System Requirements

**Minimum:**

- 4-core CPU, 2.5GHz+
- 8GB RAM
- Built-in microphone
- Python 3.8+

**Recommended:**

- 8-core CPU, 3.0GHz+
- 16GB RAM
- NVIDIA GPU (6GB+ VRAM)
- High-quality USB microphone

## 🧪 Testing

### Audio System Test

```bash
python3 audio_test.py
```

This comprehensive test will:

- List all available audio devices
- Test microphone capture and playback
- Verify real-time audio streaming
- Check audio quality and levels

### Manual Testing Checklist

- [ ] Microphone detection works
- [ ] Audio capture streams correctly
- [ ] VAD detects speech vs silence
- [ ] Whisper processes audio successfully
- [ ] UI updates in real-time
- [ ] Multiple languages are detected
- [ ] Translation to English works
- [ ] Long sessions (30+ minutes) are stable

## 🐛 Troubleshooting

### Common Issues

**No audio devices found:**

```bash
# macOS
brew install portaudio

# Linux (Ubuntu/Debian)
sudo apt-get install portaudio19-dev

# Check available devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Low audio quality:**

- Check microphone positioning (6-12 inches from mouth)
- Reduce background noise
- Test with different microphones
- Adjust system audio input levels

**High latency:**

- Close other audio applications
- Reduce buffer size (may cause audio dropouts)
- Use a faster computer or GPU acceleration
- Switch to smaller Whisper model (medium/small)

**Memory issues:**

- Close other applications
- Use smaller Whisper model
- Increase system swap space
- Consider using CPU-only PyTorch

### Debug Mode

Enable detailed logging:

```python
# In main.py, change logging level
logging.basicConfig(level=logging.DEBUG)
```

## 🛣️ Roadmap

### Phase 1 (Current)

- ✅ Real-time transcription
- ✅ Multi-language support
- ✅ Voice activity detection
- ✅ Web interface

### Phase 2 (Planned)

- [ ] Speaker diarization (multi-speaker support)
- [ ] Custom model fine-tuning
- [ ] WebSocket streaming API
- [ ] Mobile app companion
- [ ] Cloud deployment options

### Phase 3 (Future)

- [ ] Real-time translation between languages
- [ ] Integration with meeting platforms
- [ ] Advanced noise cancellation
- [ ] Custom wake word detection

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repo
git clone <your-repo>
cd local-transcription

# Create development environment
python3 -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev tools

# Run tests
python3 -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the incredible Whisper model
- **Google** for WebRTC VAD
- **Gradio team** for the excellent UI framework
- **PyAudio/sounddevice** maintainers for audio capture tools

## 📞 Support

- 📋 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📧 **Email**: your-email@example.com

---

**Built with ❤️ for the open-source community**

_Real-time transcription has never been this accessible!_ 🎤✨
