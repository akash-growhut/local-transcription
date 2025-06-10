# Technical Requirements Document: Real-Time Audio Transcription System

**Project:** Real-Time Microphone Transcription with Whisper  
**Version:** 1.0 (Foundation Phase)  
**Date:** 11 Jun 2025
**Scope:** MVP - Single speaker, real-time transcription

---

## 1. Executive Summary

Building a foundational real-time transcription system that captures audio from microphone, processes it through Whisper, and displays live transcription with English translation. This serves as the foundation for more complex multi-speaker systems.

---

## 2. Project Scope (Simplified)

### 2.1 Core Functionality

- ✅ **Real-time audio capture** from microphone
- ✅ **Live transcription display** (streaming text)
- ✅ **Voice Activity Detection** (VAD)
- ✅ **Translation to English** (any language input)
- ✅ **Chunk-based processing** (200ms → 1-3sec buffers)

### 2.2 Out of Scope (For Now)

- ❌ Multiple speakers
- ❌ Separate audio tracks
- ❌ File-based processing
- ❌ Speaker diarization
- ❌ Cloud deployment

---

## 3. Technical Architecture

### 3.1 System Flow

```
Microphone → Audio Stream → 200ms Chunks → Buffer (1-3sec) → VAD → Whisper → Display
     ↓              ↓            ↓             ↓           ↓        ↓         ↓
  Real-time    PyAudio/     Accumulate    Silence      large     English   Web UI/
   Capture    sounddevice    in queue     Detection    Model    Translation Console
```

### 3.2 Core Components

| Component           | Technology           | Purpose                    |
| ------------------- | -------------------- | -------------------------- |
| **Audio Capture**   | PyAudio/sounddevice  | Microphone input streaming |
| **Audio Buffering** | Python queue         | Chunk accumulation         |
| **VAD**             | webrtcvad/silero-vad | Detect speech segments     |
| **ASR**             | Whisper large        | Speech-to-text             |
| **Display**         | Gradio/Streamlit     | Real-time UI               |

---

## 4. Detailed Technical Specifications

### 4.1 Audio Processing Pipeline

#### 4.1.1 Audio Capture Settings

```python
AUDIO_CONFIG = {
    "sample_rate": 16000,      # Whisper optimal rate
    "channels": 1,             # Mono
    "chunk_duration": 0.2,     # 200ms chunks
    "buffer_duration": 2.0,    # 1-3 second accumulation
    "format": "int16"          # 16-bit audio
}
```

#### 4.1.2 Buffering Strategy

```python
class AudioBuffer:
    def __init__(self):
        self.chunk_size = int(16000 * 0.2)  # 200ms at 16kHz
        self.buffer_size = int(16000 * 2.0)  # 2 second buffer
        self.audio_queue = queue.Queue()
        self.current_buffer = []

    def add_chunk(self, chunk):
        """Add 200ms chunk to buffer"""
        self.current_buffer.extend(chunk)

        if len(self.current_buffer) >= self.buffer_size:
            self.process_buffer()
```

### 4.2 Voice Activity Detection

#### 4.2.1 VAD Implementation

```python
import webrtcvad

class VADProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        self.frame_duration = 20     # 20ms frames for VAD

    def is_speech(self, audio_chunk):
        """Detect if chunk contains speech"""
        # Convert to proper format for VAD
        frames = self.chunk_to_frames(audio_chunk)
        speech_frames = sum(self.vad.is_speech(frame, 16000) for frame in frames)

        # Return True if >50% frames contain speech
        return speech_frames > len(frames) * 0.5
```

### 4.3 Whisper Integration

#### 4.3.1 Model Configuration

```python
import whisper

class WhisperProcessor:
    def __init__(self):
        self.model = whisper.load_model("large")
        self.config = {
            "task": "translate",        # Always translate to English
            "language": None,           # Auto-detect
            "temperature": 0.0,         # Deterministic
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0
        }

    def transcribe_chunk(self, audio_buffer):
        """Process 1-3 second audio buffer"""
        if not self.is_valid_audio(audio_buffer):
            return None

        result = self.model.transcribe(
            audio_buffer,
            **self.config
        )

        return result["text"].strip()
```

### 4.4 Real-Time Display System

#### 4.4.1 Gradio Interface

```python
import gradio as gr
import threading

class RealtimeTranscriptionUI:
    def __init__(self):
        self.transcription_text = ""
        self.is_recording = False

    def create_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown("# Real-Time Transcription")

            with gr.Row():
                start_btn = gr.Button("Start Recording")
                stop_btn = gr.Button("Stop Recording")
                clear_btn = gr.Button("Clear")

            transcription_output = gr.Textbox(
                label="Live Transcription",
                lines=10,
                max_lines=20,
                value="",
                interactive=False
            )

            # Event handlers
            start_btn.click(self.start_recording)
            stop_btn.click(self.stop_recording)
            clear_btn.click(self.clear_transcription)

        return interface
```

---

## 5. Implementation Details

### 5.1 Main Processing Loop

```python
import asyncio
import threading
from collections import deque

class RealtimeTranscriber:
    def __init__(self):
        self.audio_buffer = AudioBuffer()
        self.vad_processor = VADProcessor()
        self.whisper_processor = WhisperProcessor()
        self.ui = RealtimeTranscriptionUI()

        self.is_running = False
        self.transcription_queue = queue.Queue()

    def audio_callback(self, indata, frames, time, status):
        """Called every 200ms with new audio data"""
        if status:
            print(f"Audio callback error: {status}")

        # Add chunk to buffer
        audio_chunk = indata.flatten()
        self.audio_buffer.add_chunk(audio_chunk)

    def processing_thread(self):
        """Background thread for Whisper processing"""
        while self.is_running:
            try:
                # Get accumulated buffer (1-3 seconds)
                audio_data = self.audio_buffer.get_buffer()

                if audio_data is None:
                    continue

                # Check for speech activity
                if not self.vad_processor.is_speech(audio_data):
                    continue

                # Transcribe with Whisper
                text = self.whisper_processor.transcribe_chunk(audio_data)

                if text:
                    self.transcription_queue.put(text)

            except Exception as e:
                print(f"Processing error: {e}")

    def start_recording(self):
        """Start real-time transcription"""
        self.is_running = True

        # Start audio stream
        self.audio_stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=int(16000 * 0.2)  # 200ms chunks
        )

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.processing_thread
        )

        self.audio_stream.start()
        self.processing_thread.start()
```

### 5.2 Complete Application Structure

```python
# main.py
import sounddevice as sd
import numpy as np
import whisper
import webrtcvad
import gradio as gr
import queue
import threading
import time

class RealtimeWhisperApp:
    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model("large")
        print("Model loaded!")

        self.vad = webrtcvad.Vad(2)
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.is_recording = False
        self.transcription_history = []

    def audio_callback(self, indata, frames, time, status):
        """Audio input callback - called every 200ms"""
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def process_audio_worker(self):
        """Background worker for audio processing"""
        buffer = []
        buffer_duration = 0
        target_duration = 2.0  # 2 seconds

        while True:
            try:
                # Get audio chunk (200ms)
                chunk = self.audio_queue.get(timeout=1)
                buffer.extend(chunk.flatten())
                buffer_duration += 0.2

                # Process when we have enough audio
                if buffer_duration >= target_duration:
                    self.process_buffer(np.array(buffer))
                    buffer = []
                    buffer_duration = 0

            except queue.Empty:
                # Process partial buffer if recording stopped
                if not self.is_recording and buffer:
                    self.process_buffer(np.array(buffer))
                    buffer = []
                    buffer_duration = 0
            except Exception as e:
                print(f"Worker error: {e}")

    def process_buffer(self, audio_buffer):
        """Process accumulated audio buffer"""
        try:
            # Basic VAD check
            if self.has_speech(audio_buffer):
                # Transcribe with Whisper
                result = self.model.transcribe(
                    audio_buffer,
                    task="translate",  # Always translate to English
                    language=None,     # Auto-detect
                    temperature=0.0
                )

                text = result["text"].strip()
                if text:
                    self.result_queue.put({
                        "timestamp": time.time(),
                        "text": text,
                        "language": result.get("language", "unknown")
                    })
        except Exception as e:
            print(f"Processing error: {e}")

    def has_speech(self, audio_buffer):
        """Simple speech detection"""
        # Convert to int16 for webrtcvad
        audio_int16 = (audio_buffer * 32767).astype(np.int16)

        # Check energy level
        energy = np.sqrt(np.mean(audio_int16**2))
        return energy > 100  # Simple threshold

    def start_recording(self):
        """Start recording and transcription"""
        self.is_recording = True
        return "🔴 Recording... Speak now!"

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        return "⏹️ Recording stopped"

    def get_transcription(self):
        """Get latest transcription results"""
        new_text = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                new_text.append(f"[{result['language']}] {result['text']}")
                self.transcription_history.extend(new_text)
            except queue.Empty:
                break

        return "\n".join(self.transcription_history[-10:])  # Last 10 results

    def clear_transcription(self):
        """Clear transcription history"""
        self.transcription_history = []
        # Clear any pending results
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        return ""

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Real-Time Whisper Transcription") as interface:
            gr.Markdown("# 🎤 Real-Time Audio Transcription")
            gr.Markdown("Speak into your microphone and see live transcription!")

            with gr.Row():
                start_btn = gr.Button("🔴 Start Recording", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Recording", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            status_text = gr.Textbox(
                label="Status",
                value="Ready to record",
                interactive=False
            )

            transcription_output = gr.Textbox(
                label="Live Transcription",
                lines=8,
                max_lines=15,
                value="",
                interactive=False,
                placeholder="Transcription will appear here..."
            )

            # Event handlers
            start_btn.click(
                fn=self.start_recording,
                outputs=status_text
            )

            stop_btn.click(
                fn=self.stop_recording,
                outputs=status_text
            )

            clear_btn.click(
                fn=self.clear_transcription,
                outputs=transcription_output
            )

            # Auto-refresh transcription every 500ms
            interface.load(
                fn=self.get_transcription,
                outputs=transcription_output,
                every=0.5
            )

        return interface

    def run(self):
        """Start the application"""
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=int(16000 * 0.2),  # 200ms chunks
            dtype=np.float32
        )

        # Start background worker
        self.worker_thread = threading.Thread(
            target=self.process_audio_worker,
            daemon=True
        )

        print("Starting audio stream...")
        self.stream.start()
        self.worker_thread.start()

        # Create and launch interface
        interface = self.create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

if __name__ == "__main__":
    app = RealtimeWhisperApp()
    app.run()
```

---

## 6. Hardware Requirements

### 6.1 Minimum Specifications

- **CPU:** 4 cores, 2.5GHz+
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** NVIDIA GPU with 6GB+ VRAM (GTX 1060 or better)
- **Storage:** 10GB free space
- **Microphone:** Any USB/built-in microphone

### 6.2 Software Requirements

```txt
# requirements.txt
openai-whisper>=20231106
torch>=2.0.0
torchaudio>=2.0.0
sounddevice>=0.4.6
webrtcvad>=2.0.10
gradio>=4.0.0
numpy>=1.21.0
```

---

## 7. Installation & Setup

### 7.1 Environment Setup

```bash
# Create virtual environment
python -m venv whisper-realtime
source whisper-realtime/bin/activate  # Linux/Mac
# whisper-realtime\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Test microphone
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 7.2 Quick Start

```bash
# Run the application
python main.py

# Open browser to http://localhost:7860
# Click "Start Recording" and speak!
```

---

## 8. Performance Targets

### 8.1 Latency Requirements

| Component               | Target      | Maximum    |
| ----------------------- | ----------- | ---------- |
| **Audio Capture**       | <50ms       | <100ms     |
| **Buffer Accumulation** | 1-3 seconds | 5 seconds  |
| **Whisper Processing**  | <2 seconds  | <5 seconds |
| **Total Latency**       | <3 seconds  | <6 seconds |

### 8.2 Quality Targets

- **Accuracy:** >85% for clear English speech
- **Language Detection:** >90% accuracy
- **Translation Quality:** Conversational level for major languages

---

## 9. Testing Plan

### 9.1 Basic Functionality Tests

- [ ] Microphone capture works
- [ ] Audio buffering accumulates correctly
- [ ] VAD detects speech vs silence
- [ ] Whisper processes audio successfully
- [ ] UI updates in real-time
- [ ] Start/Stop/Clear buttons work

### 9.2 Performance Tests

- [ ] Test with 10+ minutes continuous speech
- [ ] Test language switching (English ↔ Hindi)
- [ ] Test with background noise
- [ ] Test memory usage over time
- [ ] Test with different microphone qualities

---

## 10. Success Criteria

### 10.1 MVP Success (Phase 1)

- ✅ Capture audio from microphone in real-time
- ✅ Display transcription with <5 second latency
- ✅ Auto-detect and translate languages to English
- ✅ Handle continuous speech for 30+ minutes
- ✅ Provide clear start/stop/clear controls

### 10.2 Quality Metrics

- ✅ Works on standard laptop/desktop setup
- ✅ Handles normal conversation pace
- ✅ Graceful handling of silence periods
- ✅ Reasonable accuracy for multiple languages

---

## 11. Future Enhancements (Phase 2+)

### 11.1 Multi-Speaker Support

- Speaker diarization integration
- Separate audio track processing
- Speaker identification and labeling

### 11.2 Advanced Features

- Real-time streaming transcription
- Cloud deployment on EC2
- WebSocket-based real-time updates
- Enhanced VAD with neural networks
- Custom model fine-tuning

### 11.3 Performance Optimizations

- GPU acceleration for VAD
- Optimized audio buffering strategies
- Parallel processing pipelines
- Memory usage optimization

---

This foundational system provides a solid base for real-time transcription that can be extended to multi-speaker scenarios in future phases! 🚀
