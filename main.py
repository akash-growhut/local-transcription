#!/usr/bin/env python3
"""
Real-Time Audio Transcription System
Based on Technical Requirements Document v1.0

A sophisticated real-time transcription system that captures audio from microphone,
processes it through Whisper, and displays live transcription with English translation.
"""

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import webrtcvad
import gradio as gr
import queue
import threading
import time
import logging
from collections import deque
from typing import Optional, Dict, List, Any
import warnings
import os

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioBuffer:
    """Manages audio buffering with configurable duration and chunk size"""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.2, buffer_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.buffer_duration = buffer_duration
        
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer_size = int(sample_rate * buffer_duration)
        
        self.audio_queue = queue.Queue()
        self.current_buffer = deque(maxlen=self.buffer_size)
        
        logger.info(f"AudioBuffer initialized: {chunk_duration}s chunks, {buffer_duration}s buffer")
    
    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add audio chunk to buffer"""
        self.current_buffer.extend(chunk.flatten())
    
    def get_buffer_if_ready(self) -> Optional[np.ndarray]:
        """Get buffer if it has enough audio data"""
        if len(self.current_buffer) >= self.buffer_size:
            # Get buffer data and clear for next accumulation
            buffer_data = np.array(list(self.current_buffer))
            self.current_buffer.clear()
            return buffer_data
        return None
    
    def get_current_buffer(self) -> np.ndarray:
        """Get current buffer regardless of size (for final processing)"""
        if len(self.current_buffer) > 0:
            buffer_data = np.array(list(self.current_buffer))
            self.current_buffer.clear()
            return buffer_data
        return np.array([])

class VADProcessor:
    """Voice Activity Detection using WebRTC VAD"""
    
    def __init__(self, aggressiveness: int = 2, frame_duration_ms: int = 20):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(16000 * frame_duration_ms / 1000)  # 16kHz sample rate
        
        logger.info(f"VAD initialized with aggressiveness level {aggressiveness}")
    
    def is_speech(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> bool:
        """Detect if audio buffer contains speech"""
        try:
            # Convert to int16 format required by webrtcvad
            if audio_buffer.dtype != np.int16:
                audio_int16 = (audio_buffer * 32767).astype(np.int16)
            else:
                audio_int16 = audio_buffer
            
            # Ensure audio is the right length for VAD frames
            if len(audio_int16) < self.frame_length:
                return False
            
            # Split into frames and check each one
            frames = []
            for i in range(0, len(audio_int16) - self.frame_length + 1, self.frame_length):
                frame = audio_int16[i:i + self.frame_length]
                if len(frame) == self.frame_length:
                    frames.append(frame)
            
            if not frames:
                return False
            
            # Count speech frames
            speech_frames = 0
            for frame in frames:
                try:
                    if self.vad.is_speech(frame.tobytes(), sample_rate):
                        speech_frames += 1
                except Exception as e:
                    logger.debug(f"VAD frame processing error: {e}")
                    continue
            
            # Return True if more than 30% of frames contain speech
            speech_ratio = speech_frames / len(frames) if frames else 0
            return speech_ratio > 0.3
            
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            # Fallback to energy-based detection
            return self._energy_based_vad(audio_buffer)
    
    def _energy_based_vad(self, audio_buffer: np.ndarray) -> bool:
        """Fallback energy-based voice activity detection"""
        try:
            energy = np.sqrt(np.mean(audio_buffer ** 2))
            return energy > 0.01  # Empirical threshold
        except Exception:
            return False

class WhisperProcessor:
    """Whisper-based speech recognition and translation"""
    
    def __init__(self, model_size: str = "large-v3"):
        logger.info(f"Loading Faster-Whisper {model_size} model...")
        
        # Set device based on availability
        device = "cpu"  # Default to CPU for compatibility
        compute_type = "int8"  # Memory efficient
        
        try:
            # Try to use GPU if available (for future enhancement)
            self.model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type,
                download_root=os.path.join(os.getcwd(), "models")
            )
            logger.info("Faster-Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to smaller model
            logger.info("Falling back to medium model...")
            self.model = WhisperModel(
                "medium", 
                device=device, 
                compute_type=compute_type,
                download_root=os.path.join(os.getcwd(), "models")
            )
    
    def transcribe_chunk(self, audio_buffer: np.ndarray) -> Dict[str, Any]:
        """Process audio buffer and return transcription result"""
        try:
            if len(audio_buffer) == 0:
                return {"text": "", "language": "unknown", "error": None}
            
            # Ensure audio is float32 and normalized
            if audio_buffer.dtype != np.float32:
                audio_buffer = audio_buffer.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_buffer)) > 0:
                audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
            
            # Transcribe with Faster-Whisper
            segments, info = self.model.transcribe(
                audio_buffer,
                task="translate",  # Always translate to English
                language=None,     # Auto-detect language
                temperature=0.0,   # Deterministic
                beam_size=5,
                condition_on_previous_text=False
            )
            
            # Combine all segments into single text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            final_text = " ".join(text_parts).strip()
            detected_language = info.language if hasattr(info, 'language') else "unknown"
            
            return {
                "text": final_text,
                "language": detected_language,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Whisper processing error: {e}")
            return {"text": "", "language": "unknown", "error": str(e)}

class RealtimeWhisperApp:
    """Main application class for real-time transcription"""
    
    def __init__(self, model_size: str = "large"):
        # Initialize components
        self.audio_buffer = AudioBuffer()
        self.vad_processor = VADProcessor()
        self.whisper_processor = WhisperProcessor(model_size)
        
        # Application state
        self.is_recording = False
        self.transcription_history: List[Dict[str, Any]] = []
        self.result_queue = queue.Queue()
        
        # Threading
        self.worker_thread = None
        self.stream = None
        
        logger.info("RealtimeWhisperApp initialized successfully")
    
    def audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Audio input callback - called every 200ms"""
        if status:
            logger.warning(f"Audio callback error: {status}")
        
        if self.is_recording:
            # Add chunk to buffer
            self.audio_buffer.add_chunk(indata)
    
    def process_audio_worker(self) -> None:
        """Background worker for audio processing"""
        logger.info("Audio processing worker started")
        
        while True:
            try:
                if not self.is_recording:
                    time.sleep(0.1)
                    continue
                
                # Check if we have enough audio to process
                audio_data = self.audio_buffer.get_buffer_if_ready()
                
                if audio_data is None:
                    time.sleep(0.1)
                    continue
                
                # Check for speech activity
                if not self.vad_processor.is_speech(audio_data):
                    logger.debug("No speech detected, skipping chunk")
                    continue
                
                logger.info("Processing audio chunk...")
                
                # Transcribe with Whisper
                result = self.whisper_processor.transcribe_chunk(audio_data)
                
                if result["text"]:
                    result["timestamp"] = time.time()
                    self.result_queue.put(result)
                    logger.info(f"Transcribed [{result['language']}]: {result['text'][:50]}...")
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(0.1)
    
    def start_recording(self) -> str:
        """Start recording and transcription"""
        try:
            if self.is_recording:
                return "Already recording!"
            
            self.is_recording = True
            logger.info("Starting recording...")
            return "🔴 Recording started! Speak now..."
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return f"Error: {str(e)}"
    
    def stop_recording(self) -> str:
        """Stop recording"""
        try:
            if not self.is_recording:
                return "Not currently recording"
            
            self.is_recording = False
            
            # Process any remaining audio in buffer
            remaining_audio = self.audio_buffer.get_current_buffer()
            if len(remaining_audio) > 0 and self.vad_processor.is_speech(remaining_audio):
                result = self.whisper_processor.transcribe_chunk(remaining_audio)
                if result["text"]:
                    result["timestamp"] = time.time()
                    self.result_queue.put(result)
            
            logger.info("Recording stopped")
            return "⏹️ Recording stopped"
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return f"Error: {str(e)}"
    
    def get_transcription(self) -> str:
        """Get latest transcription results"""
        try:
            # Process all pending results
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    self.transcription_history.append(result)
                except queue.Empty:
                    break
            
            # Format recent transcriptions
            if not self.transcription_history:
                return "Waiting for speech..."
            
            # Show last 15 transcriptions
            recent_transcriptions = self.transcription_history[-15:]
            formatted_text = []
            
            for result in recent_transcriptions:
                timestamp = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
                lang = result["language"].upper()
                text = result["text"]
                formatted_text.append(f"[{timestamp}] [{lang}] {text}")
            
            return "\n".join(formatted_text)
            
        except Exception as e:
            logger.error(f"Error getting transcription: {e}")
            return f"Error: {str(e)}"
    
    def clear_transcription(self) -> str:
        """Clear transcription history"""
        try:
            self.transcription_history.clear()
            
            # Clear any pending results
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("Transcription history cleared")
            return ""
            
        except Exception as e:
            logger.error(f"Error clearing transcription: {e}")
            return f"Error: {str(e)}"
    
    def get_stats(self) -> str:
        """Get application statistics"""
        try:
            total_transcriptions = len(self.transcription_history)
            languages_detected = set(r["language"] for r in self.transcription_history)
            status = "🔴 Recording" if self.is_recording else "⚫ Stopped"
            
            stats = [
                f"**Status:** {status}",
                f"**Total Transcriptions:** {total_transcriptions}",
                f"**Languages Detected:** {', '.join(sorted(languages_detected)) if languages_detected else 'None'}",
                f"**Audio Buffer:** {self.audio_buffer.buffer_duration}s",
                f"**VAD Enabled:** ✅",
                f"**Model:** Whisper Large"
            ]
            
            return "\n".join(stats)
            
        except Exception as e:
            return f"Stats unavailable: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(
            title="Real-Time Whisper Transcription",
            theme=gr.themes.Soft(),
            css="""
            .status-box { background: #f0f0f0; padding: 10px; border-radius: 5px; }
            .transcription-box { font-family: 'Courier New', monospace; }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🎤 Real-Time Audio Transcription System
            
            **Features:**
            - Real-time microphone capture and transcription
            - Automatic language detection and English translation
            - Voice Activity Detection (VAD) to filter silence
            - Powered by OpenAI Whisper Large model
            
            **Instructions:** Click "Start Recording", speak clearly into your microphone, and watch the live transcription appear below!
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    start_btn = gr.Button("🔴 Start Recording", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop Recording", variant="secondary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                
                with gr.Column(scale=2):
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to record - Click 'Start Recording' to begin",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
            
            with gr.Row():
                with gr.Column(scale=3):
                    transcription_output = gr.Textbox(
                        label="Live Transcription",
                        lines=12,
                        max_lines=20,
                        value="",
                        interactive=False,
                        placeholder="Transcription will appear here...\n\nFormat: [Timestamp] [Language] Transcribed text",
                        elem_classes=["transcription-box"]
                    )
                
                with gr.Column(scale=1):
                    stats_output = gr.Markdown(
                        label="Statistics",
                        value=self.get_stats()
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
            
            # Auto-refresh transcription and stats every 500ms
            interface.load(
                fn=self.get_transcription,
                outputs=transcription_output,
                every=0.5
            )
            
            interface.load(
                fn=self.get_stats,
                outputs=stats_output,
                every=2.0
            )
        
        return interface
    
    def run(self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
        """Start the application"""
        try:
            # Check available audio devices
            logger.info("Available audio devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.info(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
            
            # Start audio stream
            logger.info("Starting audio stream...")
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
            
            self.stream.start()
            self.worker_thread.start()
            
            logger.info("Audio system initialized successfully!")
            
            # Create and launch interface
            interface = self.create_interface()
            logger.info(f"Starting web interface on http://{server_name}:{server_port}")
            
            interface.launch(
                server_name=server_name,
                server_port=server_port,
                share=share,
                quiet=False,
                show_error=True
            )
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                
            logger.info("Application cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main application entry point"""
    try:
        logger.info("🚀 Starting Real-Time Whisper Transcription System")
        logger.info("=" * 60)
        
        # Create application
        app = RealtimeWhisperApp(model_size="large-v3")
        
        # Run the application
        app.run(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise
    finally:
        logger.info("Application shutdown")

if __name__ == "__main__":
    main() 