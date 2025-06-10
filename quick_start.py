#!/usr/bin/env python3
"""
Quick Start Real-Time Transcription
Simplified version with faster startup using medium model
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
from typing import Dict, Any
import os

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTranscriber:
    """Simplified real-time transcriber for quick demo"""
    
    def __init__(self):
        logger.info("🚀 Quick Start Transcription Loading...")
        
        # Use medium model for faster loading
        self.model = WhisperModel("medium", device="cpu", compute_type="int8")
        self.vad = webrtcvad.Vad(2)
        
        # State
        self.is_recording = False
        self.audio_buffer = []
        self.results = []
        
        logger.info("✅ Quick Transcriber Ready!")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if self.is_recording and not status:
            self.audio_buffer.extend(indata.flatten())
            
            # Process when we have 2 seconds of audio
            if len(self.audio_buffer) >= 32000:  # 2 seconds at 16kHz
                self.process_audio()
    
    def process_audio(self):
        """Process accumulated audio"""
        try:
            audio_data = np.array(self.audio_buffer[:32000], dtype=np.float32)
            self.audio_buffer = self.audio_buffer[16000:]  # Keep overlap
            
            # Simple energy check
            energy = np.sqrt(np.mean(audio_data ** 2))
            if energy < 0.01:
                return
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_data, 
                task="translate",
                language=None,
                beam_size=1  # Faster
            )
            
            text_parts = [seg.text.strip() for seg in segments]
            if text_parts:
                result = {
                    "timestamp": time.time(),
                    "text": " ".join(text_parts),
                    "language": info.language
                }
                self.results.append(result)
                logger.info(f"Transcribed: {result['text'][:50]}...")
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        return "🔴 Recording started! Speak now..."
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        return "⏹️ Recording stopped"
    
    def get_transcription(self):
        """Get latest transcription"""
        if not self.results:
            return "Waiting for speech..."
        
        # Show last 10 results
        recent = self.results[-10:]
        formatted = []
        for r in recent:
            timestamp = time.strftime("%H:%M:%S", time.localtime(r["timestamp"]))
            formatted.append(f"[{timestamp}] [{r['language'].upper()}] {r['text']}")
        
        return "\n".join(formatted)
    
    def clear_transcription(self):
        """Clear history"""
        self.results = []
        return ""
    
    def create_interface(self):
        """Create simple Gradio interface"""
        with gr.Blocks(title="Quick Transcription", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎤 Quick Real-Time Transcription")
            gr.Markdown("**Medium model for faster startup** - Click Start, speak, and see results!")
            
            with gr.Row():
                start_btn = gr.Button("🔴 Start Recording", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Recording", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            status = gr.Textbox(label="Status", value="Ready to record")
            
            transcription = gr.Textbox(
                label="Live Transcription",
                lines=8,
                value="",
                interactive=False
            )
            
            # Event handlers
            start_btn.click(fn=self.start_recording, outputs=status)
            stop_btn.click(fn=self.stop_recording, outputs=status)
            clear_btn.click(fn=self.clear_transcription, outputs=transcription)
            
            # Auto-refresh every 500ms
            interface.load(fn=self.get_transcription, outputs=transcription, every=0.5)
        
        return interface
    
    def run(self):
        """Start the application"""
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=16000,
                blocksize=1600,  # 100ms chunks
                dtype=np.float32
            )
            
            self.stream.start()
            logger.info("🎤 Audio stream started")
            
            # Launch interface
            interface = self.create_interface()
            interface.launch(
                server_name="0.0.0.0",
                server_port=7861,  # Different port from main app
                share=False,
                quiet=True
            )
            
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            if hasattr(self, 'stream'):
                self.stream.stop()

if __name__ == "__main__":
    app = QuickTranscriber()
    app.run() 