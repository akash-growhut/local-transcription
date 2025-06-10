#!/usr/bin/env python3
"""
Audio System Test Script
Tests microphone capture and basic audio processing functionality
"""

import sounddevice as sd
import numpy as np
import time
import sys

def list_audio_devices():
    """List all available audio devices"""
    print("=" * 60)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 60)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append(f"INPUT({device['max_input_channels']})")
            input_devices.append(i)
        if device['max_output_channels'] > 0:
            device_type.append(f"OUTPUT({device['max_output_channels']})")
        
        status = "✅" if device_type else "❌"
        print(f"{status} Device {i}: {device['name']}")
        print(f"    Type: {' + '.join(device_type) if device_type else 'No audio channels'}")
        print(f"    Sample Rate: {device['default_samplerate']} Hz")
        print()
    
    return input_devices

def test_microphone_capture(duration=5):
    """Test microphone capture for specified duration"""
    print("=" * 60)
    print(f"MICROPHONE CAPTURE TEST ({duration} seconds)")
    print("=" * 60)
    
    print("Starting audio capture...")
    try:
        # Record audio
        sample_rate = 16000
        audio_data = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype=np.float32
        )
        
        print(f"Recording for {duration} seconds... Speak into your microphone!")
        sd.wait()  # Wait until recording is finished
        
        # Analyze the recorded audio
        print("\nAnalyzing captured audio...")
        
        # Calculate statistics
        max_amplitude = np.max(np.abs(audio_data))
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        print(f"✅ Audio captured successfully!")
        print(f"   Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Max Amplitude: {max_amplitude:.4f}")
        print(f"   RMS Energy: {rms_energy:.4f}")
        
        # Check if audio was actually captured
        if max_amplitude > 0.001:
            print("🎤 Good! Audio signal detected.")
            
            # Test playback
            print("\nTesting playback...")
            try:
                sd.play(audio_data, sample_rate)
                print("🔊 Playing back captured audio...")
                sd.wait()
                print("✅ Playback completed!")
            except Exception as e:
                print(f"❌ Playback failed: {e}")
                
        else:
            print("⚠️  Warning: Very low audio signal. Check microphone connection.")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio capture failed: {e}")
        return False

def test_real_time_callback():
    """Test real-time audio callback"""
    print("=" * 60)
    print("REAL-TIME CALLBACK TEST (10 seconds)")
    print("=" * 60)
    
    chunk_count = 0
    energy_values = []
    
    def audio_callback(indata, frames, time, status):
        nonlocal chunk_count, energy_values
        
        if status:
            print(f"Callback status: {status}")
        
        # Calculate energy
        energy = np.sqrt(np.mean(indata ** 2))
        energy_values.append(energy)
        chunk_count += 1
        
        # Print progress every 50 chunks (about every second)
        if chunk_count % 50 == 0:
            avg_energy = np.mean(energy_values[-50:])
            print(f"Chunk {chunk_count:3d}: Average energy = {avg_energy:.4f}")
    
    try:
        print("Starting real-time audio stream...")
        print("Speak into your microphone to see energy levels!")
        
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=int(16000 * 0.02),  # 20ms chunks
            dtype=np.float32
        ):
            time.sleep(10)  # Run for 10 seconds
        
        print(f"\n✅ Real-time callback test completed!")
        print(f"   Total chunks processed: {chunk_count}")
        print(f"   Average energy: {np.mean(energy_values):.4f}")
        print(f"   Max energy: {np.max(energy_values):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time callback test failed: {e}")
        return False

def main():
    """Run all audio tests"""
    print("🎤 AUDIO SYSTEM DIAGNOSTIC")
    print("This script will test your microphone and audio system.")
    print()
    
    # Test 1: List devices
    input_devices = list_audio_devices()
    
    if not input_devices:
        print("❌ ERROR: No input devices found!")
        print("Please check your microphone connection.")
        sys.exit(1)
    
    print(f"Found {len(input_devices)} input device(s): {input_devices}")
    
    # Test 2: Basic capture
    input("\nPress Enter to start microphone capture test...")
    success1 = test_microphone_capture(duration=3)
    
    if not success1:
        print("❌ Basic microphone test failed. Please check your setup.")
        sys.exit(1)
    
    # Test 3: Real-time callback
    input("\nPress Enter to start real-time callback test...")
    success2 = test_real_time_callback()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Device Detection: {'✅' if input_devices else '❌'}")
    print(f"Basic Capture:    {'✅' if success1 else '❌'}")
    print(f"Real-time Stream: {'✅' if success2 else '❌'}")
    print()
    
    if success1 and success2:
        print("🎉 All tests passed! Your audio system is ready for real-time transcription.")
        print("You can now run: python main.py")
    else:
        print("⚠️  Some tests failed. Please check your microphone setup.")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 