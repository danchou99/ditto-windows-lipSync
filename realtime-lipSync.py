import os
import time
import threading
import queue
import numpy as np
import pyaudio
import wave
import tempfile
import subprocess
from pathlib import Path

# STT and TTS imports
import speech_recognition as sr
from gtts import gTTS
import openai

# Ditto imports
from stream_pipeline_offline import StreamSDK
import librosa

#from dotenv import load_dotenv
# Load environment variables
#load_dotenv()

OPENAI_API_KEY = "" #os.getenv("OPENAI_API_KEY", None)

class RealtimeLipSync:
    def __init__(self, 
                 data_root="./checkpoints/ditto_trt_3090",
                 cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                 avatar_path="./example/image.png",
                 output_path="./tmp/realtime_output.mp4",
                 openai_api_key=OPENAI_API_KEY,
                 chunk_duration=2.0,  # seconds per audio chunk
                 sample_rate=16000):
        
        self.data_root = data_root
        self.cfg_pkl = cfg_pkl
        self.avatar_path = avatar_path
        self.output_path = output_path
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        # Audio settings
        # self.chunk_size = int(sample_rate * chunk_duration)
        self.chunk_size = 6480
        self.audio_format = pyaudio.paFloat32
        #self.audio_format = pyaudio.paInt16 # Use Int16 for compatibility with gTTS
        self.channels = 1
        
        # Queues for inter-thread communication
        self.audio_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        
        # Control flags
        self.is_running = False
        self.is_speaking = False
        
        # Initialize components
        self._init_audio()
        self._init_stt()
        self._init_llm(openai_api_key)
        self._init_ditto()
        
    def _init_audio(self):
        """Initialize PyAudio for microphone input"""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
    def _init_stt(self):
        """Initialize Speech-to-Text recognizer"""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
    def _init_llm(self, api_key):
        """Initialize LLM (OpenAI)"""
        if api_key:
            openai.api_key = api_key
            self.use_llm = True
        else:
            self.use_llm = False
            print("Warning: No OpenAI API key provided. LLM responses disabled.")
            
    def _init_ditto(self):
        print("Initializing Ditto lip-sync system...")
        
        # Initialize SDK with online mode
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root)
        
        # Setup for real-time processing
        self.sdk.setup(
            source_path=self.avatar_path,
            output_path=self.output_path,
            online_mode=True,  # Enable online mode for real-time
            chunksize=(3, 5, 2),  # Audio chunk processing
            sampling_timesteps=50,
            overlap_v2=10
        )
        
        # Start Ditto processing threads
        self.sdk.setup_Nd(N_d=1000)  # Large enough for continuous processing
        
        print("Ditto system initialized successfully!")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for real-time audio capture"""
        if self.is_running:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            # audio_data = np.frombuffer(in_data, dtype=np.int16)  # Use Int16 for compatibility with gTTS
            # Check if audio level is above threshold (speech detection)
            audio_level = np.sqrt(np.mean(audio_data**2))
            
            # print(f"[Audio Callback] Level: {audio_level:.5f}")
            if audio_level > 0.01:  # Adjust threshold as needed (Initial value=0.01). For Int16 it is around 300-1000 (check accordingly)
                self.audio_queue.put(audio_data)
                
        return (in_data, pyaudio.paContinue)
    
    def _stt_worker(self):
        """Worker thread for Speech-to-Text processing"""
        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1)
                
                # Convert to audio file for STT
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                    
                # Save audio chunk as WAV
                with wave.open(temp_audio_path, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_chunk.tobytes())
                
                # Perform STT
                try:
                    with sr.AudioFile(temp_audio_path) as source:
                        audio = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio)
                        
                        if text.strip():
                            print(f"STT: {text}")
                            self.llm_queue.put(text)
                            
                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    print(f"STT Error: {e}")
                finally:
                    os.unlink(temp_audio_path)
            except queue.Empty:
                continue
                
    def _llm_worker(self):
        """Worker thread for LLM processing"""
        while self.is_running:
            try:
                user_text = self.llm_queue.get(timeout=1)
                
                if not self.use_llm:
                    # Simple echo response for testing
                    response = f"I heard you say: {user_text}"
                else:
                    # Get LLM response
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Keep responses concise and natural for speech synthesis."},
                                {"role": "user", "content": user_text}
                            ],
                            max_tokens=1000
                        ).choices[0].message.content
                    except Exception as e:
                        print(f"LLM Error: {e}")
                        response = "I'm sorry, I couldn't process that request."
                
                print(f"LLM Response: {response}")
                self.tts_queue.put(response)
                
            except queue.Empty:
                continue
                
    def _tts_worker(self):
        """Worker thread for Text-to-Speech processing"""
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=1)
                
                # Generate TTS audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                '''with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                    temp_mp3_path = temp_mp3.name'''
                
                try:
                    # Use gTTS for speech synthesis
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(temp_audio_path)
                    # tts.save(temp_mp3_path)

                    #convert to true wav
                    '''wav_path = temp_mp3_path.replace(".mp3", ".wav")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp_mp3_path,
                        "-ar", str(self.sample_rate), "-ac", str(self.channels),
                        wav_path], check=True)'''
                    
                    # audio, sr = librosa.load(wav_path, sr=self.sample_rate)

                    
                    # Load audio and process with Ditto
                    audio, sr = librosa.load(temp_audio_path, sr=self.sample_rate)
                    
                    # Process audio in chunks for real-time lip-sync
                    # chunk_samples = int(self.sample_rate * self.chunk_duration)
                    chunk_samples = 6480
                    
                    for i in range(0, len(audio), chunk_samples):
                        chunk = audio[i:i + chunk_samples]
                        
                        # Pad if necessary
                        if len(chunk) < chunk_samples:
                            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                        
                        # Process with Ditto
                        print("running Ditto processing on audio chunk...")
                        self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))
                        
                        # Small delay for real-time processing
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"TTS Error: {e}")
                finally:
                    os.unlink(temp_audio_path)
                    #os.unlink(wav_path)
            except queue.Empty:
                continue
    
    def start(self):
        """Start the real-time lip-sync system"""
        print("Starting real-time lip-sync system...")
        self.is_running = True
        
        # Start worker threads
        self.stt_thread = threading.Thread(target=self._stt_worker)
        self.llm_thread = threading.Thread(target=self._llm_worker)
        self.tts_thread = threading.Thread(target=self._tts_worker)
        
        self.stt_thread.start()
        self.llm_thread.start()
        self.tts_thread.start()
        
        # Start audio stream
        self.stream.start_stream()
        
        print("System started! Speak into the microphone...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the real-time lip-sync system"""
        print("Stopping real-time lip-sync system...")
        self.is_running = False
        
        # Stop audio stream
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        # Close Ditto SDK
        self.sdk.close()
        
        # Wait for threads to finish
        self.stt_thread.join()
        self.llm_thread.join()
        self.tts_thread.join()
        
        print("System stopped.")


def main():
    """Main function to run the real-time lip-sync system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time lip-sync system")
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_3090",
                       help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                       help="path to cfg_pkl")
    parser.add_argument("--avatar_path", type=str, default="./example/isha_v1.mp4",
                       help="path to avatar image/video")
    parser.add_argument("--output_path", type=str, default="./tmp/realtime_output.mp4",
                       help="path to output video")
    parser.add_argument("--openai_key", type=str, default=None,
                       help="OpenAI API key for LLM responses")
    parser.add_argument("--chunk_duration", type=float, default=2.0,
                       help="audio chunk duration in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Initialize and start the system
    system = RealtimeLipSync(
        data_root=args.data_root,
        cfg_pkl=args.cfg_pkl,
        avatar_path=args.avatar_path,
        output_path=args.output_path,
        openai_api_key=args.openai_key,
        chunk_duration=args.chunk_duration
    )
    
    system.start()


if __name__ == "__main__":
    main() 
