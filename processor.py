import torch
import numpy as np
from pathlib import Path
from PIL import Image
from TTS.api import TTS
import cv2
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, vfx
import whisper
from insightface.app import FaceAnalysis
import librosa
import soundfile as sf
from scipy.signal import medfilt
import time

class HighQualityProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.setup_models()
        
    def setup_models(self):
        """Initialize high-quality AI models"""
        print("Loading AI models...")
        
        # Load XTTS V2 for highest quality voice cloning
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                      progress_bar=True).to(self.device)
        
        # Load Whisper large-v3 for better voice analysis
        self.whisper = whisper.load_model("large-v3")
        
        # Load InsightFace for high-quality face processing
        self.face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0)
        
        print("Models loaded successfully!")
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Advanced image enhancement pipeline"""
        # Convert to numpy array
        img = np.array(image)
        
        # Analyze face
        faces = self.face_analyzer.get(img)
        if not faces:
            raise ValueError("No face detected in the image")
        
        # Get main face
        main_face = faces[0]
        
        # Extract face area with margin
        bbox = main_face['bbox'].astype(int)
        margin = 50
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.shape[1], x2 + margin)
        y2 = min(img.shape[0], y2 + margin)
        
        # Face-aware enhancement
        face_img = img[y1:y2, x1:x2]
        
        # Advanced image enhancement pipeline
        enhanced = cv2.detailEnhance(face_img, sigma_s=10, sigma_r=0.15)
        enhanced = cv2.edgePreservingFilter(enhanced, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Color correction
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Replace enhanced face in original image
        img[y1:y2, x1:x2] = enhanced
        
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    def process_audio(self, audio_path: str) -> str:
        """High-quality audio processing"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Noise reduction
        y_clean = librosa.decompose.nn_filter(y,
                                            aggregate=np.median,
                                            metric='cosine',
                                            width=int(sr/10))
        
        # Normalize audio
        y_norm = librosa.util.normalize(y_clean)
        
        # Remove silence
        y_trim, _ = librosa.effects.trim(y_norm, top_db=20)
        
        # Apply subtle compression
        y_compress = self.compress_audio(y_trim)
        
        # Save processed audio
        output_path = f"outputs/processed_audio_{int(time.time())}.wav"
        sf.write(output_path, y_compress, sr)
        
        return output_path
    
    def compress_audio(self, audio: np.ndarray, threshold: float = -20.0,
                      ratio: float = 4.0) -> np.ndarray:
        """Apply professional audio compression"""
        # Convert to dB
        db = librosa.amplitude_to_db(np.abs(audio))
        
        # Apply compression
        mask = db > threshold
        db[mask] = threshold + (db[mask] - threshold) / ratio
        
        # Convert back to amplitude
        return librosa.db_to_amplitude(db) * np.sign(audio)
    
    def generate_speech(self, reference_audio: str, text: str) -> str:
        """Generate high-quality speech with emotion preservation"""
        # Process reference audio
        clean_reference = self.process_audio(reference_audio)
        
        # Analyze original speech for emotion and style
        emotion_embeddings = self.whisper.embed_audio(clean_reference)
        
        # Generate speech with style transfer
        output_path = f"outputs/generated_speech_{int(time.time())}.wav"
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=clean_reference,
            language="en"
        )
        
        return output_path
    
    def create_video(self, image: Image.Image, audio_path: str,
                    duration_factor: float = 1.2) -> str:
        """Create high-quality video with professional effects"""
        # Load and get audio duration
        audio = AudioFileClip(audio_path)
        duration = audio.duration * duration_factor  # Add extra time for smooth transitions
        
        # Create base clip with enhanced image
        enhanced_image = self.enhance_image(image)
        
        def create_frame(t):
            # Complex movement pattern
            zoom = 1 + 0.05 * np.sin(t * np.pi / duration)
            shift_x = 20 * np.sin(t * 2 * np.pi / duration)
            shift_y = 10 * np.cos(t * 2 * np.pi / duration)
            
            # Create transform matrix
            img = np.array(enhanced_image)
            height, width = img.shape[:2]
            M = np.float32([[zoom, 0, shift_x], [0, zoom, shift_y]])
            
            # Apply transform
            return cv2.warpAffine(img, M, (width, height))
        
        # Create main clip
        main_clip = ImageClip(create_frame, duration=duration)
        
        # Add effects
        final_clip = (main_clip
                     .set_audio(audio)
                     .fadein(1.0)
                     .fadeout(1.0)
                     .fx(vfx.colorx, 1.1))  # Slight color enhancement
        
        # Add subtle vignette
        def add_vignette(frame):
            rows, cols = frame.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, cols/4)
            kernel_y = cv2.getGaussianKernel(rows, rows/4)
            kernel = kernel_y * kernel_x.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            return np.uint8(np.minimum(frame * (mask * 0.3 + 0.7), 255))
        
        final_clip = final_clip.fl(add_vignette)
        
        # Export with high quality
        output_path = f"outputs/memorial_video_{int(time.time())}.mp4"
        final_clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            bitrate="8000k"
        )
        
        return output_path

    def create_memorial_video(self, image: Image.Image, reference_audio: str,
                            text: str) -> str:
        """Main processing pipeline"""
        try:
            # 1. Generate enhanced speech
            print("Generating speech...")
            speech_path = self.generate_speech(reference_audio, text)
            
            # 2. Create final video
            print("Creating video...")
            video_path = self.create_video(image, speech_path)
            
            return video_path
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise