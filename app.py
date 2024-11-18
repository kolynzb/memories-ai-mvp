import gradio as gr
from pathlib import Path
from processor import HighQualityProcessor

# Ensure output directory exists
Path("outputs").mkdir(exist_ok=True)

# Initialize processor
processor = HighQualityProcessor()

def process_memorial(image, audio, text, progress=gr.Progress()):
    """Process inputs with progress tracking"""
    try:
        progress(0, desc="Starting processing...")
        
        # Input validation
        if not image:
            raise ValueError("Please upload an image")
        if not audio:
            raise ValueError("Please upload a voice recording")
        if not text or len(text.strip()) < 10:
            raise ValueError("Please enter text (at least 10 characters)")
            
        progress(0.2, desc="Analyzing inputs...")
        
        # Process the video
        progress(0.4, desc="Generating speech...")
        video_path = processor.create_memorial_video(image, audio, text)
        
        progress(1.0, desc="Complete!")
        return video_path
        
    except Exception as e:
        raise gr.Error(str(e))

# Create interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # High-Quality Memorial Video Generator
    Create beautiful memorial videos with advanced AI-powered voice recreation and animation.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Photo",
                type="pil",
                tool="editor",
                elem_id="photo_upload"
            )
            
            audio_input = gr.Audio(
                label="Upload Voice Recording (minimum 10 seconds)",
                type="filepath",
                elem_id="voice_upload"
            )
            
            text_input = gr.Textbox(
                label="Enter Text",
                placeholder="Enter the message you'd like the voice to say...",
                lines=3,
                elem_id="text_input"
            )
            
            submit_btn = gr.Button("Generate Video", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            
    submit_btn.click(
        fn=process_memorial,
        inputs=[image_input, audio_input, text_input],
        outputs=video_output
    )
    
    gr.Markdown("""
    ### For Best Results:
    1. Upload a clear, front-facing photo with good lighting
    2. Provide a clear voice recording with minimal background noise
    3. The voice recording should be at least 10 seconds long
    4. Write natural, conversational text
    """)

if __name__ == "__main__":
    demo.launch(share=True)