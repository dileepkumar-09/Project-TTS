import torch
import gradio as gr
from TTS.api import TTS
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)

def generate_audio(text: str, ref_audio_path: str):
    os.makedirs("outputs", exist_ok=True)
    
    output_path = "outputs/output.wav"
    
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=ref_audio_path,
        language="en" 
    )
    
    return output_path

demo = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Text(label="Text to Synthesize", value="Hello, this is a test of the cloned voice."),
        gr.Audio(label="Reference WAV (Speaker Voice)", type="filepath", sources=["upload"]),
    ],
    outputs=[
        gr.Audio(label="Generated Audio", type="filepath"),
    ],
    title="YourTTS Voice Cloning Demo",
    description="Uses YourTTS for speech generation."
)

demo.launch()