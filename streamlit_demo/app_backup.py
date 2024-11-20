import streamlit as st
import json
import os
from io import BytesIO

from utils.model import load_model, transcribe_full, unload_current_model, pho_whisper_args
from utils.text import output_render
import librosa
import torch

# Global variables to store model and processor
model = None
processor = None

def unload_model():
    global model, processor
    if model is not None or processor is not None:
        st.write("Unloading current model from GPU...")
        del model
        del processor
        model = None
        processor = None
        torch.cuda.empty_cache()
        st.write("CUDA memory cleared.")

# Function to load the model
def load_selected_model(model_name):
    global model, processor
    unload_model()  # Ensure the previous model is unloaded
    st.write(f"Loading model: {model_name}")
    model, processor = load_model(model_name)

# Dropdown menu for selecting the model
option = st.selectbox(
    "Model",
    (
        "vinai/PhoWhisper-small",
        "xyzDivergence/whisper-medium-vietnamese-lyrics-transcription",
        "xyzDivergence/whisper-large-v2-vietnamese-lyrics-transcription",
    ),
    on_change=lambda: load_selected_model(option),
)

# Load the model for the initial selection
load_selected_model(option)

st.title(f"Audio Transcription App using {option}")
st.write("Upload an audio file to transcribe and download the result as a JSON file.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    file_name = os.path.splitext(uploaded_file.name)[0]

    # Transcribe the audio
    with st.spinner("Transcribing audio..."):
        if "vinai" in option:
            transcription = transcribe_full(
                audio_path=uploaded_file,
                processor=processor,
                model=model,
                no_speech_threshold=pho_whisper_args.no_speech_threshold,
                logprob_threshold=pho_whisper_args.logprob_threshold,
                temperature=pho_whisper_args.temperature,
            )
            format = "phowhisper"
        elif "xyzDivergence" in option:
            audio_data, sr = librosa.load(uploaded_file, sr=16000)  # Load audio as numpy array
            transcription = model(audio_data)
            format = "whisper"
            print(transcription)

    # Display the transcription result
    st.write("Transcription result:")

    text_result = output_render(transcription, format=format)

    st.text_area("Transcription Text", text_result, height=200)

    # Download button for JSON result
    json_result = json.dumps(transcription, ensure_ascii=False).encode("utf-8")  # ensure_ascii=False allows special characters
    st.download_button(
        label="Download Transcription as JSON",
        data=json_result,
        file_name=f"{file_name}.json",
        mime="application/json",
    )

    st.download_button(
        label="Download Transcription as text",
        data=text_result,
        file_name=f"{file_name}.txt",
        mime="text/plain",
    )
