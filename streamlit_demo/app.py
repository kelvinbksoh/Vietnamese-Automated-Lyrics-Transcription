import streamlit as st
import json
import os
from io import BytesIO

from utils.model import load_model, transcribe_full, unload_current_model, pho_whisper_args
from utils.text import output_render
from utils.correction import load_llm, generate_corrections
import librosa


if "model_name" not in st.session_state:
    st.session_state["model_name"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "processor" not in st.session_state:
    st.session_state["processor"] = None


def reload_model():
    unload_current_model(False)
    st.session_state["model"], st.session_state["processor"] = load_model(st.session_state["model_name"])

option = st.selectbox(
    "Model",
    ("vinai/PhoWhisper-small", \
    "xyzDivergence/whisper-medium-vietnamese-lyrics-transcription",\
    "xyzDivergence/whisper-large-v2-vietnamese-lyrics-transcription"),
    key='model_name',
    on_change=reload_model
)

use_llm_corrector = st.checkbox("Enable LLM Correction")
llm_corrector = None
if use_llm_corrector:
    st.write("Loading LLM Corrector...")
    llm_corrector = load_llm()
    
# Streamlit app configuration
st.title(f"Audio Transcription App using {st.session_state['model_name']}")

st.write("Upload an audio file to transcribe and download the result as a JSON file.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])


# Check if file is uploaded
if uploaded_file is not None:
    file_name = os.path.splitext(uploaded_file.name)[0]

    # Transcribe the audio
    with st.spinner("Transcribing audio..."):

        if 'vinai' in st.session_state["model_name"]:
            transcription = transcribe_full(
                audio_path=uploaded_file,
                processor=st.session_state["processor"],
                model=st.session_state["model"],
                no_speech_threshold=pho_whisper_args.no_speech_threshold,
                logprob_threshold=pho_whisper_args.logprob_threshold,
                temperature = pho_whisper_args.temperature
                )
            format = 'phowhisper'

        elif 'xyzDivergence' in st.session_state["model_name"]:
            audio_data, sr = librosa.load(uploaded_file, sr=16000)  # Load audio as numpy array
            transcription = st.session_state["model"](audio_data)
            format = 'whisper'
            print(transcription)
        
    
    # Display the transcription result
    st.write("Transcription result:")

    text_result = output_render(transcription, format=format)

   # If LLM correction is enabled, process the result
    if use_llm_corrector and llm_corrector:
        with st.spinner("Applying LLM corrections..."):
            text_result = generate_corrections(llm_corrector, text_result)

    st.text_area("Transcription Text", text_result, height=200)
    
    # Download button for JSON result
    json_result = json.dumps(transcription, ensure_ascii=False).encode('utf-8')  # ensure_ascii=False allows special characters
    st.download_button(label="Download Transcription as JSON",
                       data=json_result,
                       file_name=f"{file_name}.json",
                       mime="application/json")
    
    st.download_button(label="Download Transcription as text",
                       data=text_result,
                       file_name=f"{file_name}.txt",
                       mime="text/plain")