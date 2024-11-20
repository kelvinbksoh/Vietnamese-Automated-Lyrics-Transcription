

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import numpy as np
import pprint
import streamlit as st

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import json
from tqdm import tqdm

import librosa
import numpy as np

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import pprint
import random
import gc


class pho_whisper_args:
    model =  "vinai/PhoWhisper-small"
    prompt = 'lời nhạc: '
    language = 'vi'
    no_speech_threshold = None #0.6
    logprob_threshold = None #-1.0
    temperature = (0.4,0.8)

def unload_current_model(reload_model_name): 
    if 'model' in st.session_state:
        del st.session_state["model"]
        st.session_state["model"] = None
    
    if 'processor' in st.session_state:
        del st.session_state["processor"]
        st.session_state["processor"] = None

    if reload_model_name:
        st.session_state["model_name"] = None
        
    torch.cuda.empty_cache()
    gc.collect()


@st.cache_data
def load_model(model_name):
    if 'xyzDivergence' in model_name:
        #Our finetuned Whisper
        model = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30, 
            device='cuda',
            torch_dtype=torch.float16,
            return_timestamps="word",
            tokenizer=model_name
        )
        processor = None

    elif 'vinai' in model_name:
        #PhoWhisper   
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return model, processor

def find_audios(parent_dir, exts=['.wav', '.mp3', '.flac', '.webm', '.mp4', '.m4a']):
    audio_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                audio_files.append(os.path.join(root, file))
    return audio_files


#################### Whisper ####################
def remove_possible_overlaps(transcriptions, maximum_overlapping_tokens = 5):
    cleaned_transcriptions = []

    for i, current in enumerate(transcriptions):
        if i == 0:
            # Add the first segment without changes
            cleaned_transcriptions.append(current)
            continue
        

        previous_text = cleaned_transcriptions[-1]['text'].split(' ')
        if len(previous_text) > 0:
            if len(previous_text[-1]) > 0:
                previous_text[-1] = previous_text[-1][:-1] if previous_text[-1][-1] == '.' else previous_text[-1]
        
        current_text  = current['text'].split(' ')

        for j in range(8):
            prev_tokens = previous_text[-j-1:]
            cur_tokens  = current_text[:j+1]
            if prev_tokens == cur_tokens:
                # Matched
                current_text = current_text[j+1:]
                break

        current['text'] = ' '.join(current_text)
        cleaned_transcriptions.append(current)

    return cleaned_transcriptions


def chunk_audio(audio_path, sample_rate=16000, segment_length=30, overlap = 1.):
    """Load and split audio into 30-second chunks, 0.5 seconds overlap."""
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Calculate the number of samples per segment and the overlap in samples
    num_samples = segment_length * sample_rate
    overlap_samples = int(overlap * sample_rate)

    # Create the chunks with overlap
    chunks = [
        audio[i:i + num_samples] 
        for i in range(0, len(audio) - overlap_samples, num_samples - overlap_samples)
    ]

    chunk_timestamps = [(i/sample_rate, (i + num_samples)/ float(sample_rate)) for i in range(0, len(audio) - overlap_samples, num_samples - overlap_samples)]

    return chunks, sample_rate, chunk_timestamps

def transcribe_chunk(chunk, processor, model, no_speech_threshold=0.6, logprob_threshold=-1.0, temperature = (0.4, 0.7)):
    input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

    # Generate transcription with VAD filtering
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=400,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            temperature= random.uniform(temperature[0], temperature[1])
        )
    
    # Decode and collect no-speech probability and log-probability information
    decoded_text = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
    return {
        "text": decoded_text
    }


def transcribe_full(audio_path, processor, model, temperature = (0.4, 0.7), segment_length=30, no_speech_threshold=0.6, logprob_threshold=-1.0):
    # Chunk the audio
    chunks, _, timestamps = chunk_audio(audio_path, segment_length=segment_length)

    all_segments = []
    for index, chunk in enumerate(chunks):
        # Transcribe each chunk and gather segment details
        segment = transcribe_chunk(chunk, processor, model, no_speech_threshold, logprob_threshold, temperature)
        segment['timestamp'] = [timestamps[index][0], timestamps[index][1]]
        all_segments.append(segment)

    all_segments = remove_possible_overlaps(all_segments)

    return all_segments
