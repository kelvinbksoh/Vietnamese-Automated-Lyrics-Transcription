import os
import re
import glob
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torchaudio
import torchaudio.transforms as at

def get_train_test_audio_names():
    val_file_name = []
    for song in os.listdir(test_dir):
        if song == '.ipynb_checkpoints':
            continue
        song_name = os.path.basename(song.split('.mp3.json')[0])
        if song_name in list_of_pp_audios:
            val_file_name.append(song_name)

    train_file_name = []
    for song in os.listdir(pp_dir):
        if song == '.ipynb_checkpoints':
            continue
        song_name = os.path.basename(song)
        if song_name not in list_of_pp_audios:
            song_name = os.path.basename(song_name.split('.mp3.json')[0])
            if (song_name in val_file_name):
                continue
            train_file_name.append(song_name)
    return train_file_name, val_file_name

def load_audio(audio_path="../sample/0 Giờ 2 Phút.mp3"):
    waveform, sr = torchaudio.load(audio_path, normalize=True)
    waveform = at.Resample(sr, 16000)(waveform)#.numpy()#.flatten()
    audio_duration = waveform.shape[1] / 16000
    audio = waveform[0].numpy()
    return audio, audio_duration

# convert timestamp format to seconds
def timestamp_to_seconds(timestamp):
    minutes, seconds = timestamp.split(':')
    return int(minutes) * 60 + float(seconds)
    
# audio = waveform[0].numpy()

def generate_preprocessed_audio_list(audio_dir, audio_arr_dir):
    os.makedirs(audio_arr_dir, exist_ok=True)
    preprocessed_audio_list = []

    # for song in glob.glob(f"{pp_dir}/*.json"):            
    # for song in tqdm(audio_file_names):
    for song in tqdm(glob.glob(f"{audio_dir}/*.mp3")):
        song_name = os.path.basename(song).split('.mp3')[0]
        official = os.path.join(audio_dir, f'{song_name}.origin.lrc')

        ## Load audio
        try:
            audio_path = os.path.join(audio_dir, f'{song_name}' + '.mp3')
            audio, audio_duration = load_audio(audio_path)
        except Exception as e:
            print(str(e))
            continue

        with open(official, 'r') as _file:
            o_text = _file.read()
    
        pattern = re.compile(r'\[(\d{2}:\d{2}\.\d{2})\](.*)')
        timestamps = []
        sentences = []
        for line in o_text.split("\n"):
            match = pattern.match(line)
            if match:
                timestamp = match.group(1)
                sentence = match.group(2).strip()
                if sentence != "":
                    timestamps.append(timestamp)
                    sentences.append(sentence)    
        
        # Convert all timestamps to seconds
        timestamps_in_seconds = [timestamp_to_seconds(ts) for ts in timestamps]
        
        # Chunk the audio into 30-second intervals
        chunk_duration = 30  # 30 seconds
        
        audio_chunks = []
        chunk_start_times = []
        chunk_sentences = []
        
        sr = 16000
        for i in range(0, int(audio_duration), chunk_duration):
            chunk_start = i
            chunk_end = min(i + chunk_duration, audio_duration)
            
            # Find the sentences within this chunk's start and end time
            chunk_sentences_in_range = [
                sentence for ts, sentence in zip(timestamps_in_seconds, sentences) 
                if chunk_start <= ts < chunk_end
            ]
            
            # Extract corresponding audio chunk (in samples)
            start_sample = int(chunk_start * sr)
            end_sample = int(chunk_end * sr)
            audio_chunk = audio[start_sample:end_sample].copy()
            
            chunk_start_times.append((chunk_start, chunk_end))
            if len(chunk_sentences_in_range):
                chunk_sentences.append(chunk_sentences_in_range)
                audio_chunks.append(audio_chunk)
                sentence = " ".join(chunk_sentences_in_range)
                audio_arr_path = f"{audio_arr_dir}/{song_name}_chunk{i}"
                np.save(audio_arr_path, audio_chunk)
                data = {"audio": {"path": audio_path, "audio_arr_path": audio_arr_path + '.npy', "sampling_rate": 16000}, "sentence": sentence}
                
                preprocessed_audio_list.append(data)
        
        # Each entry in audio_chunks now corresponds to a 30-second chunk, and the sentences in chunk_sentences
        # are aligned with those audio chunks.
    return preprocessed_audio_list
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate audio array inputs for whisper")
    parser.add_argument('--train_dir', type=str, required=True, default='audio train directory')
    parser.add_argument('--test_dir', type=str, required=True, default='audio test directory')
    parser.add_argument('--audio_arr_dir_name', type=str, required=True, default='audio_arr_path_demucs')
    args = parser.parse_args()   

    train_dir = args.train_dir
    test_dir = args.test_dir
    audio_arr_dir_name = args.audio_arr_dir_name
    
    # gt_dir = 'CustomLyricWhiz/master_1st_1000/out/master_downloads_1st/'
    # pp_dir = 'CustomLyricWhiz/post_processed_transcripts/' # # len(glob.glob(f"{pp_dir}/*.json"))
    # list_of_pp_audios = [x.split('.mp3.json')[0] for x in os.listdir(pp_dir)]
    # test_dir = "master_1st_100_results_test_only"
    
    # train_file_names, test_file_names = get_train_test_audio_names()
    # print(len(train_file_names), len(test_file_names))
    
    # gt_dir = 'quoc_master_1st_100_preprocessed/'
    # gt_dir = 'htdemucs_vocal_only/'
    # gt_dir_filenames = os.listdir(gt_dir)
    # gt_dir_filenames = list(set([x.split('.')[0] for x in gt_dir_filenames]))
    # len(gt_dir_filenames)

    # train_file_names = [x for x in train_file_names if x in gt_dir_filenames]
    # test_file_names = [x for x in test_file_names if x in gt_dir_filenames]
    
    train_preprocessed_audio_list = generate_preprocessed_audio_list(train_dir, audio_arr_dir_name)
    test_preprocessed_audio_list = generate_preprocessed_audio_list(test_dir, audio_arr_dir_name)    

    np.save(f"train_preprocessed_{audio_arr_dir_name}_audio_list", train_preprocessed_audio_list)
    np.save(f"test_preprocessed_{audio_arr_dir_name}_audio_list", test_preprocessed_audio_list)