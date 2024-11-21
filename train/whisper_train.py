import torch
import torchaudio
import torchaudio.transforms as at

import os
import re
import glob
import numpy as np
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Audio
import os

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


import torch
import evaluate

import argparse

from dataclasses import dataclass
from typing import Any, Dict, List, Union

def preprocess_wav(wave_path, sentence:str="None", sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform).numpy()[0]
    return {"audio": {'path': wave_path, 'array': waveform, 'sampling_rate': sample_rate}, "sentence": sentence}

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

def generate_preprocessed_audio_list(audio_file_names):
    preprocessed_audio_list = []

    # for song in glob.glob(f"{pp_dir}/*.json"):            
    for song in tqdm(audio_file_names):
        song_name = os.path.basename(song).split('.mp3.json')[0]        
        official = os.path.join(gt_dir, f'{song_name}.origin.lrc')

        ## Load audio
        audio_path = os.path.join(gt_dir, f'{song_name}' + '.mp3')
        audio, audio_duration = load_audio(audio_path)

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
                audio_arr_path = f"audio_arr_path/{song_name}_chunk{i}"
                np.save(audio_arr_path, audio_chunk)
                data = {"audio": {"path": audio_path, "audio_arr_path": audio_arr_path + '.npy', "sampling_rate": 16000}, "sentence": sentence}
                
                preprocessed_audio_list.append(data)
        
        # Each entry in audio_chunks now corresponds to a 30-second chunk, and the sentences in chunk_sentences
        # are aligned with those audio chunks.
    return preprocessed_audio_list

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    # print(batch)
    audio_arr_path = batch["audio"]['audio_arr_path']
    audio = np.load(audio_arr_path)
    sampling_rate = batch["audio"]["sampling_rate"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"], max_length=448, truncation=True).input_ids
    return batch

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Any, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}
    
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"#args.gpu_no
    parser = argparse.ArgumentParser(description="Whisper Trainer")
    parser.add_argument('--experiment_name', type=str, required=True, default='whisper-medium-vi-demucs')
    parser.add_argument('--model_name', type=str, required=True, default='openai/whisper-small')    
    parser.add_argument('--train_audio_list', type=str, required=True, default='train_preprocessed_pp_audio_list.npy')
    parser.add_argument('--test_audio_list', type=str, required=True, default='test_preprocessed_pp_audio_list.npy')    
    # parser.add_argument('--gpu_no', type=str, required=True, default='0')    
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank for distributed training")  # Add this line
    args = parser.parse_args()   

    exp_name = args.experiment_name
    train_audio_list = args.train_audio_list
    test_audio_list = args.test_audio_list

    train_preprocessed_audio_list = np.load(train_audio_list, allow_pickle=True).tolist()[:]
    test_preprocessed_audio_list = np.load(test_audio_list, allow_pickle=True).tolist()[:]


    # Create the Hugging Face Dataset from the data
    train_dataset = Dataset.from_dict({'audio': [d['audio'] for d in train_preprocessed_audio_list],
                                       'sentence': [d['sentence'] for d in train_preprocessed_audio_list]})
    
    test_dataset = Dataset.from_dict({'audio': [d['audio'] for d in test_preprocessed_audio_list],
                                      'sentence': [d['sentence'] for d in test_preprocessed_audio_list]})
    
    # Optionally, create a DatasetDict if you have splits
    dataset_dict = DatasetDict({
                                'train': train_dataset,
                                'test': test_dataset})
    
    # Check the DatasetDict
    print(dataset_dict)

    model_name = args.model_name
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="vietnamese", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="vietnamese", task="transcribe")

    vectorized_datasets_train = dataset_dict["train"].map(prepare_dataset, num_proc=1, remove_columns=dataset_dict.column_names["train"],
                                                          batch_size=8, cache_file_name='train_cache/tmp', load_from_cache_file=True)
    vectorized_datasets_test = dataset_dict["test"].map(
                                                        prepare_dataset, num_proc=1, remove_columns=dataset_dict.column_names["train"],
                                                        batch_size=8, cache_file_name='test_cache/tmp', load_from_cache_file=True)

    cache_folders = ['train_cache', 'test_cache']
    for cache_folder in cache_folders:
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
    
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "vietnamese"
    model.generation_config.task = "transcribe"
    
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{exp_name}",  # change to a repo name of your choice
        per_device_train_batch_size=8, #16
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8, #8
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000, #1000 
        eval_steps=1000, #1000
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        # ddp_find_unused_parameters=False
        deepspeed="ds_config.json"
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets_train, #viet_dataset_dict["train"],
        eval_dataset=vectorized_datasets_test, #viet_dataset_dict["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    
