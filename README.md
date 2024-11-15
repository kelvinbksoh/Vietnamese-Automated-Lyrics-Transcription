# Vietnamese Automated Lyrics Transcription (ALT)
This project aims to perform automatic lyrics transcription on Vietnamese songs, the pre-trained model used for this task is Whisper-large-v2 from [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356).

## Setup
To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Resource
- GPU x2 RTX 3090 24GB
- RAM 128 GB
 
## Training Data
The training dataset consists of 7,000 Vietnamese songs, in total of roughly 550 hours of audio, across various Vietnamese music genres, dialects and accents. Due to IP concerns, the data is not publicly available. Each song includes lyrics along with corresponding line-level timestamps, enabling precise mapping of audio segments to their respective lyrics based on the provided timestamp information.

## Data folder structure
```bash
├── training_audio                                      # Training folder              
│   ├── song1.mp3                                       # Song in .mp3 format
│   └── song1.origin.lrc                                # The lyric file of the song
├── validation_audio                                    # Validation folder
│   ├── song2.mp3                                       # Song in .mp3 format
│   └── song2.origin.lrc                                # The lyric file of the song         
├── train                                               # Training pipeline
│   ├── generate_audio_for_whisper.py                   # pre-generate chunks of 30seconds audio
│   ├── whisper_train.py                                # whisper training script
│   ├── ds_config.json                                  # deepspeed config
```
## Preprocess audio into format for Whisper fine-tuning
Generate both training and validation preprocessed audio lists required for Whisper training.

The following command preprocesses the raw audio and lyrics by chunking the original audio into 30-second segments and aligning the timestamp of the lyrics with its audio speech:
```
python train/generate_audio_for_whisper.py --train_dir 'training_audio' \
                                     --test_dir 'validation_audio' \
                                     --audio_arr_dir_name "audio_8k"
```
This will generate two metadata files: train_preprocessed_audio_8k_audio_list.npy and test_preprocessed_audio_8k_audio_list.npy, which are used for Whisper fine-tuning.

## Whisper Fine-Tuning
The Whisper model can be fine-tuned using the following parameters:

### Arguments
- `experiment_name`: The name of the experiment to keep track. The checkpoint will be saved under this folder.
- `train_audio_list`: The preprocessed training audio list from the previous step.
- `test_audio_list`: The preprocessed testing audio list from the previous step.
- `gpu_no`: The GPU device to use for training. Use `"0"` for the first GPU, `"1"` for the second GPU, etc.

### Example
**Train with single GPU**
```bash
python train/whisper_train.py --experiment_name "whisper-large-v2-7k-1k-pp" \
                        --train_audio_list "train_preprocessed_audio_8k_pp_audio_list.npy" \
                        --test_audio_list "test_preprocessed_audio_8k_pp_audio_list.npy" \
```

**Train with multiple GPUs using DeepSpeed**
```python
deepspeed --num_gpus=2 train/whisper_train.py --experiment_name "whisper-large-v2-7k-1k-pp" --train_audio_list "train_preprocessed_audio_8k_pp_audio_list.npy" --test_audio_list "test_preprocessed_audio_8k_pp_audio_list.npy"
```

##  Whisper Evaluation
| **Model**            | **WER (Lowercase)** | **WER (Case-Sensitive)** | **CER (Lowercase)** | **CER (Case-Sensitive)** |
|----------------------|--------------------|--------------------------|--------------------|--------------------------|
| whisper-medium       | 23.15              | 26.42                    | 17.01              | 17.03                    |
| whisper-large-v2     | 20.52              | 24.61                    | 16.09              | 17.14                    |
## Whisper Inference 
Our fine-tuned Whisper model for Vietnamese lyrics transcription is now available on the Hugging Face repository: [whisper-large-v2-vietnamese-lyrics-transcription](https://huggingface.co/xyzDivergence/whisper-large-v2-vietnamese-lyrics-transcription). If you’re interested in performing Vietnamese automatic lyrics transcription, you can use the Transformers pipeline below.

#### Using Transformers Pipeline
To generate the transcription for a song, we can use the Transformers [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline). Chunking is enabled by setting `chunk_length_s=30` when instantiating the pipeline. With chunking enabled, the pipeline can be run with batched inference. It can also be extended to predict sequence level timestamps by passing `return_timestamps=True`. In the following example we are passing `return_timestamps="word"`  that provides precise timestamps for when each individual word in the audio starts and ends.
```python
>>> from transformers import pipeline
>>> asr_pipeline = pipeline(
>>>    "automatic-speech-recognition",
>>>    model="xyzDivergence/whisper-large-v2-vietnamese-lyrics-transcription", chunk_length_s=30, device='cuda',
>>>    tokenizer="xyzDivergence/whisper-large-v2-vietnamese-lyrics-transcription"
>>> )
>>> transcription = asr_pipeline("sample_audio.mp3", return_timestamps="word")
```
## Contribution
This project was made through equal contributions from:
- [Kevin Soh](https://github.com/kelvinbksoh)
- [Bernard Cheng Zheng Zhuan](https://github.com/bernardcheng)
- [Nguyen Quoc Anh](https://github.com/BatmanofZuhandArrgh)
