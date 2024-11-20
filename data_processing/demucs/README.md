# Demucs
This is a replicated LyricWhiz repo from https://github.com/zhuole1025/LyricWhiz

## Setup Instructions - Local setup using Ananconda
1. Git clone the official latest demucs repo - [Github](https://github.com/adefossez/demucs?tab=readme-ov-file)
    ```bash
    git clone https://github.com/adefossez/demucs.git
    ```

2. Use the modified environment file `environment-cuda.yml` to proceed setting the `demucs` environment.
    ```
    conda env update -f environment-cuda.yml # if you have GPUs
    conda activate demucs
    pip install -e .
    ```

## Usage
1. `test_convert.ipynb` file used to run demucs from script.

    * Note: Source-separate files will follow the following file structure:

        ```bash
        ├── output_dir_name                                 
        ├── demucs_model_name # Available models: htdemucs (default), htdemucs_ft, htdemucs_6s, hdemucs_mmi, mdx, mdx_extra, mdx_q, mdx_extra_q, SIG
        │       └── song_name_1
        |           └── vocals.mp3 # Example based on --two-stems=vocals option
        |           └── no_vocals.mp
        |       └── song_name_2
        |           └── vocals.mp3 # Example based on --two-stems=vocals option
        |           └── no_vocals.mp
    
        ```

2. Alternatively, demucs can be run from the terminal. See some examples from official repo:
    ```bash
    demucs PATH_TO_AUDIO_FILE_1 [PATH_TO_AUDIO_FILE_2 ...]   # for Demucs

    # If you used `pip install --user` you might need to replace demucs with python3 -m demucs
    python3 -m demucs --mp3 --mp3-bitrate BITRATE PATH_TO_AUDIO_FILE_1  # output files saved as MP3
            # use --mp3-preset to change encoder preset, 2 for best quality, 7 for fastest

    # If your filename contain spaces don't forget to quote it !!!
    demucs "my music/my favorite track.mp3"

    # You can select different models with `-n` mdx_q is the quantized model, smaller but maybe a bit less accurate.
    demucs -n mdx_q myfile.mp3

    # If you only want to separate vocals out of an audio, use `--two-stems=vocals` (You can also set to drums or bass)
    demucs --two-stems=vocals myfile.mp3
    ```

## Citation
```
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```
