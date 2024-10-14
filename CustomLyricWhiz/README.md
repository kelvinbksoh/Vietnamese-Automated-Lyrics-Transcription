# LyricWhiz
This is a replicated LyricWhiz repo from https://github.com/zhuole1025/LyricWhiz

## Project structure
```bash
├── code                                 
│   ├── run_whisper.py                   # Generate transcription from whisper model
│   └── chatgpt_select.py                # Ask Chatgpt to select the best lyrics
├── sample                               # audio files
│   └── *.mp3                            # audio 
├── results                              # Whisper transcription results
│   └── *.json                           # Whisper transcription
├── post_processed_transcripts           # Chatgpt postprocessed transcripts 
│   └── *.json                           # Select Whisper transcription only
```
## Whisper Transcription
To transcribe lyrics using Whisper, run the following command:
```
python code/run_whisper.py
```
After running `run_whisper.py` the generated Whisper transcription will be stored in the `results` folder.
```bash
├── results                              # Whisper transcription results
│   └── *.json                           # Whisper transcription
```
## ChatGPT Post-Processing (Require OPENAI_API_KEY)
**[NOTE]:** Please set your `OPENAI_API_KEY` in **"chatgpt_select.py"** in **line 14** of the code before running the following script

To post-process the Whisper output using ChatGPT, run the corresponding Python script in the `./code` folder:
```
python code/chatgpt_select.py
```
After running the above code, it will use gpt4-o to select the best lytics according to the list of lyrics options provided
```bash
├── post_processed_transcripts           # Chatgpt postprocessed transcripts 
│   └── *.json                           # Select Whisper transcription only
```


## Citation
```
@article{zhuo2023lyricwhiz,
  title={LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to ChatGPT},
  author={Zhuo, Le and Yuan, Ruibin and Pan, Jiahao and Ma, Yinghao and LI, Yizhi and Zhang, Ge and Liu, Si and Dannenberg, Roger and Fu, Jie and Lin, Chenghua and others},
  journal={arXiv preprint arXiv:2306.17103},
  year={2023}
}
```
  
```
@article{yuan2023marble,
  title={MARBLE: Music Audio Representation Benchmark for Universal Evaluation},
  author={Yuan, Ruibin and Ma, Yinghao and Li, Yizhi and Zhang, Ge and Chen, Xingran and Yin, Hanzhi and Zhuo, Le and Liu, Yiqi and Huang, Jiawen and Tian, Zeyue and others},
  journal={arXiv preprint arXiv:2306.10548},
  year={2023}
}
```