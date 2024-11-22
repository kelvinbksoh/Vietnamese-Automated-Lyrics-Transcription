# Installation
Create conda environment with python>=3.9 (Ensure google genai package works) and ensure you have pytorch installed
Set your GEMINI_KEY in env if you want to do LLM correction and formatting.

Run
```
pip install -r requirements.txt
```

# Run
Run
```
streamlit run app.py
```
Note that first time running will be longer to pull the model down to local.

# Note
Click on LLM correction, for Gemini to correct the lyrics. Note that this does not help with errors, but will better format the output.

Please do not switch models in 1 session, this has caused unfixed OOM issues.
Please do not reload (F5) the page in browser. Instead shutdown the app and rerun the app again from terminal.
Please wait till end of inference, or any action before starting a new action.

Model Whisper large-v2 should require 10GB, medium requires 6GB and small requires 3GB in GPU VRAM.
