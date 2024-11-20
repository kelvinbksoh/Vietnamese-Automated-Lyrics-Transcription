# Installation
Create conda environment with python>=3.9 (Ensure google genai package works) and ensure you have pytorch installed
Run
```
pip install -r requirements.txt
```

# Run
Run
```
streamlit run app.py
```

# Note
Set your GEMINI_KEY in env. Click on LLM correction, for Gemini to correct the lyrics. Note that this does not help with errors, but will better format the output.

Please do not reload (F5) the page in browser. Instead shutdown the app and rerun the app again from terminal.
Please wait till end of inference before reloading the model or putting in another file.

Model Whisper large-v2 should require 10GB, medium requires 6GB and small requires 3GB in GPU VRAM.
