from openai import OpenAI
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import re
import argparse
import os
import json

from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def chatgpt_processor(instruction_prompt, predictions_dict):
    '''
        This function post-process the generated transcription from Whisper model
        args:
            instruction_prompt: "Lyrics post-processor prompt"
            results: a json file which contains a list of dictionary of the whisper transcription e.g. {"text": .., "segments": .. , "language"..}
            
    '''
    
    context = f'''{instruction_prompt}
    {predictions_dict}
    '''.strip()

    try:
        chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": context}],
                stream=False)
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return {
            "reasons": str(e),
            "closest_prediction": "prediction_0",
            "output": str(e)
        }   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results', help='input directory')    
    parser.add_argument('--output_dir', type=str, default='./post_processed_transcripts', help='output directory')
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    instruction_prompt = """
    Task: As a GPT-4 based lyrics transcription post-processor,
    your task is to analyze multiple ASR model-generated versions
    of a Vietnamese song’s lyrics and determine the most accurate version
    closest to the true lyrics. Also filter out invalid lyrics
    when all predictions are nonsense.
    Input: The input is in JSON format:
    
    {“prediction_1”: “line1;line2;...”, ...}
    Output: Your output must be strictly in readable JSON format
    without any extra text:
    {
    “reasons”: “reason1;reason2;...”,
    “closest_prediction”: <key_of_prediction>
    “output”: “line1;line2...”
    }
    Requirements: For the "reasons" field, you have to provide
    a reason for the choice of the "closest_prediction" field. For
    the "closest_prediction" field, choose the prediction key that
    is closest to the true lyrics. Only when all predictions greatly
    differ from each other or are completely nonsense or meaningless,
    which means that none of the predictions is valid,
    fill in "None" in this field. For the "output" field, you need
    to output the final lyrics of closest_prediction. If the "closest_
    prediction" field is "None", you should also output "None"
    in this field. The language of the input lyrics is English.
    """

    for audio_json in os.listdir(results_dir):
        if audio_json.endswith('.json'):
            with open(results_dir + '/' + audio_json , 'r') as file:
                data = json.load(file)
            print(audio_json)
            predictions_dict = {}
            for i in range(len(data)):
                predictions_dict[f'prediction_{i}'] = data[i]['text']
                print(f"sample #{i}:{data[i]['text']}")
        
            response = chatgpt_processor(instruction_prompt, predictions_dict)
            # ## Sanity check
            # response = '''
            # ```json
            # {
            #     "reasons": "All predictions are identical and make sense.",
            #     "closest_prediction": "prediction_0",
            #     "output": "Shall we go for a walk?"
            # }
            # ```'''
            
            response = json.loads(response.replace("```json", "").replace("```", "").strip())
            print(response)
            best_transcription_idx = int(response['closest_prediction'].split('_')[-1])
            with open(output_dir + '/' + audio_json, 'w') as f:
                data_dict = data[best_transcription_idx]
                chatgpt_dict = {"chatgpt_response": response}
                data_dict.update(chatgpt_dict)
                json.dump(data_dict, f, indent=4, ensure_ascii=False)        
        
if __name__ == '__main__':
    main()


