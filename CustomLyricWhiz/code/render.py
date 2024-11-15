import os
import json
import pprint
import re

def convert_timestamp(seconds_start, seconds_end):
    # Convert the start timestamp to minutes and seconds    
    minutes_start = int(seconds_start // 60)
    seconds_start_remainder = seconds_start % 60
    
    # Format the start timestamp to MM:SS.SS
    formatted_start = f"{minutes_start:02}:{seconds_start_remainder:05.2f}"
    
    # Convert the end timestamp to minutes and seconds (optional if needed)
    minutes_end = int(seconds_end // 60)
    seconds_end_remainder = seconds_end % 60
    
    # Format the end timestamp to MM:SS.SS (optional if needed)
    formatted_end = f"{minutes_end:02}:{seconds_end_remainder:05.2f}"
    
    # Return the formatted string
    return f"[{formatted_start}]"

def split_combined_word(word):
    # Use regular expression to split at the capital letter in the middle
    split_words = re.findall(r'[a-z]+|[A-Z][a-z]*', word)
    return split_words

def chunk_lyrics(chunks, segment_length=20):
    "Chunk the lyrics into 30-second chunks, 0.5 seconds overlap."
    # chunks = [chunks[i:i+segment_length] for i in range(0, len(chunks), segment_length)]
    start_sec = 0
    output_chunks = []
    
    output_dict = {
        'text': '',
        'timestamp': [],
    }

    for index,  text_dict in enumerate(chunks):
        start_sec = text_dict['timestamp'][0]
        end_sec = text_dict['timestamp'][1]

        if index == 0: start_of_chunk = text_dict['timestamp'][0]
        
        end_of_chunk = start_of_chunk + segment_length

        if index == 0:
            output_dict['text'] += text_dict['text'] + ' '
            output_dict['timestamp'] = [start_sec, end_sec]

        elif index == len(chunks) - 1:
            output_dict['text'] += text_dict['text']
            output_dict['timestamp'][1] = end_sec
            output_chunks.append(output_dict)
            
        elif end_of_chunk >= end_sec:
            output_dict['text'] += text_dict['text'] + ' '
            output_dict['timestamp'][1] = end_sec
        
        else:
            temp_dict = output_dict.copy()
            output_chunks.append(temp_dict)    
            #Reset
            output_dict['text'] = text_dict['text'] + ' '
            output_dict['timestamp'] = [start_sec, end_sec]
            start_of_chunk = start_sec            
    
    return output_chunks
    
def output_render(
    json_result,
    format: str = 'phowhisper',
    ):

    if format == 'phowhisper':
        render_output = []
        for text_dict in json_result:
            cur_text = convert_timestamp(text_dict['timestamp'][0], text_dict['timestamp'][1]) + ' ' + text_dict['text']
            render_output.append(cur_text)

        render_output = '\n'.join(render_output)

    elif format == 'whisper':
        chunks = json_result['chunks']
        chunks = chunk_lyrics(chunks, segment_length=30)
        render_output = output_render(
            chunks,
            format='phowhisper'
        )
    
    return render_output
    