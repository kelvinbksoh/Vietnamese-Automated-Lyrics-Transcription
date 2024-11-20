import os
from glob import glob 
import pandas as pd
import tqdm
import json
import shutil
import unicodedata

if __name__ == '__main__':
    folder_path = 'out/100-output-test'
    files = glob(f'{folder_path}/*')
    print(len(files))

    file_names = [x.split(folder_path+'/')[-1].split('.mp3')[0].split('.json')[0] for x in files]
    # file_names = [unicodedata.normalize('NFC', x) for x in file_names]

    # df = pd.DataFrame()
    # df['song_name'] = sorted(file_names)
    # df.to_csv('out/test_set_100.csv', index=False)


    new_folder_path = 'out/validation-audio-100-demuc'
    os.makedirs(new_folder_path, exist_ok=True)

    old_folder_path = 'out/validation-audio-100-pp-demucs'
    files_1000 = glob(f'{old_folder_path}/*')
    print(len(files_1000))

    not_demucs = []
    for file in tqdm.tqdm(file_names):
        try:
            shutil.copy(f'{old_folder_path}/{file}/vocals.mp3', f'{new_folder_path}/{file}.mp3')
        except:
            print(file)

        
        # shutil.copy(f'{old_folder_path}/{file}.mp3', new_folder_path)
        # shutil.copy(f'{old_folder_path}/{file}.origin.lrc', new_folder_path)