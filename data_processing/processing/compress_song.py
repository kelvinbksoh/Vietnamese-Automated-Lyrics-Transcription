import os
import glob
import pandas as pd
from tqdm import tqdm
import zipfile

def compress_to_zip(files, output_zip):
    """
    Compresses up to 1000 songs from a folder into a single ZIP file.
    :param folder_path: Path to the folder containing songs.
    :param output_zip: Path to the output ZIP file.
    :param limit: Maximum number of files to compress (default is 1000).
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Get a list of files in the folder


        # Add each file to the zip
        for file in tqdm(files, desc="Compressing files"):
            zipf.write(file)  # Add the file to the zip archive

    print(f"Compressed {len(files)} files into {output_zip}")


if __name__ == "__main__":
    files = glob.glob('out/master_downloads_1st/*')
    print(len(files))
    
    data = [x.split('out/master_downloads_1st/')[-1] for x in files]
    unique_songnames = list(set([x.split('.mp3')[0].split('.origin.lrc')[0] for x in data]))

    print(len(unique_songnames))
    unique_lyrics = set([x.split('.origin.lrc')[0] for x in data if '.origin.lrc' in x])
    print(len(unique_lyrics))

    df = pd.DataFrame()
    df['song_name'] = unique_songnames
    lyrics_exist = [1 if x in unique_lyrics else 0 for x in unique_songnames]

    df['lyrics_exist'] = lyrics_exist
    df.to_csv('out/master_1st_tally.csv', index=False)


    print('======')
    print('All songs: ', len(data))
    print('Unique songs: ', len(unique_songnames))
    print('Unique songs with lyrics: ', len(unique_lyrics))
    print('Unique songs without lyrics: ', len(unique_songnames) - len(unique_lyrics))


    # Pick the first 1000 songs with lyrics
    df = df[df['lyrics_exist'] == 1]
    df = df.head(1000)
    
    df.to_csv('out/master_1st_tally_1000.csv', index=False)
    
    #Zip the 1000 files
    files_1000 = [f'out/master_downloads_1st/{x}.mp3' for x in df['song_name']]
    files_1000 += [f'out/master_downloads_1st/{x}.origin.lrc' for x in df['song_name']]
    output_zip = 'out/master_1st.zip'
    files_1000 = sorted(files_1000)
    compress_to_zip(files_1000, output_zip)