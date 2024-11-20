import pandas as pd
import tqdm
from glob import glob
import difflib

if __name__ == "__main__":
    # List of CSV file paths to merge
    csv_files = glob('out/downloads_*/*.csv')

    # Initialize an empty DataFrame to hold the merged data
    merged_df = pd.DataFrame()

    # Loop through the CSV files and concatenate them into one DataFrame
    for file in tqdm.tqdm(csv_files):
        df = pd.read_csv(file)
        merged_df = pd.concat([merged_df, df])

    # Remove duplicates based on the 'download_link' column
    merged_df.drop_duplicates(subset='download_link', inplace=True)

    # Optionally, reset the index
    merged_df.reset_index(drop=True, inplace=True)

    # Save the merged and deduplicated DataFrame to a new CSV file
    # merged_df.to_csv('out/all_scraped.csv', index=False)
    
    # Sort to make sure links looking closer are next to each other
    merged_df = merged_df.sort_values(by=["prefix","download_link"], ignore_index=True)

    filtered_data = []
    cur_prefix = ''
    cur_songname = ''
    for i in tqdm.tqdm(range(len(merged_df)-1)):
        download_link = merged_df['download_link'][i]
        prefix = merged_df['prefix'][i]

        if prefix not in download_link:
            pass
        elif prefix in download_link and prefix != cur_prefix:
            cur_prefix = prefix
        else:
            continue
        
        if '(' in merged_df['song_name'][i]:
            continue

        if "instrumental" in download_link.lower():
            continue
        
        url_songname = download_link.split('https://zingmp3.vn/bai-hat/')[-1].split('.html')[0]
        next_url_songname = merged_df['download_link'][i+1].split('https://zingmp3.vn/bai-hat/')[-1].split('/')[0]
        similarity = difflib.SequenceMatcher(None, url_songname, next_url_songname).ratio()
        
        if similarity < 0.6:
            filtered_data.append({
                    'song_name': merged_df['song_name'][i],
                    'download_link': download_link,
                    'prefix': prefix,
                    'similarity_with_next': similarity
                })


    filtered_df = pd.DataFrame(filtered_data)

    filtered_df.to_csv('out/filtered_links.csv', index=False)


