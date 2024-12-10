import re
import tqdm
import random
from vnmese_utils import is_vietnamese
# Some heuristic to remove duplicates, remixes, cuz searches would get them anyway

RANDOM_SEED = 314
random.seed(RANDOM_SEED)

if __name__ == '__main__':
    with open('./in/track_candidates.txt', 'r', encoding='utf-8') as file:
        samples = file.read().strip().split('\n')       

    print(len(samples))

    new_list = []
    cur_song = None
    for sample in tqdm.tqdm(samples):
        song_name, artist_name, mashed_str = sample.split('\t')
        # song_name = remove_specific_parentheses(song_name) #Remove different version of same song
        if 'remix' in song_name or 'version' in song_name or 'cover' in song_name or 'live' in song_name:
            continue

        #If number is in name, remove
        if any(char.isdigit() for char in song_name):
            continue

        if not is_vietnamese(song_name):
            continue

        if "(" in song_name:
            main_part = song_name[:song_name.index("(")]
            if not is_vietnamese(main_part):
                continue

        if song_name != cur_song:
            new_list.append(sample)
            cur_song = song_name
    
    print(len(new_list))
    #Shuffle
    random.shuffle(new_list)
    
    with open('./in/filtered_tracks_candidates.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(new_list))
            

    


    



