import re
import tqdm
import random
# Some heuristic to remove duplicates, remixes, cuz searches would get them anyway

RANDOM_SEED = 314
random.seed(RANDOM_SEED)
VNCHARS = set("áàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ")

def is_vietnamese(text):
    text = text.lower()
    # Check if any Vietnamese-specific character is in the text
    for char in text:
        if char in VNCHARS:
            return True
    return False

if __name__ == '__main__':
    with open('./in/sample_tracks_candidates.txt', 'r', encoding='utf-8') as file:
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
    
    with open('./in/filtered_sample_tracks.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(new_list))
            

    


    



