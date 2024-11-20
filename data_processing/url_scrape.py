import urllib
import pandas as pd
# from fuzzywuzzy import fuzz
from playwright.sync_api import sync_playwright
# from playwright_stealth import stealth_sync
import unicodedata
import tqdm
import random
import os
import time


MAIN_SITE = 'https://zingmp3.vn'
BANG_XOA_DAU = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 + "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5
)

RANDOM_SEED = 314
random.seed(RANDOM_SEED)


def remove_diacritics(txt: str) -> str:
    if not unicodedata.is_normalized("NFC", txt):
        txt = unicodedata.normalize("NFC", txt)
    return txt.translate(BANG_XOA_DAU)


def get_song_link(page, search_link, song_prefix):
    '''
    Input:
    page: the browser page object (reused)
    link: link to the search result of the song
    song_name: name of the song, in Vietnamese
    Output:
    Best matching link to the song
    '''
    full_link = 'Not Found'
    
    # Navigate to the search result page
    # print('Link', search_link)
    page.goto(search_link)
    #, wait_until='networkidle') #IF WAIT, we for sure getting all songs. 
    # But it will double, even tripple scraping time 

    # Wait for the content to load and grab the link
    all_links = page.query_selector_all('a[href*="/bai-hat/"]')
    
    matching_links = []
        
    # Loop through each link and filter those ending with ".html"
    for link in all_links:
        href = link.get_attribute('href')
        if href and href.endswith(".html"):
            full_link = f"{MAIN_SITE}{href}"
            matching_links.append(full_link)

    return matching_links


def generate_dloads(samples, limit=None, batch_size = 5, out_csv = 'out/downloads.csv'):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    data = {'song_name':[], 'prefix': [], 'download_link':[]} #'query':[], 
    df = pd.DataFrame.from_dict(data)
    df.to_csv(out_csv, index=False)
                  
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Launch the browser once
        page = browser.new_page()  # Create a new page once
        # stealth_sync(page) # uncomment to use playwwight-stealth for captchas
        
        for idx, file in enumerate(tqdm.tqdm(samples)):
            song_name, artist_name, mashed_str = samples[idx].split('\t')

            prefix = remove_diacritics(song_name).replace(' ', '-')
            prefix = ''.join([char.upper() if i > 0 and prefix[i-1] == '-' or i == 0 else char for i, char in enumerate(prefix)])

            # search_link = f'http://webcache.googleusercontent.com/search?q=cache:https://zingmp3.vn/tim-kiem/bai-hat?q={song_name.replace(" ", "%20")}'
            search_link = f'https://zingmp3.vn/tim-kiem/bai-hat?q={song_name.replace(" ", "%20")}'

            # Use the same page object to get the song link   
            
            # if (idx + 1) % batch_size == 1:
            #     time.sleep(random.uniform(0.5, 1.5)) #Wait, to fight ddos defense          
            
            accepted_links = get_song_link(page, search_link, prefix)

            data['download_link'].extend(accepted_links)
            data['song_name'].extend([song_name] * len(accepted_links))
            data['prefix'].extend([prefix] * len(accepted_links))            
            
            # Write to CSV in batches
            if (idx + 1) % batch_size == 0:
                time.sleep(random.uniform(2, 5)) #Wait, to fight ddos defense 
                df = pd.DataFrame.from_dict(data)
                df.to_csv(out_csv, mode='a', header=False, index=False)
                data = {'song_name': [], 'prefix': [],  'download_link': []}  # Reset the batch data

        # Write any remaining data not written in the last batch
        if data['song_name']:
            df = pd.DataFrame.from_dict(data)
            df.to_csv(out_csv, mode='a', header=False, index=False)

        browser.close()  # Close the browser once at the end

if __name__ == '__main__':
    # link = 'https://zingmp3.vn/tim-kiem/bai-hat?q=0%20gi%E1%BB%9D%202%20ph%C3%BAt%20l%C3%BD%20tu%E1%BA%A5n%20ki%E1%BB%87t'
    # song_name =  '0-gio-2-phut-ly-tuan-kiet' #'0-Gio-2-Phut-Ly-Tuan-Kiet'
    # get_song_link(link, song_name)

    filepath = 'in/filtered_sample_tracks.txt'
    with open(filepath, 'r', encoding='utf-8') as file:
        samples = file.read().strip().split('\n')
    
    # samples = samples[:212]

    # Batching
    proc_bs = 100
    for i in range(0, len(samples), proc_bs):
        generate_dloads(samples[i:i+proc_bs], out_csv=f'out/downloads_{i}.csv')