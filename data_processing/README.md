# TOBE REFACTOR IN DECEMBER 2024

# vietnamese-song-scraping
Code to scrape songs and lyrics from Zing mp3 and other preprocessing

# Installation
pip install -r requirements.txt
playwright install


SCRAPING FLOW:
- Iterate over sample_tracks_candidates
    + each line has song name in Vietnamese + artist + search term to guarantee to get that song if replace - with space
- Search that song with request, which will return a search result page:
    + Which looks like this https://zingmp3.vn/tim-kiem/bai-hat?q=0%20gio%202%20phut%20ly%20tuan%20kiet
- Pull the search page's elements (or render the page locally if too much javascript then pull the elements)
- Extract the link to the song:
    + Which looks like this https://zingmp3.vn/bai-hat/0-Gio-2-Phut-Ly-Tuan-Kiet/ZW7OCIDE.html (It's the ID that we don't have)
- Download using the link



.py Files:
- original_zingmp3_crawler.py: original scraping file
- url_scrape.py: test rendering 1 search page
- audio_scrape: test downloading 1 song

Output files, processed chronologically:
- in/sample_tracks_candidates.txt: Original track names
- in/filtered_sample_tracks.txt: filtered from sample_tracks_candidates
- out/all_scraped.csv: All urls scraped from in/filtered_sample_tracks.txt, up to Khoc-Lan-Cuoi
- filtered_links.csv: filtered urls, remove possible english 
- unique_songname_link: For each prefix in filtered_links.csv only take 1 song
- master_1st_12000: Unique songs and whether lyrics exist
- master_1st_1000: Extract 1000 songs
- master_1st_100: Extract 100