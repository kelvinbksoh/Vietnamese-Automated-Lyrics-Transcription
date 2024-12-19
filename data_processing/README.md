# Data Processing

# vietnamese-song-scraping
Code to scrape songs and lyrics from Zing mp3 and other preprocessing

# Installation
pip install -r requirements.txt
playwright install


# Scraping Flow:
Based on the original `original_zingmp3_crawler.py` from [nguyenvulebinh's python zingmp3 crawler](https://github.com/nguyenvulebinh/lyric-alignment/tree/main/data_preparation), which, as of Oct 2024, no longer works. We follow the same flow:

- Iterating through in/filtered_tracks_candidates.txt, use `playwright` to render the zingmp3's page and extract the song links in batches into csvs:
```
python url_scrape.py
```
This will generate a various master_links.csv containing links like `https://zingmp3.vn/bai-hat/0-Gio-2-Phut-Ly-Tuan-Kiet/ZW7OCIDE.html`, where they are further filtered based on constraints.

- Then iterating through the .csv file (for example, the `in/8k_songs.csv`), yt_dlp was used to download the song, lyrics, along with the metada.
```
python audio_scrape.py
```

# Preprocessing:
Experiments are performed in 
- `1_EDA.ipynb`: EDA on full dataset
- `2_preprocessing.ipynb`: Experimentation and pre-processing of songs and lyrics in
- `demucs/test_convert.ipynb`: Experimentation with `demuc` 
- `3_silence_suppression.ipynb`: Experimentation with silence suppression on demuc-ed data