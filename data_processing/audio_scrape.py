import requests
import re
import subprocess
import sys
import os
import tqdm
import time
import pandas as pd
import random
import csv
from csv import DictWriter

# from requests_html import HTMLSession
import yt_dlp

'''

Sample:
- About nhaccuatui.com
0 giờ 2 phút (remix 2)	dj silly fox lý tuấn kiệt	0-gio-2-phut-remix-2-ly-tuan-kiet-dj-silly-fox

https://www.nhaccuatui.com/bai-hat/0-gio-2-phut-dj-silly-fox-remix-ly-tuan-kiet.sDUbUhVwydsT.html

Some songs dont have lyrics
The 2 disappear at the end

yt_dlp doesn't work for nhaccutui
yt_dlp.utils.DownloadError: ERROR: Unsupported URL: https://www.nhaccuatui.com/bai-hat/0-gio-2-phut-ly-tuan-kiet.eyEEpK2zydhU.html
Probably have to use some API, there are a lot but have not tried.

- About zingmp3.vn

https://zingmp3.vn/album/0-Gio-2-Phut-Ly-Tuan-Kiet/ZWZCWED6.html
can be downloaded by yt-dlp, if u have the link

to get the link, using requests or requests_html have not worked. They pull the raw html of the search page and use regex to get the link, 
but the raw html doesn't have the links to songs
ChatGPT suggested that i use playwright or Selenium to open a browser in the background, but that would take some time. pls look around for a better method


'''

def download_song(link, folder_path, max_file_size=10000000, only_metadata=False):
    '''
    # Example usage
    # download_song('https://zingmp3.vn/bai-hat/Danh-Om-Xot-Xa-Luu-Chi-Vy/ZW6CUUDE.html', './out')
    # download_song('https://zingmp3.vn/album/0-Gio-2-Phut-Ly-Tuan-Kiet/ZWZCWED6.html', './out')
    '''

    os.makedirs(folder_path, exist_ok=True)

    ydl_opts = {
        'ratelimit': 20000000, # Maximum download rate in bytes per second (Dependent on internet speed)
        'throttledratelimit': 100000, # Minimum download rate in bytes per second below which throttling is assumed and the video data is re-extracted, e.g. 100K
        'retries': 10, # Number of retries
        'buffersize': 512, # Size of download buffer, e.g. 1024 or 16K
        'http_chunk_size': None, # Size of a chunk for chunk-based HTTP downloading, e.g. 10485760 or 10M (default is disabled). May be useful for bypassing bandwidth throttling imposed by a webserver (experimental)
        'writesubtitles': True,
        'format': 'bestaudio/best',  # Adjust as needed
        'outtmpl': os.path.join(folder_path, '%(title)s.%(ext)s'),  # Specify output filename template
        'skip_download':True
    }

    # if use_google_cached:
    #   link = 'http://webcache.googleusercontent.com/search?q=cache:' + link
    start_time = time.time()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        song_metadata = ydl.extract_info(url=link, download=False)
        if only_metadata:
          print(f'Elapsed Time (in seconds): ', time.time() - start_time)
          return song_metadata
        if song_metadata['subtitles'] != None or song_metadata['filesize_approx'] <= max_file_size: # skip song download if there is no subtitles available or if the filesize exceeds threshold (e.g. 10MB)
          ydl.download([link])
          print(f'Elapsed Time (in seconds): ', time.time() - start_time)

def download_all_songs(file_name:str, limit=10, last_stop_idx=None):

  download_df = pd.read_csv(file_name)
  download_ls = download_df['download_link'].to_list()

  if last_stop_idx:
    download_ls = download_ls[last_stop_idx:]

  start_time = time.time()
  for idx, link in enumerate(download_ls):
    if idx < limit:
      try:
        download_song(link, './out')
      except:
        print('last_stop_idx: ', idx)

  print(f'Total Elapsed Time (in seconds): ', time.time() - start_time)
  return

def download_all_metadata(file_name:str, limit=10, last_stop_idx=None):

  download_df = pd.read_csv(file_name)
  download_ls = download_df['link'].to_list()

  if last_stop_idx:
    download_ls = download_ls[last_stop_idx:]

  init_time = time.time()
  for idx, link in enumerate(download_ls):
    start_time = time.time()
    if idx < limit:
      time.sleep(random.uniform(1, 4))
      try:
        metadata = download_song(link, './out', only_metadata=True)
        new_row = {
          'title': metadata['title'],
          'fulltitle': metadata['fulltitle'],
          'duration': metadata['duration'],
          'artist': metadata['artist'],
          'album': metadata['album'],
          'abr': metadata['abr'],
          'vbr': metadata['vbr'],
          'tbr': metadata['tbr'],
          'release_year': metadata['release_year'],
          'filesize_approx': metadata['filesize_approx'],
        }
        with open('./out/metadata.csv', 'a') as f_object:
          dictwriter_object = DictWriter(f_object, fieldnames=list(new_row.keys()))
          dictwriter_object.writerow(new_row)
          f_object.close()
          print(f'Total Elapsed Time (in seconds): ', time.time() - start_time)
      except Exception as e:
        if last_stop_idx:
          song_index = last_stop_idx + idx
        else:
          song_index = idx
        with open('out/download_failures.csv','a+', newline ='') as file:
          dictwriter_object = DictWriter(file, fieldnames=['idx','reason'])
          dictwriter_object.writerow({'idx':song_index, 'reason':e})
          file.close()
        print('last_stop_idx: ', song_index)
        print(f'Total Elapsed Time (in seconds): ', time.time() - start_time)

  print(f'Total Elapsed Time (in seconds): ', time.time() - init_time)
  return

if __name__ == "__main__":

  # download_song('https://zingmp3.vn/bai-hat/LK-Tru-Tinh-Chon-Loc-2023-Dong-Dao/Z779DIDA.html', './out', only_metadata=True) 
  # download_all_songs('out/spedup_downloads.csv', limit=10)

  download_all_metadata('in/8k_songs.csv', limit=8000, last_stop_idx=3870)
  # download_all_metadata('in/8k_songs.csv', limit=8000, last_stop_idx=5000)