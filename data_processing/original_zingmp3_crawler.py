import requests
import re
import subprocess
import sys
import os
import tqdm

print('Current working directory:', os.getcwd())

with open('./sample_tracks_candidates.txt', 'r', encoding='utf-8') as file:
  samples = file.read().strip().split('\n')
print("Total:", len(samples))

def get_link(query):
  """
  Get the link of the song from zingmp3.vn based on the search query
  
  Parameters
  ----------
  query : str
    The search query
  
  Returns
  -------
  link : str
    The link of the song, or None if the link is not found
  """
  regex = r"https:\/\/mp3\.zing\.vn\/bai-hat\/.*.html"
  url = "https://mp3.zing.vn/tim-kiem/bai-hat?q=" + query
  print(url)
  response = requests.request("GET", url, headers={}, data={})
  matches = re.finditer(regex, response.text, re.MULTILINE)
  try:
    link = list(matches)[0][0]
  except:
    link = None
  return link

def craw_song(link):
  """
  Crawl a song from zingmp3 based on the link
  This function will use yt-dlp to download the song and its lyrics
  The function will return a list of file path of the downloaded files
  """
  result_crawl = subprocess.run(['./yt-dlp', '--write-subs', link], stdout=subprocess.PIPE)
  regex = r"\[download\] Destination: .*"
  matches = re.finditer(regex, result_crawl.stdout.decode("utf-8"), re.MULTILINE)
  list_file = [item[0].replace('[download] Destination: ', '') for item in list(matches)]
  return list_file

for idx in tqdm.tqdm(range(len(samples))):
  try:
    query = ' '.join(samples[idx].split('\t')[:-1])
    print(query)
    song_name = samples[idx].split('\t')[-1]
    
    link = get_link(query)

    print(link)

    
    if link is not None:
      print(song_name)
      print(link)
      list_file = craw_song(link)
      if len(list_file) > 1:
        for file_name in list_file:
          new_file_name = song_name + "." + file_name.split('.')[-1]
          os.rename(file_name, new_file_name)
          print("done file", new_file_name)
  except KeyboardInterrupt:
    print("error", query)
    sys.exit(0)
  except:
    print("error", query)
    pass