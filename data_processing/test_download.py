import yt_dlp
import os

folder_path = 'out'
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

link = 'https://zingmp3.vn/album/Ai-Se-Cam-Thong-Single-Nhat-Vu/ZZZ6CBU9.html'
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([link])
