import pandas as pd

# Load the dataset to inspect its contents
file_path = 'out/filtered_links.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
df.head()

# To achieve the desired result, I'll group the dataset by 'song_name' and randomly select one link for each song.
# This can be done using the groupby method and applying a random selection within each group.

# Filter the dataset by selecting one random link for each song_name
# df_filtered = df.groupby('song_name').apply(lambda x: x.sample(1)).reset_index(drop=True)
df_filtered = df.groupby('song_name').first().reset_index()

df_filtered = df_filtered.sample(frac=1)

df_filtered.to_csv('out/unique_songname_links.csv', index = None)