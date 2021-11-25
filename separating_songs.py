import pandas as pd
df = pd.read_json('ts-lyrics.json')
happy_df = df[df['emotion'] == 'joy']
sad_df = df[df['emotion'] == 'sadness']
anger_df = df[df['emotion'] == 'anger']
happy_df = happy_df.reset_index()
sad_df = sad_df.reset_index()
anger_df = anger_df.reset_index()
happy_df.to_json('happy_song.json')
sad_df.to_json('sad_song.json')
anger_df.to_json('anger_song.json')