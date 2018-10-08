import pandas as pd
import time
from pymongo import MongoClient
import re
from youtube_videos import youtube_search

# Use regular expressions to parse time stamps.
def isTimeFormat( s ):
	if re.match('[\d]+:[\d]+',s):
		return True
	return False

# Path to TED data stored in CSV file.
df = pd.read_csv('~/tedtalk_data.csv')

ntalks, nfeats = df.shape

all_data = []
for nt in range(ntalks):
	try:
		text = df['transcript'][nt]
		text = text.replace('(Laughter)','').replace('(Applause)','')
		textsplit = text.split()
	except:
		data = {}
		data['headline'] = df['headline'][nt]
		data['speaker'] = df['speaker'][nt]
		data['URL'] = df['URL'][0]
		data['description'] = df['description'][nt]
		data['tags'] = df['tags'][0].replace( ',', ', ')
		data['num_segments'] = 0
		all_data.append( data )
		continue

	if isTimeFormat( textsplit[-1] ):
		textsplit.pop()

	timestamp_indices = []
	bad_time_indices = []

	# Find bad time indices.
	for i in range(len(textsplit)-1):
		if isTimeFormat(textsplit[i]) and isTimeFormat(textsplit[i+1]):
			bad_time_indices.append( i )

	# Remove bad times from textsplit
	textsplit = [ t for i,t in enumerate(textsplit) if i not in bad_time_indices ]

	# Find time indices (only good ones left)
	timestamp_indices = [i for i,t in enumerate(textsplit) if isTimeFormat(t) ]
	timestamps = "".join(textsplit[i]+' ' for i in timestamp_indices[:-1])
	timestamps = timestamps + textsplit[ timestamp_indices[-1] ]
	
	# Construct segments of text
	segments = []
	for i in range( len(timestamp_indices)-1 ):
		segments.append( "".join( t+' ' for t in textsplit[ timestamp_indices[i]+1:timestamp_indices[i+1] ]) )
	
	segments.append( "".join(t+' ' for t in textsplit[timestamp_indices[-1]+1:]) )
	
	num_segments = len(segments)

	
	data = {}
	data['headline'] = df['headline'][nt]
	data['speaker'] = df['speaker'][nt]
	data['URL'] = df['URL'][nt]
	data['description'] = df['description'][nt]
	data['tags'] = df['tags'][nt].replace( ',', ', ')
	data['num_segments'] = num_segments
	data['timestamps'] = timestamps
	for i in range( num_segments ):
		data[ 'seg'+str(i) ] = segments[i]

	# Find youtube video ID for this TED talk
	try:
		search_string = df['headline'][nt] + " " + df['speaker'][nt]
		tok, just_json = youtube_search(search_string, max_results=1)
		vidID = just_json[0]['id']['videoId']

		data['youtube_videoID'] = vidID
	except:
		data['youtube_videoID'] = 'N/A'

	all_data.append(data)

client = MongoClient()
db = client['TedData']
posts = db.posts
result = posts.insert_many( all_data )
print('Multiple posts: {0}'.format(result.inserted_ids))
