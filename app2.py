import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import wikipedia
import wikipediaapi
import string
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
import pandas as pd
from youtube_videos import youtube_search

# Load mongo database
client = MongoClient()
db = client['TedData']
posts = db.posts
docs = posts.find({})

punctuation =  list(string.punctuation)

stemmer = PorterStemmer()
def stem_tokens( tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append( stemmer.stem(item))
	return stemmed

def tokenize(text):
	tokens = word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

tfidf = TfidfVectorizer( tokenizer=tokenize, stop_words='english')

wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

#app = dash.Dash()

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)


app.layout = html.Div(children=[

    #html.H1(children='TED Recommender', style={'textAlign':'center'},className='app-header'),
	html.Div(children=[
		html.Img(src='/assets/logo.jpg',style={'textAlign':'center'}),
		html.H5(children='TED talk suggestions based on Wikipedia articles'),
	], style={'textAlign':'center'}
	),

	html.Div( children=[
		html.H3(children='Enter Wikipedia article', style={'textAlign':'center'}),
		html.Div([
			dcc.Input(id='wikisearch', value='', type='text'),
			html.Button(id='submit-button', n_clicks=0, children='Submit'),
		], style={'textAlign':'center'}),
		
		html.Br([]),
		
		html.Div(id='recommendation', style={'textAlign':'center'}),

		html.Div(id='segments', style={'textAlign':'center'})
	])
])

def rec_ted_talk( wikistring ):
	search_list = wikipedia.search( wikistring )
	page = wiki.page( search_list[0] )
	text = page.text.lower()
	wikitext = ''.join( c for c in text if c not in punctuation )
	all_docs = [wikitext]
	all_ids = [None]

	for d in docs:
		dfeat = d['headline'] + ' ' + d['speaker'] + ' ' + d['description'] + ' ' + d['tags']
		lowers_dfeat = dfeat.lower()
		lowers_dfeat = lowers_dfeat.replace(',', ', ')
		no_punct = ''.join( c for c in lowers_dfeat if c not in punctuation)
		all_docs.append( no_punct )
		all_ids.append( d['_id'] )
	
	docs.rewind()
	
	tfs = tfidf.fit_transform( all_docs )

	cosine_similarities = linear_kernel( tfs[0:1], tfs).flatten()
	related_docs_indices = cosine_similarities.argsort()

	first_match = related_docs_indices[-2] 
	return [page, all_ids[first_match] ]

def rec_ted_timestamp( page, ted_id ):
	wiki_intro = page.summary.lower()
	wiki_intro = ''.join( c for c in wiki_intro if c not in punctuation)
	wiki_sections = [ wiki_intro ]

	sections = page.sections
	for s in sections:
		section_text = s.title + " " + s.text
		section_text = section_text.lower()
		no_punct = ''.join( c for c in section_text if c not in punctuation )
		wiki_sections.append( no_punct )

	# Search through first match
	section_time_match = [] 

	# cosine similarity of timestamp with the largest cosine similarity.
	max_cos_sim = []
	
	best_doc = posts.find_one({'_id': ted_id})
	ns = best_doc['num_segments']

	if (ns > 0 ):
		timestamps = best_doc['timestamps'].split()

		# List to store wikipedia section and ted talk segments.
		ws_plus_ted = []
		for i in range(ns):
			ted_text = best_doc['seg'+str(i)]
			lowers = ted_text.lower()
			no_punct = ''.join( c for c in lowers if c not in punctuation )
			ws_plus_ted.append( no_punct )


		# Compare wikipedia sections to ted talk segments.
		for ws in wiki_sections:
			ws_plus_ted.insert(0,ws)
		
			tfs = tfidf.fit_transform( ws_plus_ted )
			cosine_similarities = linear_kernel( tfs[0:1], tfs).flatten()
			related_section_indices = cosine_similarities.argsort()

			best_time_match_index = related_section_indices[-2] -1
		
			#print( cosine_similarities)
			#print( cosine_similarities[best_time_match_index])
			max_cos_sim.append( cosine_similarities[best_time_match_index] )
			
			section_time_match.append( timestamps[best_time_match_index] )
			#print( ws_plus_ted[ best_time_match_index ] )
			del ws_plus_ted[0]
	
	return sections, section_time_match, max_cos_sim

def generate_table(page, section_time_match, max_cos_sim):
	section = page.sections
	d = {}
	d['Section'] = ['Introduction']
	d['Wikipedia Text'] = [page.summary]
	if (len(section_time_match) > 0):
		d['Time stamp'] = [section_time_match[1]]
		for i,s in enumerate(section):
			if (s.text != ''):
				d['Section'].append( s.title )
				d['Wikipedia Text'].append( s.text )
				if (s.title == 'External links') or (s.title == 'Further reading') or (s.title == 'References') or (s.title == 'See also') or (max_cos_sim[i]<0.06) :
					d['Time stamp'].append('-')
				else:
					d['Time stamp'].append( section_time_match[i+1] )
	else:
		d['Time stamp'] = ['N/A']
		for i,s in enumerate( section ):
			if (s.text != ''):
				d['Section'].append( s.title )
				d['Wikipedia Text'].append( s.text )
				d['Time stamp'].append('N/A')

	df = pd.DataFrame(data=d)

	return html.Table(
		[html.Tr([html.Th(col) for col in df.columns])] +

		[html.Tr([
			html.Td( df.iloc[i][col]) for col in df.columns
		],style={'textAlign':'left'}) for i in range( len(df) )]
	)


@app.callback(Output('recommendation', 'children'),
			[Input('submit-button', 'n_clicks')],
			[State('wikisearch', 'value')])
def update_output(n_clicks, input_value):
	if (input_value == ""):
		return ""
	page, ted_id = rec_ted_talk( input_value )
	best_doc = posts.find_one({'_id':ted_id})
	sect, stm, max_cos = rec_ted_timestamp(page, ted_id)
	
	#headline = best_doc['headline'].lower().replace(",","").replace(".","").replace("?","").replace("-","_")
	#headline = "_".join( headline.split() )
	#headline = headline.replace("'","_")
	
	#speaker = best_doc['speaker'].lower().replace(",","").replace(".","").replace("?","").replace("-","_")
	#speaker = "_".join( speaker.split() )
	#speaker = speaker.replace("'","_")
	#url = speaker+"_"+headline

	yt_search_string = best_doc['speaker'] + ' ' + best_doc['headline']
	tok, just_json = youtube_search( yt_search_string, max_results=1)
	vidID = just_json[0]['id']['videoId']
	return html.Div(children=[ 
				html.Div(children=[
					#html.Iframe(src='https://embed.ted.com/talks/'+url, width=800, height=400)
					html.Iframe(src='https://www.youtube.com/embed/'+vidID, width=800, height=400)
				]),
				html.Label( [ html.A(best_doc['headline'] + ' by ' + best_doc['speaker'], href=best_doc['URL'],target='_blank') ]),
				generate_table(page,stm, max_cos)
			]) 

if __name__ == '__main__':
    app.run_server(debug=True)
