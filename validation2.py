from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import wikipedia
#import wikipediaapi
import string
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read in database of TED talks
client = MongoClient()
db = client['TedData']
posts = db.posts
docs = posts.find({})

# Initialize wikipedia API
#wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

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


# Build corpus
all_docs = []
all_titles = []
for d in docs:
	dfeat = d['headline'] + ' ' + d['description'] + ' ' + d['tags'] + d['transcript']
	dhead = d['headline']

	lowers_dfeat = dfeat.lower()
	lowers_dhead = dhead.lower()

	lowers_dfeat = lowers_dfeat.replace(',', ', ')
	lowers_dhead = lowers_dhead.replace(',', ', ')

	no_punct_dfeat = ''.join( c for c in lowers_dfeat if c not in list(string.punctuation))

	no_punct_dhead = ''.join( c for c in lowers_dhead if c not in list(string.punctuation))
	
	all_docs.append( no_punct_dfeat )
	all_titles.append( no_punct_dhead )
	

tfidf_docs = TfidfVectorizer( tokenizer=tokenize, stop_words='english')
tfs_docs = tfidf_docs.fit_transform( all_docs )

tfidf_titles = TfidfVectorizer( tokenizer=tokenize, stop_words='english')
tfs_titles = tfidf_titles.fit_transform( all_titles )


#sample_pages = ['Climate Change','Blockchain','Bitcoin','Power pose', 'Pixar']
NUM_ARTICLES = 1000
sample_pages = [wikipedia.random() for i in range(NUM_ARTICLES)]
all_doc_cosine_sims = []
all_title_cosine_sims = []
for sp in sample_pages:
	try:
		#page = wikipedia.page(sp.title())
		page = wikipedia.page(sp)
	except:
		continue
	
	text = page.content.lower()
	wikitext = ''.join( c for c in text if c not in list(string.punctuation) )

	wiki_docs = tfidf_docs.transform([wikitext])
	cosine_similarities_docs = linear_kernel( wiki_docs, tfs_docs).flatten()
	#cosine_similarities_docs = cosine_similarity( wiki_docs, tfs_docs).flatten()

	wiki_titles = tfidf_docs.transform( [page.title] )
	cosine_similarities_titles = linear_kernel( wiki_titles, tfs_docs).flatten()
	#cosine_similarities_titles = cosine_similarity( wiki_titles, tfs_docs).flatten()

	#all_doc_cosine_sims = all_doc_cosine_sims + list(np.arccos(cosine_similarities_docs))
	#all_title_cosine_sims = all_title_cosine_sims + list(np.arccos(cosine_similarities_titles))

	all_doc_cosine_sims = all_doc_cosine_sims + list(1.0 - cosine_similarities_docs)
	all_title_cosine_sims = all_title_cosine_sims + list(1.0-cosine_similarities_titles)
#all_doc_cosine_sims = 1.0 - all_doc_cosine_sims
#all_title_cosine_sims = 1.0 - all_title_cosine_sims

sns.distplot( all_doc_cosine_sims, bins=1000, kde=False)
sns.distplot( all_title_cosine_sims, bins=1000, kde=False)
plt.show()
