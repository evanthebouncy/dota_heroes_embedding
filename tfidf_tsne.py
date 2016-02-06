# @author: manic-mailman@github

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.manifold import TSNE
from jitter import jitter

#reading in the crap, minor preprocessing
responses = []
lemmatizer = WordNetLemmatizer() #lemmatising so arrows -> arrow, hells -> hell, etc.
baddies = set(['Spectre', 'Phoenix\n', 'Io\n'])
with open('data') as f:
	for line in f:
		words = line.split(' ')
		if words[0] not in baddies: #first response is always name 
			line = ' '.join([lemmatizer.lemmatize(word.lower()) for word in words])
			responses.append(line)

#vectorizing, removing all tokens that either show up only once, or on more than 50% of the the hero responses 
#modifying the stoplist with the frequent terms that show up despite the thresholding.
stopwords = set(list(ENGLISH_STOP_WORDS) + 
	['aha', 
	'ahh',
	'wont', 
	'hee', 
	'hoo', 
	'hu', 
	'heah', 
	'ya', 
	'hh', 
	'ti', 
	'aw', 
	'haha', 
	'em', 
	'said', 
	'hey', 
	'id', 
	'le', 
	'hoh', 
	'heheh',
	'gonna',
	'dy',
	'agh'
	'stay',
	'mm',
	'yeah',
	'okay',
	'got',
	'right',
	'aint',
	'sorry',
	'mmm',
	'yep',
	'yea',
	'nice',
	'fun',
	'didnt',
	'say',
	'ooh',
	'want'])
vectorizer = TfidfVectorizer(stop_words = stopwords, max_df = 0.5, min_df = 2, sublinear_tf = True)
tfidf_mat = vectorizer.fit_transform(responses)

#tsne - we can afford pretty conservative hyperparamters for learning_rate and early_exaggeration, our data is pretty small
tsne = TSNE(n_components = 2, 
  perplexity = 10.0, 
  early_exaggeration = 2.0,
  learning_rate = 50,
  init = 'pca', 
  method = 'exact',
  random_state = 1) 
dim_red_mat = tsne.fit_transform(tfidf_mat.toarray())

#outputting to csv as (hero name, x coordinate, y coordinate)
hero_names = [line.split(' ')[0] for line in responses]
output = pd.DataFrame(data = dim_red_mat, index = hero_names, columns = ['x', 'y'])
output.to_csv('tsne_coordinates.csv', index_label = 'hero')

# dump to json
x, y = dim_red_mat[:, 0], dim_red_mat[:,1]
data_dict = dict()
for i, name in enumerate(hero_names):
  data_dict[name] = np.array([x[i], y[i]])

# apply jitter
data_dict = jitter(data_dict)

# turn numpyary into a normal list
for i, name in enumerate(hero_names):
  data_dict[name] = list(data_dict[name])

data_json = open("data_vis.js", "w")
data_json.write("data_vis = "+json.dumps(data_dict)+"\n")
