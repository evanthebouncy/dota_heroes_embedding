import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.manifold import TSNE

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

#plotting
hero_names = [line.split(' ')[0] for line in responses]
x, y = dim_red_mat[:, 0], dim_red_mat[:,1]
plt.scatter(x, y, linewidths = 1, alpha = 0)
for i, name in enumerate(hero_names):
	plt.annotate(name, (x[i], y[i]))
plt.xlabel('1st Embedding Dimension', fontsize = 16)
plt.ylabel('2nd Embedding Dimension', fontsize = 16)
plt.show()
