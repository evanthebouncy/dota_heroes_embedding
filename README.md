# dota_heroes_embedding
visualizing hero similarities based on their dialogue

# files

##scraper.py
evan's dota2wiki scraper, creates the data file

##data
the raw text file of scraped responses

##tfidf_tsne.py
creates the tfidf matrix from the raw text, runs a 2d tsne embedding and outputs the results to csv

##tsne_coordinates.csv
the x, y coordinates of each hero 
