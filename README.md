GitHub Repo to accompany my Bachelor Thesis, Mapping political polarization using AI based sentiment analysis.

The sentanalysis.yml file is used to create a conda environment with all the required packages already installed.

The Twitter-scraper folder contains the scraper code in scraper.py and the JSON parser in json-parser.py used to parse only the text from the received API message, with this parser you can also choose to filter out any non-Dutch tweets.

This folder also contains three csv files, one containing the human-rated gold-label, one containing the AI labeled tweets and one containing both the human and AI labeled tweets and the number of tweets with a label that differ between the human labeled and AI labeled.

The Sentiment Analysis Model folder contains the code used to train the AI-model in train_optimized.py. Along with three scripts used for testing one or multiple models on one or multiple tweets. 
