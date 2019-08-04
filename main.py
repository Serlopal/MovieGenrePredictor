import argparse
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
sys.stderr = stderr
import json
from nltk import word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
import string
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def strip_punctuation(s):
	for x in string.punctuation:
		s = s.replace(x, "")
	return s


def check_valid_plot(p):
	if len(p) == 0:
		raise Exception("--plot must be a non empty string!")

	words = word_tokenize(p)

	if len(words) < min_plot_len:
		raise Exception("--description must contain at least {} words.".format(min_plot_len))

	if len(words) > max_plot_len:
		raise Exception("--description must not contain more than {} words.".format(max_plot_len))


def check_valid_title(title):
	if len(title) == 0:
		raise Exception("--title must be a non empty string!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--title", action="store", type=str, help="title of the movie provided")
	parser.add_argument("--description", action="store", type=str, help="brief summary of the movie's plot")
	args = parser.parse_args()
	title = args.title
	plot = args.description

	# ---------------
	# INPUT CHECKING
	# ---------------
	check_valid_title(title)
	check_valid_plot(plot)

	# ---------------
	# MODEL LOADING
	# ---------------
	model_path = "model/model.h5"
	tokenizer_path = "model/keras_tokenizer.json"
	# load keras model
	model = load_model(model_path)
	# load keras tokenizer
	with open(tokenizer_path) as f:
		data = json.load(f)
		tokenizer = tokenizer_from_json(data)
	# load extra parameters
	with open('model/parameters.pickle', 'rb') as h:
		param_dict = pickle.load(h)
	max_seq_length = param_dict["max_length"]
	used_genres = list(param_dict["used_genres"])

	# ---------------
	# PROGRAM PARAMETERS
	# ---------------
	min_plot_len = 4  # in number of words
	max_plot_len = max_seq_length  # in number of words

	# ---------------
	# INPUT CLEANING
	# ---------------
	# tokenize plot
	input_string = word_tokenize(plot)
	# remove stop words
	stop = list(get_stop_words('en')) + list(string.punctuation)
	# remove punctuation
	input_string = [strip_punctuation(w.lower()) for w in input_string if w not in stop and w != ""]
	# put together in a single string
	input_string = " ".join(input_string)

	# ---------------
	# GENRE PREDICTING
	# ---------------
	dense_input = pad_sequences(tokenizer.texts_to_sequences([input_string]), maxlen=max_seq_length, padding='post')
	pred = model.predict(dense_input)
	pred_genre = used_genres[np.argmax(pred)]

	# ---------------
	# OUTPUT PRINTING
	# ---------------
	# put together results dictionary
	result = {"title": title, "description": plot, "genre:": pred_genre}
	result_json = json.dumps(result, ensure_ascii=False)
	print(result_json)