# standard libraries
import argparse
import os
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.tokenize import word_tokenize

# custom libraries
import config_set
import model_dispatcher
import clean_text


def train_model(n_folds, model, save_model='n'):

	# read the training
	df = pd.read_csv(config_set.INPUT_FILE_PATH, converters={config_set.labels: pd.eval})

	# initialize vectorizer
	tfidf = TfidfVectorizer(
		analyzer='word', 
		max_features=5000, 
		ngram_range=(1, 2), 
		stop_words='english', 
		token_pattern='(?ui)[a-z]+[a-z]+',
		max_df=0.9
	) 

	# initiate the kfolds method
	kf = model_selection.KFold(n_splits=n_folds)

	X_train = df[config_set.features]
	y_data = df[config_set.labels]

	# convert labels to 0's and 1's
	multilabel = MultiLabelBinarizer()
	y = multilabel.fit_transform(y_data)
	y_train = pd.DataFrame(y, columns=multilabel.classes_)

	for train_index, test_index in kf.split(X_train, y_train):
			
		X_train_folds = X_train.iloc[train_index]
		y_train_folds = y_train.iloc[train_index, :]
		X_test_fold = X_train.iloc[test_index]
		y_test_fold = y_train.iloc[test_index, :]

		# transform training and validation data
		X_train_folds_trans = tfidf.fit_transform(X_train_folds)
		X_test_fold_trans = tfidf.transform(X_test_fold)

		# print(tfidf.get_feature_names())

		# initialize model
		clf = model_dispatcher.models[model]
		
		# fit the model on training data
		clf.fit(X_train_folds_trans, y_train_folds)
		
		# make predictions on test data
		preds = clf.predict(X_test_fold_trans)
		
		# calculate metrics
		print(classification_report(y_test_fold, preds))	
		score = f1_score(y_test_fold, preds, average='macro')
		print("f1_score:", score)

	# save the model
	if save_model=='y':

		# transform thseconfe dataset and train on all the data
		X_train_trans = tfidf.fit_transform(X_train)

		# fit the model on all the data
		clf.fit(X_train_folds_trans, y_train_folds)

		joblib.dump(
			clf,
			config_set.MODEL_OUTPUT_PATH
		)

		joblib.dump(
			tfidf,
			config_set.VECTORIZER_PATH
		)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--n_folds",
		type=int
	)
	
	parser.add_argument(
		"--model",
		type=str
	)
	
	parser.add_argument(
		"--save",
		type=str
	)

	args = parser.parse_args()

	train_model(
		n_folds=args.n_folds,
		model=args.model,
		save_model=args.save
	)