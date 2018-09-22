from os.path import isfile, join
from os import listdir
from os import walk
from csv import writer
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
#from sklearn.metrics import classification_report
#from sklearn.decomposition import NMF
from bs4 import BeautifulSoup
from nltk import word_tokenize 
from nltk.stem.porter import PorterStemmer
import string

_DEBUG_LEVEL = 1

#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#http://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/
#http://sdsawtelle.github.io/blog/output/spam-classification-part2-vectorization-and-svm-pipeline.html

"""
   
	[Executed at: Mon Jul 17 12:52:13 PDT 2017]

	unigram test evaluation score : [25/25]
	unigramtfidf test evaluation score : [25/25]
	bigram test evaluation score : [25/25]
	bigramtfidf test evaluation score : [23/25]
	"Natural Language Processing",98.0 

	[Executed at: Sun Jul 16 9:38:42 PDT 2017]

	unigram test evaluation score : [24/25]
	unigramtfidf test evaluation score : [25/25]
	bigram test evaluation score : [3/25]
	bigramtfidf test evaluation score : [0/25]
	"Natural Language Processing",52.0 
	
	[Executed at: Sun Jul 16 11:49:00 PDT 2017]

	unigram test evaluation score : [25/25]
	unigramtfidf test evaluation score : [25/25]
	bigram test evaluation score : [0/25]
	bigramtfidf test evaluation score : [0/25]
	"Natural Language Processing",50.0 
"""	

if _DEBUG_LEVEL == 0:
	train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
	test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation
else:
	train_path = "aclImdb/train/" # use terminal to ls files under this directory
	test_path = "imdb_te.csv" # test data for grade evaluation


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
	'''Implement this module to extract
	and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
	
	"""
	load the stop words
	"""
	f = open(outpath + "stopwords.en.txt", 'r')
	stop_words = set(f.readlines())
	f.close()
	stop_words_list = [x.strip() for x in stop_words] # remove whitespace characters like `\n` at the end of each line 

	"""
	create the csvs
	"""
	row_number = 0
	imdb_tr_name = outpath + name
	if _DEBUG_LEVEL == 0:
		fd1 = open(imdb_tr_name, 'w') # imdb_tr.csv
		imdb_tr = writer(fd1)
		imdb_tr.writerow(["", "text", "polarity"])
		
		#----------------------------pos files
		pos_path = inpath + train_path + "pos/"
		pos = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]
		pos_files = []
		for (dirpath, dirnames, filenames) in walk(pos_path):
			pos_files.extend(filenames)
			break
		for t in pos_files:
			f = open(pos_path + t, 'r')
			#line = ' '.join(filter(lambda x: x.lower() not in stop_words_list,  f.readline().lower().split()))
			line = ' '.join(filter(lambda x: x.lower(),  f.readline().lower().split()))
			line = line.strip()
			imdb_tr.writerow([row_number, line, 1])
			row_number += 1
			f.close()
	
		#----------------------------neg files
		neg_path = inpath + train_path + "neg/"
		onlyfiles = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]
		neg_files = []
		for (dirpath, dirnames, filenames) in walk(neg_path):
			neg_files.extend(filenames)
			break
	
		for t in neg_files:
			f = open(neg_path + t, 'r')
			line = ' '.join(filter(lambda x: x.lower(),  f.readline().lower().split()))
			line = line.strip()
			imdb_tr.writerow([row_number, line, 0])
			row_number += 1
			f.close()
		
		fd1.close()
	
	train_dataset = pd.read_csv(imdb_tr_name, index_col='UID', names=['UID', 'text', 'polarity'], encoding = "ISO-8859-1", skiprows=1) 
	test_dataset = pd.read_csv(inpath + test_path, names=['UID', 'text'], encoding = "ISO-8859-1", skiprows=1)

	if _DEBUG_LEVEL == 1:
		print "train_dataset:"
		print train_dataset
		print "test_dataset"
		print test_dataset
	"""    
	I would recommend using fit() on training data to obtain the vector which would correspond 
	to train data set vocabulary. Then transform the vector using transform(), first with the 
	training data (to obtain X_train) and then with the test data (to obtain X_test). 
	This way you wouldn't need to use partial fit (partial fit does only 1 iteration, 
	hence performance may not be good).
	"""

	return train_dataset, test_dataset, stop_words_list

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenize(text):
	text = "".join([ch for ch in text if ch not in string.punctuation])
	tokens = word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

def Tokenize(text):
	text1 = BeautifulSoup(text, "html.parser")
	tokens = word_tokenize(text1.get_text())
	tokens = [i for i in tokens if i not in string.punctuation]
	stems = stem_tokens(tokens, stemmer)
	return stems

def SGD_unigram(train_dataset, test_dataset, stop_words_list, outpath="./", name='unigram.output.txt'):
	if _DEBUG_LEVEL == 1:
		print "SGD_unigram"
		print "==========="
	bow_transformer = CountVectorizer(tokenizer=Tokenize, stop_words=stop_words_list, ngram_range=(1, 1)).fit(train_dataset['text'])
	X_train = bow_transformer.transform(train_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Sparse Matrix: ', X_train.shape)
		#print "unigram feature names:"
		#unigrams = bow_transformer.get_feature_names()
		#print unigrams[-10:]
	X_test = bow_transformer.transform(test_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Test Matrix: ', X_test.shape)

	tf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
	X_train_tf = tf_transformer.transform(X_train)
	X_test_tf = tf_transformer.transform(X_test)

	clf = SGDClassifier(loss="hinge", penalty="l1") #, alpha=1e-3)
	clf.fit(X_train_tf, train_dataset['polarity'])
	predicted = clf.predict(X_test_tf)
	#if _DEBUG_LEVEL == 1:
	#	print (classification_report(test_dataset, predicted))

	fd = open(outpath + name, 'w') # unigram.output.txt
	unigram = writer(fd)
	for x in enumerate(predicted):
		unigram.writerow([x[1]])
	fd.close()

	return

def SGD_bigram(train_dataset, test_dataset, stop_words_list, outpath="./", name='bigram.output.txt'):
	if _DEBUG_LEVEL == 1:
		print "SGD_bigram"
		print "==========="
	bow_transformer = CountVectorizer(tokenizer=Tokenize, stop_words=stop_words_list, ngram_range=(1, 2)).fit(train_dataset['text'])
	X_train = bow_transformer.transform(train_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Sparse Matrix: ', X_train.shape)
		#print "unigram feature names:"
		#unigrams = bow_transformer.get_feature_names()
		#print unigrams[-10:]
	X_test = bow_transformer.transform(test_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Test Matrix: ', X_test.shape)

	tf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
	X_train_tf = tf_transformer.transform(X_train)
	X_test_tf = tf_transformer.transform(X_test)

	clf = SGDClassifier(loss="hinge", penalty="l1") #, alpha=1e-3)
	clf.fit(X_train_tf, train_dataset['polarity'])
	predicted = clf.predict(X_test_tf)

	fd = open(outpath + name, 'w') # bigram.output.txt
	bigram = writer(fd)
	for x in enumerate(predicted):
		bigram.writerow([x[1]])
	fd.close()

	return

def SGD_unigram_TfIdf(train_dataset, test_dataset, stop_words_list, outpath="./", name='unigramtfidf.output.txt'):
	if _DEBUG_LEVEL == 1:
		print "SGD_unigram_TfIdf"
		print "================="
	#bow_transformer = TfidfVectorizer(tokenizer=Tokenize, stop_words=stop_words_list, ngram_range=(1, 1)).fit(train_dataset['text'])
	bow_transformer = CountVectorizer(tokenizer=Tokenize, stop_words=stop_words_list, ngram_range=(1, 1)).fit(train_dataset['text'])
	X_train = bow_transformer.transform(train_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Sparse Matrix: ', X_train.shape)
	X_test = bow_transformer.transform(test_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Test Matrix: ', X_test.shape)

	tf_transformer = TfidfTransformer(use_idf=True).fit(X_train)
	X_train_tf = tf_transformer.transform(X_train)
	X_test_tf = tf_transformer.transform(X_test)

	clf = SGDClassifier(loss="hinge", penalty="l1") #, alpha=1e-3)
	clf.fit(X_train_tf, train_dataset['polarity'])
	predicted = clf.predict(X_test_tf)
	#if _DEBUG_LEVEL == 1:
	#	print (classification_report(test_dataset.values, predicted))

	fd = open(outpath + name, 'w') # unigram.output.txt
	unigram = writer(fd)
	for x in enumerate(predicted):
		unigram.writerow([x[1]])
	fd.close()

	return

def SGD_bigram_TfIdf(train_dataset, test_dataset, stop_words_list, outpath="./", name='bigramtfidf.output.txt'):
	if _DEBUG_LEVEL == 1:
		print "SGD_bigram_TfIdf"
		print "================"
	bow_transformer = TfidfVectorizer(tokenizer=Tokenize, stop_words=stop_words_list, ngram_range=(1, 2), sublinear_tf=True).fit(train_dataset['text'])
	X_train = bow_transformer.transform(train_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Sparse Matrix: ', X_train.shape)
	X_test = bow_transformer.transform(test_dataset['text'])
	if _DEBUG_LEVEL == 1:
		print ('Shape of Test Matrix: ', X_test.shape)
	
	#n_topics=2
	#nmf = NMF(n_components=n_topics, random_state=101).fit(X_train)

	tf_transformer = TfidfTransformer(use_idf=True).fit(X_train)
	X_train_tf = tf_transformer.transform(X_train)
	X_test_tf = tf_transformer.transform(X_test)

	clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=100, verbose=0) #, alpha=1e-3)
	clf.fit(X_train_tf, train_dataset['polarity'])
	predicted = clf.predict(X_test_tf)

	fd = open(outpath + name, 'w') # bigram.output.txt
	bigram = writer(fd)
	for x in enumerate(predicted):
		bigram.writerow([x[1]])
	fd.close()

	return


if __name__ == "__main__":
	initialTime = time.clock()

	train_dataset, test_dataset, stop_words_list = imdb_data_preprocess("./")

	'''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
	if _DEBUG_LEVEL == 0:
		SGD_unigram(train_dataset, test_dataset, stop_words_list)
	
	'''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
	if _DEBUG_LEVEL == 0:
	   SGD_bigram(train_dataset, test_dataset, stop_words_list)

	'''train a SGD classifier using unigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigramtfidf.output.txt'''
	if _DEBUG_LEVEL == 0:
		SGD_unigram_TfIdf(train_dataset, test_dataset, stop_words_list)
	
	'''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to bigramtfidf.output.txt'''
	SGD_bigram_TfIdf(train_dataset, test_dataset, stop_words_list)

	finalTime = time.clock()
	if _DEBUG_LEVEL == 1:
		print ('time', finalTime - initialTime)
	
	pass
