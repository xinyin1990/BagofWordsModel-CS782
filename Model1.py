import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup     
import pandas as pd
import numpy as np
import re
import nltk

#Read the train data
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
	delimiter="\t", quoting=3)
#print(train.shape)
#print(train.columns.values)

#Read the test data
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header = 0, delimiter="\t", \
quoting = 3)

#Verify that there are 25000 rows and 2 columns
#print(test.shape)

#Initialize the BeautifulSoup object on a singel movie review

def review_to_words(raw_review):
	##Function to convert a raw review to a string of words 
	#1. Remove HTML
	review_text = BeautifulSoup(raw_review,"html.parser")
	#

	#2. Remove non-letters

	#Use regular expressions to do a find-and-replace (letters only)
	letters_only = re.sub("[^a-zA-Z]", " ", review_text.get_text())
	#3. Convert to lower case, Split into individual words 
	lower_case = letters_only.lower()

	#Split into words
	words = lower_case.split()
	
	#4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set 
	#Download text data set, including stop words
	#nltk.download()
	from nltk.corpus import stopwords
	stops = set(stopwords.words("english"))
	#print(stopwords.words("english"))

	#5.Remove stop words from "words"
	words = [w for w in words if not w in stops]

	#6.Join the words back into one string seperated by space, and return the result.
	return(" ".join(words))

#Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size 
num_test_reviews = test["review"].size 

#Initialize an empty list to hold the clean reviews 
clean_train_reviews = []
clean_test_reviews = []

#Loop over each review; create an index i that goes from 0 to the length of the movie review list
print ("Cleaning and parsing the training set movie reviews...\n")

clean_train_reviews = []
for i in range(0, num_reviews):
	#If the index is evenly divisible by 1000, print a message
	if((i+1)%1000 == 0):
		print ("Review %d of %d\n" % (i+1, num_reviews))
	clean_train_reviews.append(review_to_words(train["review"][i]))	

for i in range(0, num_test_reviews):
	if((i+1)%1000 == 0):
		print ("Review %d of %d\n" % (i+1, num_test_reviews))
	clean_test_reviews.append(review_to_words(test["review"][i]))

clean_all_reviews =[]
clean_all_reviews = clean_train_reviews + clean_test_reviews
print ("Creating the bag of words... \n")

#Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
#
vectorizer = CountVectorizer(analyzer = "word",   \
                            tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 

#pipeline = Pipeline([('vectorizer', CountVectorizer(analyzer = "word", \
#							tokenizer = None, \
#							max_features = 20000, \
#							stop_words = None, \
#							preprocessor = None )), \
#					('tfidf', TfidfTransformer(norm = 'l2', \
#						use_idf = True, \
#						sublinear_tf = False)), \
#					])

print("vectorizing...")
'''tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')		'''		
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word', token_pattern = u'(?u)\\b\\w+\\b',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
#fit_transform()does two functions: First, it fits the model and learns the vocabulary;
#second, it transform our training data into feature vectors. The input to fit_transform should 
#be a list of strings.
#train_data_features = vectorizer.fit_transform(clean_train_reviews)
#train_data_features = train_data_features.toarray()

t = tfv.fit(clean_all_reviews)
print(t.get_feature_names())
all_data_features = tfv.transform(clean_all_reviews)

#all_data_features = vectorizer.fit_transform(clean_all_reviews)
#all_data_features = all_data_features.toarray()

#print(train_data_features)
print(all_data_features.shape)
lentrain = len(clean_train_reviews)

X = all_data_features[:lentrain]
X_test = all_data_features[lentrain:]
y = train["sentiment"]

#Take a look at the words in the vocabulary
#train_data_features = vectorizer.fit_transform(clean_train_reviews)
#vocab = vectorizer.get_feature_names()

#test_data_features = vectorizer.fit_transform(clean_test_reviews)
#vocab = vectorizer.get_feature_names()

'''
#Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis = 0)
#For each, print the vocabulary word and the number of times it appears in the training set
for tag, count in zip(vocab, dist):
	print (count, tag)
print ("Training the random forest...")
'''

#Initialize a Random Forest classifier with 100 trees
#forest = RandomForestClassifier(n_estimators = 100)
gnb = MultinomialNB()
logistic = LogisticRegression(C = 1e5)

#print ("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(forest, X, y, cv=20, scoring='roc_auc')))
print ("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(gnb, X, y, cv=20, scoring='roc_auc')))
print ("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(logistic, X, y, cv=20, scoring='roc_auc')))


#Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable 
#
#This may take a few minutes to run
print("Train models ...\n")
#forest = forest.fit(X, y)
gnb = gnb.fit(X, y)
logistic = logistic.fit(X, y)

#Get a bag of words for the test set, and convert to a numpy array
print("Predict on test data...\n")
#test_data_features = vectorizer.transform(clean_test_review)
#test_date_features = test_data_features.toarray()

#Use the random forest to make sentiment label predictions
#result1 = forest.predict(X_test)
result2 = gnb.predict_proba(X_test)[:,1]
result3 = logistic.predict_proba(X_test)[:,1]

#Copy the results to a pandas dataframe with an "id " column and a "sentiment" column
#output1 = pd.DataFrame(data = {"id":test["id"], "sentiment":result1})
output2 = pd.DataFrame(data = {"id":test["id"], "sentiment":result2})
output3 = pd.DataFrame(data = {"id":test["id"], "sentiment":result3})

#Use pandas to write the comma-separated output file
output3.to_csv(os.path.join(os.path.dirname(__file__), 'Bag_of_Words_model.csv'), index = False, quoting = 3)