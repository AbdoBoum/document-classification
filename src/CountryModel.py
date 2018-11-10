import string
import numpy as np
import pandas as pd # for csv file

import matplotlib.pyplot as plt

from nltk.stem.porter import * # to perform stemming use to remove verbs

# to convert a collection of raw documents to a matrix of TF-IDF _features.
from sklearn.feature_extraction.text import CountVectorizer

# for feature selection
from sklearn.metrics import confusion_matrix

# implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) _features.
from sklearn.preprocessing import LabelEncoder

# splitting the data in test and train
from sklearn.model_selection import train_test_split

# classifying with naive bayes
from sklearn.naive_bayes import MultinomialNB

# for saving model

from src.textCleaning import clean_text_en

from src.Plot import plot_confusion_matrix

import pickle


# ----------------------- Imported all the libraries ----------------------------------------------

news = pd.read_csv('..\dataset\pays.csv', dtype={'contenue':str}, delimiter=';',error_bad_lines = False)
news_list = [w.lower().translate(str.maketrans('','',string.punctuation)) for w in news['contenue']]

filtered_words = []
for sentence in news_list:
	filtered_words.append(clean_text_en(sentence))
stemmed=[]
stemmer = PorterStemmer()

for words in filtered_words:
	st=''
	for i in words.split(' '):
		st=st+' '+stemmer.stem(i)

	stemmed.append(st)

# ------------------------------ data processing over ---------------------------
vector = CountVectorizer()
x = vector.fit_transform(stemmed)

code = LabelEncoder()
y = code.fit_transform(news['pays'])  # it takes the whole coloumn and then generates the required amount of labels

class_names = list(set(" ".join(news['pays']).split(" ")))


# ---------------split into train and test sets -----------------------------

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
# xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x_chi, y, test_size=0.2)
# xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x_MI, y, test_size=0.2)
# xtrain3, xtest3, ytrain3, ytest3 = train_test_split(x_vt, y, test_size=0.2)
# xtrain4, xtest4, ytrain4, ytest4 = train_test_split(x_pca, y, test_size=0.2)

# ------------------------------------------------naive bayes implementation ------------------------------------------------

featurenames=' '.join(vector.get_feature_names())
file = open("..\_features\_country_features.txt","w", encoding="utf8")
file.write(featurenames)
file.close()
# print(xtrain4)
nb = MultinomialNB()
nb.fit(xtrain, ytrain)	# fitting the text
print("Normal implementation")
print(nb.score(xtest, ytest))		# to calculate the score for the classification
pickle.dump(nb, open("..\exported_models\_country_model.pickle.dat", "wb"))
# ---------------------- Plotting ------------------------------------

ypred = nb.predict(xtest)
cnf_matrix = confusion_matrix(ytest, ypred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
plt.show()

from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest, ypred)
print("\nAccuracy: ", acc)
