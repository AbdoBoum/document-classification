import string
import numpy as np
import pandas as pd # for csv file
from tashaphyne.stemming import ArabicLightStemmer #arabic stemmer

# to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import CountVectorizer

# for feature selection
from sklearn.metrics import confusion_matrix

# implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) features.
from sklearn.preprocessing import LabelEncoder

# splitting the data in test and train
from sklearn.model_selection import train_test_split

# classifying with naive bayes
from sklearn.naive_bayes import MultinomialNB


import matplotlib.pyplot as plt

import pickle
# for saving model

from src.textCleaning import clean_text

from src.Plot import plot_confusion_matrix


# ----------------------- Imported all the libraries ----------------------------------------------

news = pd.read_csv('..\dataset\Arabicdataset.csv',error_bad_lines = False)
filtered_words = []
news_list = [s.lower().translate(str.maketrans('','',string.punctuation)) for s in news['news']]

for sentence in news_list:
    filtered_words.append(clean_text(sentence))
stemmer = ArabicLightStemmer()
stemmed=[]
for words in filtered_words:
	st=''
	for i in words.split(' '):
		st=st+' '+stemmer.light_stem(i)

	stemmed.append(st)


# ------------------------------ data processing over ---------------------------
vector = CountVectorizer()
x = vector.fit_transform(stemmed)

code = LabelEncoder()
y = code.fit_transform(news['type'])  # it takes the whole coloumn and then generates the required amount of labels

class_names = list(set(" ".join(news['type']).split(" ")))



# ---------------split into train and test sets -----------------------------

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# ------------------------------------------------naive bayes implementation ------------------------------------------------

featurenames=' '.join(vector.get_feature_names())
file = open("arabic_category_features.txt","w", encoding="utf8")
file.write(featurenames)
file.close()
nb = MultinomialNB()
nb.fit(xtrain, ytrain)	# fitting the text
print("Normal implementation")
print(nb.score(xtest, ytest))		# to calculate the score for the classification
pickle.dump(nb, open("arabic_category_model.pickle.dat", "wb"))
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