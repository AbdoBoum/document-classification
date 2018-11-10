import string
import numpy as np
import pandas as pd # for csv file


from nltk.stem.porter import * # to perform stemming use to remove verbs

from nltk.corpus import * # for removing stop words

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
import pickle

import matplotlib.pyplot as plt

from src.Plot import plot_confusion_matrix

from src.textCleaning import clean_text_en

# ----------------------- Imported all the libraries ----------------------------------------------

# grabbing the data
news = pd.read_csv('..\dataset\Englishdataset.csv')
#news=news_headlines.head(n=100000)
news_list = [s.lower().translate(str.maketrans('','',string.punctuation)) for s in news['news']]

stop = set(stopwords.words('english'))
filtered_words = []
for sentence in news_list:
	filtered_words.append(clean_text_en(sentence))

stemmer = PorterStemmer()
stemmed=[]
for words in filtered_words:
	st=''
	for i in words.split(' '):
		st=st+' '+stemmer.stem(i)

	stemmed.append(st)


# ------------------------------ data processing over ---------------------------


# vector = TfidfVectorizer()
vector = CountVectorizer()
x = vector.fit_transform(stemmed)

code = LabelEncoder()
y = code.fit_transform(news['type'])  # it takes the whole coloumn and then generates the required amount of labels

class_names = list(set(" ".join(news['type']).split(" ")))


# ------------------------------------------------- feature selection ----------------------------------------------------------------

# uncomment them to implemet currently no feature reduction technique is implemented as time requires large amount of processing time

# mi = mutual_info_classif(x, y, discrete_features=True)
# x_chi = SelectKBest(chi2, k=1000).fit_transform(x, y)
# x_MI = SelectKBest(mutual_info_classif, k=10000).fit_transform(x, y)
# x_vt = VarianceThreshold().fit_transform(x) # it is set to its default value which is
# pca = decomposition.TruncatedSVD(n_components=10) # pca.fit(x) # x_pca = pca.transform(x) # depriciated implementation of pca

# svd = decomposition.TruncatedSVD(n_components=10000)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# x_pca = lsa.fit_transform(x)
# min_max_scaler = MinMaxScaler()
# x_pca = min_max_scaler.fit_transform(x_pca)


# ---------------split into train and test sets -----------------------------

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
# xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x_chi, y, test_size=0.2)
# xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x_MI, y, test_size=0.2)
# xtrain3, xtest3, ytrain3, ytest3 = train_test_split(x_vt, y, test_size=0.2)
# xtrain4, xtest4, ytrain4, ytest4 = train_test_split(x_pca, y, test_size=0.2)

# ------------------------------------------------naive bayes implementation ------------------------------------------------



featurenames=' '.join(vector.get_feature_names())
file = open("..\_features\_english_category_features.txt","w", encoding="utf8")
file.write(featurenames)
file.close()
# print(xtrain4)
nb = MultinomialNB()
nb.fit(xtrain, ytrain)	# fitting the text
print("Normal implementation")
print(nb.score(xtest, ytest))# to calculate the score for the classification
pickle.dump(nb, open("..\exported_models\_english_category_model.pickle.dat", "wb"))

# ---------------------- Plotting ------------------------------------

ypred = nb.predict(xtest)
cnf_matrix = confusion_matrix(ytest, ypred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest, ypred)
print("\nAccuracy: ", acc)