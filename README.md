# document-classification
Repo for document classification
--------------------------------------------------------------------------------------------
Document classification is one of the important task in supervised machine learning (ML). To
perform document classification automatically, documents need to be represented such that it
is understandable to the machine learning classifier. We made 3 models of classifiers, the first
to classify arabic documents by category, the second for english documents and the third to
classify documents by country. To do this we used a dataset extracted from Moroccan news
websites classified into 5 categories (culture, various, business, politics, sport) and composed
of 30 000 articles for the first model, and another which consists of 2225 documents from the
BBC news website corresponding to articles published into 5 news domains (business,
entertainment, politics, sport, technology) between 2004 and 2005 and for the last model we
used a dataset composed of 35,000 articles classified into 7 countries. Then we realized the
extraction of features using " Count and TfIdf feature vectors ". For each feature vector
representation, we trained the Naïve Bayes classifier and then tested the generated classifier
on test documents. In our results, we found that Count vectorizer performed 2% better than
TFIDF on a dataset of over 10,000 samples, and 1% better than it on a dataset of over 30,000 samples
