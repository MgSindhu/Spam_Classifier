import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import pickle


df = pd.read_csv("spam.csv", encoding = "latin-1")

df["label"] = df["class"].map({"ham" : 0, "spam" : 1})
x = df["message"]
y = df["label"]

cv = CountVectorizer()
x = cv.fit_transform(x)

pickle.dump(cv, open("transform.pkl", "wb"))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
joblib.dump(clf, 'NB_spam_model.pkl')
NB_spam_model = open('NB_spam_model.pkl','rb')
clf = joblib.load(NB_spam_model)


