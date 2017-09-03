from fetch_dataset import fetch_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import time



start = time.time()

train_data = fetch_data(["cornell","texas","wisconsin","misc"])

test_data = fetch_data(["washington"])
vectorizer = TfidfVectorizer()

train_vectors = vectorizer.fit_transform(train_data.data)

test_vectors = vectorizer.transform(test_data.data)

clf = MultinomialNB(alpha=.1)

clf.fit(train_vectors, train_data.target)

pred = clf.predict(test_vectors)


print("f1 score:")
f1_score = metrics.f1_score(test_data.target, pred, average='macro')
print(f1_score)

print("confusion matrix:")
confusion_matrix = metrics.confusion_matrix(test_data.target, pred)
print(confusion_matrix)

print("report:")
report = metrics.classification_report(test_data.target, pred)
print(report)

feature_names = np.asarray(vectorizer.get_feature_names())

print("top 10 keywords per class:")
for i, label in enumerate(train_data.target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))


print("Tempo passado:")

print(time.time() - start)



