print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from fetch_dataset import fetch_train_data
from fetch_dataset import fetch_test_data

# Build a classification task using 3 informative features
train_data = fetch_train_data()

vectorizer = CountVectorizer(max_df=1.0, ngram_range=(1,2), max_features=5000)
#1000 - 750
#2000 - 1500 (quase mesmo score)
# vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 9)


X = vectorizer.fit_transform(train_data.data)

y = train_data.target

# Create the RFE object and compute a cross-validated score.
# clf = SGDClassifier(alpha=0.00001, penalty='elasticnet')
clf = MultinomialNB(alpha=.01)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),
              scoring='accuracy', n_jobs=-1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
