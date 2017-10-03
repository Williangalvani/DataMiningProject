

from fetch_dataset import fetch_train_data
from fetch_dataset import fetch_test_data

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import VotingClassifier


from sklearn import metrics


# CountVectorizer 
# ngram=(1,2) chi2_select=50000
# LinearSVC              accuracy:   0.796 penalty='l2'
# RandomForestClassifier accuracy:   0.793 n_estimators=100
# LinearSVC              accuracy:   0.779 penalty='l1'


data_train = fetch_train_data()
data_test = fetch_test_data()


y_train, y_test = data_train.target, data_test.target

target_names = data_train.target_names

vectorizer = CountVectorizer(max_df=1.0, ngram_range=(1,2))

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

feature_names = vectorizer.get_feature_names()

ch2 = SelectKBest(chi2, k=50000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
if feature_names:
 # keep selected feature names
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
if feature_names:
    feature_names = np.asarray(feature_names)


clf1 = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
     verbose=0)


clf2 = RandomForestClassifier(n_estimators=100)


# clf3 = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
#      verbose=0)

clf3 = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
       n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)

eclf1 = VotingClassifier(estimators=[('linsvcl2', clf1), ('randForest', clf2), ('linsvcl1', clf3)], voting='hard')

eclf1 = eclf1.fit(X_train, y_train)

pred = eclf1.predict(X_test)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred, target_names=target_names))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))

