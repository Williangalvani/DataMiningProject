
import numpy as np
from time import time

from fetch_dataset import fetch_train_data
from fetch_dataset import fetch_test_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

# TfidfVectorizer
# SGDClassifier          accuracy:   0.827  penalty='l1'
# SGDClassifier          accuracy:   0.827  penalty='elasticnet'
# LinearSVC              accuracy:   0.825  penalty='l1'

# HashingVectorizer
# SGDClassifier          accuracy:   0.810 penalty='elasticnet'
# SGDClassifier          accuracy:   0.809 penalty='l2'
# SGDClassifier          accuracy:   0.805 penalty='l1'

# CountVectorizer
# LinearSVC              accuracy:   0.771 penalty='l2' 
# RandomForestClassifier accuracy:   0.766 n_estimators=100
# LinearSVC              accuracy:   0.751 penalty='l1'

# CountVectorizer 
# ngram=(1,2) chi2_select=50000
# LinearSVC              accuracy:   0.796 penalty='l2'
# RandomForestClassifier accuracy:   0.793 n_estimators=100
# LinearSVC              accuracy:   0.779 penalty='l1'


data_train = fetch_train_data()
data_test = fetch_test_data()
pred_list = []

y_train, y_test = data_train.target, data_test.target

target_names = data_train.target_names

t0 = time()
vectorizer_tfid = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

X_train_tfid = vectorizer_tfid.fit_transform(data_train.data)
X_test_tfid = vectorizer_tfid.transform(data_test.data)


clf1_tfid = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
       n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf1_tfid.fit(X_train_tfid, y_train)
pred_list.append(clf1_tfid.predict(X_test_tfid))


clf2_tfid = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf2_tfid.fit(X_train_tfid, y_train)
pred_list.append(clf2_tfid.predict(X_test_tfid))


clf3_tfid = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
     verbose=0)
clf3_tfid.fit(X_train_tfid, y_train)
pred_list.append(clf3_tfid.predict(X_test_tfid))

duration = time() - t0

print("TfidfVectorizer and classifiers %fs " % duration)


t0 = time()
vectorizer_hash  = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=50000)

X_train_hash = vectorizer_hash.fit_transform(data_train.data)
X_test_hash = vectorizer_hash.transform(data_test.data)

clf1_hash = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
       n_jobs=1, penalty='elasticnet', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf1_hash.fit(X_train_hash, y_train)
pred_list.append(clf1_hash.predict(X_test_hash))

clf2_hash = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
     verbose=0)
clf2_hash.fit(X_train_hash, y_train)
pred_list.append(clf2_hash.predict(X_test_hash))

clf3_hash = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf3_hash.fit(X_train_hash, y_train)
pred_list.append(clf3_hash.predict(X_test_hash))

duration = time() - t0
print("HashingVectorizer and classifiers %fs " % duration)



t0 = time()
vectorizer_count = CountVectorizer(max_df=1.0, ngram_range=(1,2))

X_train_count = vectorizer_count.fit_transform(data_train.data)
X_test_count = vectorizer_count.transform(data_test.data)


ch2 = SelectKBest(chi2, k=50000)
X_train_count = ch2.fit_transform(X_train_count, y_train)
X_test_count = ch2.transform(X_test_count)

clf1_count = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
     verbose=0)
clf1_count.fit(X_train_count, y_train)
pred_list.append(clf1_count.predict(X_test_count))

clf2_count = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf2_count.fit(X_train_count, y_train)
pred_list.append(clf2_count.predict(X_test_count))

clf3_count = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
       n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf3_count.fit(X_train_count, y_train)
pred_list.append(clf3_count.predict(X_test_count))

duration = time() - t0
print("CountVectorizer and classifiers %fs " % duration)


#voting classifier
t0 = time()

predictions = np.asarray([pred for pred in pred_list]).T

pred =  np.apply_along_axis(lambda x: np.bincount(x, weights=None).argmax(), axis=1, arr=predictions)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred, target_names=target_names))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))


duration = time() - t0
print("Voting and metrics %fs " % duration)

