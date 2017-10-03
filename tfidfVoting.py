

from fetch_dataset import fetch_train_data
from fetch_dataset import fetch_test_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier


from sklearn import metrics


# SGDClassifier accuracy:   0.827  penalty='l1'
# SGDClassifier accuracy:   0.827  penalty='elasticnet'
# LinearSVC accuracy:       0.825  penalty='l1'


data_train = fetch_train_data()
data_test = fetch_test_data()


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

y_train, y_test = data_train.target, data_test.target

target_names = data_train.target_names

clf1 = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
       n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)

clf2 = RandomForestClassifier(n_estimators=100)

clf3 = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
     verbose=0)

eclf1 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting='hard')

eclf1 = eclf1.fit(X_train, y_train)

pred = eclf1.predict(X_test)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred, target_names=target_names))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))

