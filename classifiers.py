from sklearn import svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import manipulator as mp


def logRes(X_train, X_test, y_train, y_test, kaggle):
    clf = LogisticRegression(C=17, max_iter=100000, tol=1e-6, solver='newton-cg', multi_class='multinomial', fit_intercept=False, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mp.finalize(y_test, y_pred, kaggle)


def sgdClass(X_train, X_test, y_train, y_test, kaggle):
    clf = SGDClassifier(loss='hinge', max_iter=100000, tol=1e-7, shuffle=True, alpha=10, n_jobs=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mp.finalize(y_test, y_pred, kaggle)


def randFost(X_train, X_test, y_train, y_test, kaggle):
    clf = RandomForestClassifier(n_estimators=500, criterion="gini", bootstrap=True, min_samples_split=4, max_depth=55, max_features='sqrt', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mp.finalize(y_test, y_pred, kaggle)


def supportVM(X_train, X_test, y_train, y_test, kaggle):
    clf = svm.SVC(kernel='rbf', gamma='scale', C=17)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mp.finalize(y_test, y_pred, kaggle)






'''
_init__(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2,
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None,
 min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None,
 random_state=None, verbose=0, warm_start=False, class_weight=None)
'''


'''
LogisticRegression
__init__(penalty=’l2’, dual=False, tol=0.0001,
C=1.0, fit_intercept=True, intercept_scaling=1,
class_weight=None, random_state=None, solver=’warn’,
max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)
'''

'''
RandomForestClassifier
__init__(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
warm_start=False, class_weight=None)[source]¶
'''
'''
SVC
__init__(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0,
shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)[source]
'''

'''
SGDClassifier
SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None,
learning_rate=’optimal’, eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
n_iter_no_change=5, class_weight=None, warm_start=False, average=False, n_iter=None)[source]
'''
