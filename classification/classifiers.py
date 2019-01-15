# Author: Rizart Dona
# File: classifiers.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD
from gensim.models.word2vec import Word2Vec
from nltk.stem.porter import PorterStemmer
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
from scipy import interp
from sklearn import svm
import numpy as np
import warnings
import argparse
import pandas
import nltk
import time
import csv
import sys


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Classifier template


def run_classifier(X, y, clf, clfname, feature):

    sys.stdout.write("running 10-fold Cross Validation (%s) .. " % clfname)
    sys.stdout.flush()

    # start timer
    stime = time.time()

    # setup metrics
    metrics_map = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F-Measure': [],
        'AUC': []
    }

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_tpr = dict()
    mean_fpr = dict()

    kf = model_selection.StratifiedKFold(n_splits=10)
    for train, test in kf.split(X, y):
        # setup data
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # fit train set
        clf.fit(X_train, y_train)

        # predict test set
        y_pred = clf.predict(X_test)

        # aggregate metrics
        metrics_map['Accuracy'].append(
            metrics.accuracy_score(y_test, y_pred))
        metrics_map['Precision'].append(
            metrics.precision_score(y_test, y_pred, average='weighted'))
        metrics_map['Recall'].append(
            metrics.recall_score(y_test, y_pred, average='weighted'))
        metrics_map['F-Measure'].append(
            metrics.f1_score(y_test, y_pred, average='weighted'))

        # AUC
        ytb = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
        ypb = label_binarize(y_pred, classes=[0, 1, 2, 3, 4])
        for i in range(5):
            fpr[i], tpr[i], _ = metrics.roc_curve(ytb[:, i], ypb[:, i])

            if i not in mean_tpr:
                mean_tpr[i] = 0.0
                mean_fpr[i] = np.linspace(0, 1, 100)

            mean_tpr[i] += interp(mean_fpr[i], fpr[i], tpr[i])
            mean_tpr[i][0] = 0.0

    # Compute macro-average ROC curve and ROC area

    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([mean_fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    meanm_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        meanm_tpr += interp(all_fpr, mean_fpr[i], mean_tpr[i])

    # macro-average AUC
    meanm_tpr /= 50
    fpr["macro"] = all_fpr
    tpr["macro"] = meanm_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # ROC plot
    plt.figure()
    lw = 2
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'darkorange'])

    # plot macro-average
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # plot acerages for classes
    for i, color in zip(range(5), colors):
        mean_tpr[i] /= 10
        mean_tpr[i][-1] = 1.0
        mean_auc = metrics.auc(mean_fpr[i], mean_tpr[i])
        metrics_map['AUC'].append(mean_auc)
        plt.plot(mean_fpr[i], mean_tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, mean_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot for %s (%s)' % (clfname, feature))
    plt.legend(loc="lower right")
    plt.savefig('roc_plots/%s_%s_roc_10fold.png' % (clfname, feature))

    # produce averages
    for m in metrics_map:
        metrics_map[m] = '%.2f' % float(
            reduce(lambda x, y: x + y, metrics_map[m]) / len(metrics_map[m]))

    # end time
    etime = time.time()
    seconds = int(etime - stime)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("(%02d:%02d:%02d)\n" % (h, m, s))

    return metrics_map

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Features


def bow(data):

    # vectorize document content
    vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
    sys.stdout.write("vectorizing documents (BoW) .. ")
    sys.stdout.flush()
    stime = time.time()
    X = vectorizer.fit_transform(data['Content'])
    etime = time.time()
    seconds = int(etime - stime)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("(%02d:%02d:%02d)\n" % (h, m, s))
    return X


def svd(data, Xbow):

    # vectorize document content
    sys.stdout.write("vectorizing documents (SVD) .. ")
    sys.stdout.flush()
    stime = time.time()
    svd = TruncatedSVD(n_components=100, algorithm='arpack')
    X = svd.fit_transform(Xbow)
    etime = time.time()
    seconds = int(etime - stime)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("(%02d:%02d:%02d)" % (h, m, s))
    explained_variance = svd.explained_variance_ratio_.sum()
    print ' [SVD variance: %.2f%%]' % (explained_variance * 100)
    return X


def w2v(data):

    sys.stdout.write("vectorizing documents (W2V) .. ")
    sys.stdout.flush()
    stime = time.time()
    dim = 200
    sentences = [row['Content'].split() for _, row in data.iterrows()]
    model = Word2Vec(sentences, workers=4, size=dim, min_count=10)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    X = np.array([np.mean([w2v[w] for w in words if w in w2v] or
                          [np.zeros(dim)], axis=0) for words in sentences])
    etime = time.time()
    seconds = int(etime - stime)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("(%02d:%02d:%02d)\n" % (h, m, s))
    return X


def my_features(data):

    # you must run this in a python console before running stemmer
    # $> nltk.download('punkt')
    stemmer = PorterStemmer()

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(stemmer.stem(item))
        return stems

    sys.stdout.write("vectorizing documents (stemming+BoW) .. ")
    sys.stdout.flush()
    stime = time.time()
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english',
                            max_features=5000)
    X = tfidf.fit_transform(data['Content'])
    etime = time.time()
    seconds = int(etime - stime)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("(%02d:%02d:%02d)\n" % (h, m, s))
    return X, tfidf


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Grid search results

svm_map = {
    'BoW': svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape=None, degree=3, gamma=0.001,
                   kernel='rbf', max_iter=-1, probability=False,
                   random_state=None, shrinking=True, tol=0.001),
    'SVD': svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape=None, degree=3, gamma='auto',
                   kernel='linear', max_iter=-1, probability=False,
                   random_state=None, shrinking=True, tol=0.001),
    'W2V': svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape=None, degree=3, gamma=0.001,
                   kernel='rbf', max_iter=-1, probability=False,
                   random_state=None, shrinking=True, tol=0.001)
}

rf_map = {
    'BoW': RandomForestClassifier(bootstrap=False, class_weight='balanced',
                                  criterion='gini', max_depth=None,
                                  max_features='log2', max_leaf_nodes=None,
                                  min_impurity_split=1e-07, min_samples_leaf=1,
                                  min_samples_split=2,
                                  min_weight_fraction_leaf=0.0,
                                  n_estimators=30, n_jobs=-1, oob_score=False,
                                  random_state=None, warm_start=False),
    'SVD': RandomForestClassifier(bootstrap=False, class_weight='balanced',
                                  criterion='entropy', max_depth=None,
                                  max_features='auto', max_leaf_nodes=None,
                                  min_impurity_split=1e-07, min_samples_leaf=1,
                                  min_samples_split=2,
                                  min_weight_fraction_leaf=0.0,
                                  n_estimators=30, n_jobs=-1, oob_score=False,
                                  random_state=None, warm_start=False),
    'W2V': RandomForestClassifier(bootstrap=False,
                                  class_weight='balanced_subsample',
                                  criterion='entropy', max_depth=None,
                                  max_features='auto', max_leaf_nodes=None,
                                  min_impurity_split=1e-07, min_samples_leaf=1,
                                  min_samples_split=2,
                                  min_weight_fraction_leaf=0.0,
                                  n_estimators=30, n_jobs=-1, oob_score=False,
                                  random_state=None, warm_start=False)
}
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# My Method


def extract_predictions(X, y, input_test, le, tfidf):

    clf = svm.LinearSVC(C=1, class_weight=None, dual=False,
                        fit_intercept=False, intercept_scaling=1,
                        loss='squared_hinge', max_iter=1000, multi_class='ovr',
                        penalty='l2', random_state=None, tol=0.0001)

    clf.fit(X, y)

    tdata = pandas.read_csv(input_test, sep="\t")
    X_test = tfidf.transform(tdata['Content'])

    y_pred = clf.predict(X_test)
    categs = le.inverse_transform(y_pred)

    # write to csv
    ids = []
    for index, row in tdata.iterrows():
        ids.append(row['Id'])

    with open('testSet_categories.csv', 'w') as csvfile:
        fieldnames = ['Test_Document_ID', 'Predicted_Category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for i in range(0, len(y_pred.flat)):
            row = {}
            row['Test_Document_ID'] = ids[i]
            row['Predicted_Category'] = le.classes_[y_pred.flat[i]]
            writer.writerow(row)


def my_method(data, y, input_test, le):

    X, tfidf = my_features(data)

    clf = svm.LinearSVC(C=1, class_weight=None, dual=False,
                        fit_intercept=False, intercept_scaling=1,
                        loss='squared_hinge', max_iter=1000, multi_class='ovr',
                        penalty='l2', random_state=None, tol=0.0001)

    metrics = run_classifier(X, y, clf, 'LinearSVC', "MyFeatures")

    # extract predictions
    sys.stdout.write("---------------------------------------------\n")
    sys.stdout.write("extracting predictions ..\n")
    extract_predictions(X, y, input_test, le, tfidf)
    return metrics


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def run():

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run classification algorithms")
    parser.add_argument("-train",
                        required=True,
                        dest="input_train",
                        help="Input csv training dataset")
    parser.add_argument("-test",
                        required=True,
                        dest="input_test",
                        help="Input csv training dataset")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(feature=False)
    arg = parser.parse_args()

    # read data
    data = pandas.read_csv(arg.input_train, sep="\t")

    # test case
    if arg.test:
        n_values = 1000
        data = data[0:n_values]

    results = []
    with warnings.catch_warnings():
        # suppress warnings
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UndefinedMetricWarning)

        # extract features
        X_bow = bow(data)
        X_svd = svd(data, X_bow)
        X_w2v = w2v(data)
        features = [('BoW', X_bow), ('SVD', X_svd), ('W2V', X_w2v)]

        # encode categories
        le = preprocessing.LabelEncoder()
        le.fit(data["Category"])
        y = le.transform(data["Category"])

        # run classification
        for f, X in features:

            sys.stdout.write("---------------------------------------------\n")
            sys.stdout.write('Feature: %s\n' % f)
            sys.stdout.flush()

            # Support Vector Machines
            clf = svm_map[f]
            results.append(
                ['SVM (%s)' % f, run_classifier(X, y, clf, "SVM", f)])

            # Random Forests
            clf = rf_map[f]
            results.append(
                ['Random Forest (%s)' % f, run_classifier(
                    X, y, clf, "Random Forests", f)])

        # my method
        sys.stdout.write("---------------------------------------------\n")
        sys.stdout.write('Feature: stemming+BoW\n')
        results.append(['My Method', my_method(data, y, arg.input_test, le)])

    # write to csv file
    output_csv_file = 'EvaluationMetric_10fold.csv'
    with open(output_csv_file, 'w') as csvfile:
        fieldnames = ['Statistic Measure', 'SVM (BoW)', 'Random Forest (BoW)',
                      'SVM (SVD)', 'Random Forest (SVD)', 'SVM (W2V)',
                      'Random Forest (W2V)', 'My Method']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        row = {}
        for st in ['Accuracy', 'Precision', 'Recall', 'F-Measure', 'AUC']:
            for t in results:
                clf = t[0]
                metr_map = t[1]
                row['Statistic Measure'] = st
                row[clf] = metr_map[st]
            writer.writerow(row)


# -----------------------------------------------------------------------------
run()
