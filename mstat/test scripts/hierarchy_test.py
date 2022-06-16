#!/usr/bin/env python
"""
Example of using the hierarchical classifier to classify (a subset of) the digits data set.
Demonstrated some of the capabilities, e.g using a Pipeline as the base estimator,
defining a non-trivial class hierarchy, etc.
"""
from sklearn import svm
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled
from sklearn_hierarchical_classification.tests.fixtures import make_digits_dataset

from sklearn.calibration import CalibratedClassifierCV as CalClass


# Used for seeding random state
RANDOM_STATE = 44


def classify_digits():
    r"""Test that a nontrivial hierarchy leaf classification behaves as expected.
    We build the following class hierarchy along with data from the handwritten digits dataset:
            <ROOT>
           /      \
          A        B
         / \    /  |  \
        1   7  2   C   9
                 /   \
                3     8
    """
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": ["1", "7"],
        "B": ["C", "9", "2"],
        "C": ["3", "8"],
    }
    base_estimator = make_pipeline(
        TruncatedSVD(n_components=24),
        svm.SVC(
            gamma=0.001,
            kernel="rbf",
            probability=True
        )
        #LDA()
    )

    cal_estimator = CalClass(base_estimator=base_estimator, cv=10)

    clf = HierarchicalClassifier(
        base_estimator=cal_estimator,
        class_hierarchy=class_hierarchy,
    )

    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 2, 9],
        as_str=False,
    )
    # cast the targets to strings so we have consistent typing of labels across hierarchy
    y = y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(X_train.shape)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Demonstrate using our hierarchical metrics module with MLB wrapper
    with multi_labeled(y_test, y_pred, clf.graph_) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
        print("h_fbeta_score: ", h_fbeta)
    print(clf.classes_)
    print(clf.class_hierarchy_)

    print(clf.predict(X_test[2:3]), clf.predict_proba(X_test[2:3]), y_test[2:3])

    X_test, y_test = make_digits_dataset(
        targets=[4,5],
        as_str=False,
    )

    y_test = y_test.astype(str)

    print(clf.predict(X_test[2:3]), clf.predict_proba(X_test[2:3]), y_test[2:3])


if __name__ == "__main__":
    classify_digits()