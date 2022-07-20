try:
    import joblib
    from sklearn.model_selection import learning_curve, validation_curve
    import pandas as pd
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    from PyQt5 import QtCore
    from mstat.dependencies.ScikitImports import *
    from mstat.dependencies.helper_funcs import *
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def plot_learning_curve(self, estimator, title, X, y, n_components, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.3, 0.7, 5)):
    """
    From https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py 

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot validation curve
    axes[0].grid()
    pca1 = int(0.25 * n_components)
    pca2 = int(3.00 * n_components)
    stp = int((pca2 - pca1) / 10)
    param_range = np.arange(start=pca1, stop=pca2, step=stp, dtype=int)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name="dim__n_components", param_range=param_range,
        scoring="accuracy", cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes[0].set_title("Learning Curves - PCA Components")
    axes[0].set_xlabel("PCA components")
    axes[0].set_ylabel("Score")
    lw = 2
    axes[0].plot(param_range, train_scores_mean, 'o-', label="Training score",
                color="darkorange", lw=lw)
    axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, 
                    color="darkorange", lw=lw)
    axes[0].plot(param_range, test_scores_mean, 'o-', label="Validation score",
                color="navy", lw=lw)
    axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    axes[0].legend(loc="best")
    

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       #return_times=True,
                       shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes[1].set_title("Learning Curves - Number of Samples")
    if ylim is not None:
        axes[1].set_ylim(*ylim)
    axes[1].set_xlabel(f"Training examples (out of {len(X)})")
    axes[1].set_ylabel("Score")

    # Plot learning curve
    axes[1].grid()
    axes[1].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[1].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[1].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[1].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Validation score")
    axes[1].legend(loc="best")

    plt.show()