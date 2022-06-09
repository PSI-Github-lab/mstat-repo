# coding: utf-8
from dependencies.ScikitImports import *
from sklearn.model_selection import validation_curve, learning_curve
import numpy as np
from matplotlib import projections, pyplot as plt, cm
import colorcet as cc
import sys, os
from datetime import *

from dependencies.ms_data.MSFileReader import MSFileReader
from dependencies.ms_data.AnalysisVis import AnalysisVis
from dependencies.ms_data.MSDataAnalyser import MSDataAnalyser
from dependencies.readModelConfig import *

#Todo

def plot_learning_curve(estimator, title, X, y, n_components, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.3, 0.7, 5)):
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
        _, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Plot validation curve
    axes[0].grid()
    param_range = np.arange(start=int(0.25 * n_components), stop=int(2.0 * n_components), step=10, dtype=int)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name="pca__n_components", param_range=param_range,
        scoring="accuracy", cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes[0].set_title("Validation Curve")
    axes[0].set_xlabel("PCA components")
    axes[0].set_ylabel("Score")
    lw = 2
    axes[0].plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    axes[0].fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    axes[0].fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    axes[0].legend(loc="best")
    

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes[1].set_title("Learning Curves")
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
                 label="Cross-validation score")
    axes[1].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[2].grid()
    axes[2].plot(train_sizes, fit_times_mean, 'o-')
    axes[2].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("Fit times (s)")
    axes[2].set_title("Scalability of the model")

    return plt

help_message = """
Console Command: python pcaldaModel.py <path/file_name.csv> <preprocessing> <pca_dim> <lda_dim> <create_learning_curve>
Arguments:
    <path/file_name.csv>    - (String) path and name of CSV file including the extension ".csv"
    <preprocessing>         - (String) preprocessing step after TIC normalization
    <pca_dim>               - (Integer) number of dimensions for PCA analysis (must be less than or equal to number of data features)
    <create_learning_curve> - (Boolean) select whether or not to create a learning curve (will add to processing time)
    <plot_pca_proj>         - (Boolean) select whether or not to plot PCA projection"""

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

preprocessing_options = ['none', 'sscl', 'rscl', 'ptfm']

def main():
    ''' CONSOLE INPUT: 
    python pcaldaModel.py <file_name.extension> <pca_dim> <lda_dim>
    OR
    python pcaldaModel.py
    OR
    python pcaldaModel.py help
    '''
    directory = 'pcalda_results'
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if not argm:
        print("Type 'python pcalda.py help' for more info")
        quit()
    else:
        file_name = argm[0]
        preprocess = argm[1]
        n_component_pca = int(argm[2])
        curve_flag = bool(int(argm[3]))
        plot_flag = bool(int(argm[4]))

    if preprocess not in preprocessing_options:
        print(f"ERROR: {preprocess} is not a valid preprocessing step. Please choose from {preprocessing_options}")

    # read data from the csv file
    file_reader = MSFileReader(file_name)
    _, _, labels, encoder = file_reader.encodeData()
    print(' DATA FROM CSV FILE '.center(80, '*'))
    print(file_reader)

    # create PCA-LDA estimator
    rnd_state = 42
    analysis = MSDataAnalyser(preprocess, n_component_pca, rnd_state, da_mode=0)
    print(analysis.class_pipeline)
    analysis.fitModel(file_reader.getTICNormalization(), labels)

    # perform cross-validation
    cv = StratifiedKFold(n_splits=10)
    cv_results = analysis.crossvalModel(cv)

    print(" Model has cross validation accuracy of %0.3f +/- %0.2f" % (cv_results['test_score'].mean(), cv_results['test_score'].std()))
    print("""   Avg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean()))

    # save PCA-LDA model file
    try:
        os.mkdir(directory)
    except:
        pass
    analysis.saveModel(f'{directory}/models', file_name.rsplit('.',1)[0].rsplit('/',1)[-1].rsplit('\\',1)[-1])

    if plot_flag:
        # visualise the resulting PCA-LDA subspace 
        clm = cm.get_cmap('cet_glasbey_light') 
        analysis.transformData()
        visualisation = AnalysisVis(analysis, encoder, directory, clm)
        #visualisation.visualisePCA3D()
        visualisation.visualiseLDA()

    if curve_flag:
        # show learning curve
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        train_sizes = np.linspace(0.3, 0.95, 10)

        title = "Learning Curves (PCA-LDA)"
        plot_learning_curve(analysis.class_pipeline, title, file_reader.getTICNormalization(), labels, n_component_pca, axes=axes, cv=cv, train_sizes=train_sizes)
        plt.tight_layout(pad=4)
    plt.show()

    print('\nPCA-LDA analysis created and saved in the results folder in current directory.')

if __name__ == "__main__":
    main()