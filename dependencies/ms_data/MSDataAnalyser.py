import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV as CalClass
import numpy as np
import os
import joblib
from datetime import *

from sklearn.cross_decomposition import PLSRegression
class PLS(PLSRegression):

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X,Y).transform(X)

from ..readModelConfig import *

class MSDataAnalyser:
    '''
    This class handles PCA and LDA analyses after the MS data has been encoded+split into label array and feature data matrix.
    '''
    fitted = False
    da_mode = None

    def __init__(self, preprocess, n_component_dim_reduct, rnd_state=13, da_mode=0) -> None:
        ''' Specify the estimator pipeline '''
        self.da_mode = da_mode
        if da_mode == 0:    # PCA-LDA mode
            base_pipeline = createModelPipeline([
                                        [preprocess],
                                        ['pca', n_component_dim_reduct],
                                        ['lda']
                                        ], rnd_state)
        elif da_mode == 1:  # PLS-QDA mode
            base_pipeline = createModelPipeline([
                                        [preprocess],
                                        ['pca', n_component_dim_reduct],
                                        ['qda']
                                        ], rnd_state)
        else:               # PLS-LDA mode
            base_pipeline = createModelPipeline([
                                        [preprocess],
                                        ['pls', n_component_dim_reduct],
                                        ['lda']
                                        ], rnd_state)
        
        self.class_pipeline = base_pipeline#CalClass(base_estimator=base_pipeline, cv=3, ensemble=False)


    def fitModel(self, data, labels, verbose=False):
        ''' fit estimator to provided data/labels '''
        self.feature_data = data
        self.label_data = labels
    
        self.class_pipeline.fit(data, labels)
        try:
            self.dimr_pipeline = self.class_pipeline.calibrated_classifiers_[0].base_estimator[:-1]
        except AttributeError:
            self.dimr_pipeline = self.class_pipeline[:-1]

        self.fitted = True

    def transformData(self, data=None, labels=None, verbose=False):
        ''' transform data with trained estimator '''
        if data is None:
            data = self.feature_data
            labels = self.label_data
        
        if not self.fitted:
            self.fitModel(data, labels, verbose=verbose)
        
        self.dimr_data = self.dimr_pipeline.transform(data)
        if self.da_mode == 1:
            print("Using QDA model. No data transform beyond PCA projection is possible.")
            self.final_transform_data = None
        else:
            try:
                self.final_transform_data = self.class_pipeline.calibrated_classifiers_[0].base_estimator.transform(data)
            except AttributeError:
                self.final_transform_data = self.class_pipeline.transform(data)
        
        return self.final_transform_data


    def crossvalModel(self, cv=5):
        try:
            return cross_validate(self.class_pipeline, self.feature_data, self.label_data, cv=cv, scoring='accuracy')
        except:
            return cross_validate(self.class_pipeline, self.feature_data, self.label_data, cv=cv, scoring='accuracy')

    def testModel(self, data, true_labels, label_names):
        try:
            predicted = self.class_pipeline.predict(data)
        except:
            predicted = self.class_pipeline.predict(data)
        if len(np.unique(predicted)) != len(np.unique(true_labels)):
            print(f"Model fails to predict every classes. Only see labels {np.unique(predicted)}")
        return classification_report(true_labels, predicted, target_names=label_names), confusion_matrix(true_labels, predicted, normalize='true')

    def saveModel(self, directory, name):
        ''' Save the PCA models to external files '''
        # Get raw matrix values that can be used by linalg operations
        #pca_proj_matrix = np.transpose(self.my_pca.components_)
        #lda_proj_matrix = self.my_lda.scalings_
        
        try:
            os.mkdir(directory)
        except:
            pass
        
        # save raw matrices and the PCA/LDA objects
        now = str(datetime.now()).replace(' ','_').replace(':','').rsplit('.',1)[0]
        #np.savetxt(f'{directory}/pca_components_{name}.txt', pca_proj_matrix, delimiter=',', newline='\n')
        #np.savetxt(f'{directory}/pcalda_scalings_{name}.txt', lda_proj_matrix, delimiter=',', newline='\n')
        joblib.dump(self.class_pipeline, f'{directory}/{name}.model')