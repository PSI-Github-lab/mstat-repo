from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.cross_decomposition import PLSRegression
class PLS(PLSRegression):

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X,Y).transform(X)


def readModelConfig(file_name):
    model_configs = []
    lines = []
    with open(file_name) as file:
        lines = file.read().splitlines()
        for line in lines:
            config = line.split(',')
            for i in range(len(config)):
                config[i] = config[i].split(' ')
            model_configs.append(config)

    return model_configs, lines

def createModelPipeline(config, rnd_state):
    steps = []
    for step in config:
        if step[0] == 'pca':
            try:
                steps.append(('pca', PCA(n_components=int(step[1]), random_state=rnd_state)))
            except:
                steps.append(('pda', PCA(random_state=rnd_state)))
        elif step[0] == 'pls':
            try:
                steps.append(('pls', PLS(n_components=int(step[1]))))
            except:
                steps.append(('pls', PLS()))
        elif step[0] == 'lda':
            try:
                steps.append(('lda', LDA(n_components=int(step[1]), solver=step[2], store_covariance=True)))
            except:
                steps.append(('lda', LDA(store_covariance=True)))
        elif step[0] == 'qda':
            steps.append(('qda', QDA(store_covariance=True)))
        elif step[0] == 'sscl':
            steps.append(('sscl', StandardScaler()))
        elif step[0] == 'rscl':
            steps.append(('rscl', RobustScaler()))
        elif step[0] == 'ptfm':
            steps.append(('ptfm', PowerTransformer()))
        elif step[0] == 'gnb':
            steps.append(('gnb', GaussianNB()))
        elif step[0] == 'dummy':
            steps.append(('dummy', DummyClassifier(strategy="uniform", random_state=rnd_state)))
        elif step[0] == 'log':
            steps.append(('log', LogisticRegression(random_state=rnd_state, max_iter=1000)))
        elif step[0] == 'tre':
            steps.append(('tre', DecisionTreeClassifier(random_state=rnd_state)))
        elif step[0] == 'svc':
            steps.append(('svc', SVC(random_state=rnd_state)))
        elif step[0] == 'knn':
            try:
                steps.append(('knn', KNeighborsClassifier(n_neighbors=int(step[1]))))
            except:
                steps.append(('knn', KNeighborsClassifier()))
        elif step[0] == 'rfc':
            steps.append(('rfc', RandomForestClassifier(random_state=rnd_state)))
        else:
            continue
    
    return Pipeline(steps)