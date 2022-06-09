import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


class MSFileReader:
    '''
    This class reads MS data after it has been collated and labelled in CSV files. It extracts the feature data and encodes the labels for data analysis purposes.
    Dependencies: pandas, numpy
    '''
    # variable declaration and data type hints
    my_encoder: LabelEncoder
    file_name: str
    file_frame: pd.DataFrame
    feature_data: list
    class_labels: list
    class_encoded: list

    def __init__(self, file_name, do_rand=False, rand_name='random', rand_state=42, rand_centers=[[0, 0, 0],[8,0,0],[0,8,0]], rand_std=[1.0, 1.0, 1.0], rand_num=300) -> None:
        ''' Read data from the file and create the label encoder '''
        self.my_encoder = LabelEncoder()
        self.do_rand = do_rand

        if not self.do_rand:
            # load the data from csv
            self.file_name = file_name
            temp_frame = pd.read_csv(self.file_name)

            # remove any rows with all zero values
            temp_frame.drop(temp_frame[temp_frame['total'] == 0].index, inplace=True)

            # extract feature data
            feature_cols = [col for col in temp_frame if isFloat(col)]     # feature data captured by bins are numeric column labels
            self.feature_data = temp_frame[feature_cols].values

            # create standard bin labels just in case
            self.bins = [float(bin) for bin in feature_cols]

            # extract file info data
            self.info_cols = [col for col in temp_frame if not isFloat(col)]
            self.info_data = temp_frame[self.info_cols].values

            # Recombine data with cleaned column formatting
            feature_frame = pd.DataFrame(self.feature_data, columns=self.bins)
            info_frame = pd.DataFrame(self.info_data, columns=self.info_cols)
            self.file_frame = pd.concat([info_frame, feature_frame], axis=1)
        else:
            # create random 3D blobs
            self.bins = [1,2,3]
            self.info_cols = ['','label','filename','total']

            self.feature_data, blob_labels = make_blobs(
                                            n_samples=[rand_num] * len(rand_centers),
                                            n_features=3,
                                            cluster_std=rand_std,
                                            #center_box=(-5,5),
                                            centers=rand_centers,
                                            shuffle=False,
                                            random_state=rand_state
                                                )

            class_labels = np.core.defchararray.add(np.array([f'{rand_name} ']*len(blob_labels)), blob_labels.astype(str))

            indices = [e for e in range(len(blob_labels))]
            dummy_names = [f'n/a {e}' for e in range(len(blob_labels))]
            totals = [float(sum(row)) for row in self.feature_data]

            self.info_data = np.vstack((indices, class_labels, dummy_names, totals)).T

            # Recombine data with cleaned column formatting
            feature_frame = pd.DataFrame(self.feature_data, columns=self.bins)
            info_frame = pd.DataFrame(self.info_data, columns=self.info_cols)
            self.file_frame = pd.concat([info_frame, feature_frame], axis=1)

            self.file_name = 'RANDOM'
         

    def encodeData(self):
        ''' Perform encoding operation then return file data frame, extracted feature data as a matrix, encoded labels as an array, and the encoder object itself '''
        # encode classifier data
        matrix = self.file_frame['label'].values
        self.class_labels = np.array(matrix.tolist())
        
        self.class_encoded = self.my_encoder.fit_transform(self.class_labels)

        return self.file_frame, self.feature_data, self.class_labels, self.my_encoder

    def getTICNormalization(self):
        rows, cols = self.feature_data.shape
        norm_data = np.empty(0)

        for row in range(rows):
            temp = np.empty(0)
            total = sum(self.feature_data[row])
            for element in self.feature_data[row]:
                temp = np.append(temp, element / total)
            
            try:
                norm_data = np.vstack((norm_data, temp))
            except ValueError:
                norm_data = temp

        return norm_data

    def getMaxNormalization(self):
        rows, cols = self.feature_data.shape
        norm_data = np.empty(0)

        for row in range(rows):
            temp = np.empty(0)
            total = max(self.feature_data[row])
            for element in self.feature_data[row]:
                temp = np.append(temp, element / total)
            
            try:
                norm_data = np.vstack((norm_data, temp))
            except ValueError:
                norm_data = temp

        return norm_data

    # UNFINISHED
    def removeOutliers(self, data, labels):
        clf = LocalOutlierFactor()
        classes = np.unique(labels)
        tot_inliers = []
        for c in classes:
            y = (labels == c)
            inliers = clf.fit_predict(data[y])
            tot_inliers = np.concatenate((tot_inliers, inliers))
        #return data[tot_inliers], labels[tot_inliers]
       
    # UNFINISHED
    def describeData(self, preprocess, start, end):
        mz = np.arange(start, end, self.bins[1] - self.bins[0])
        feature_data = self.getTICNormalization()

        self.removeOutliers(feature_data, self.class_encoded)

        info_frame = pd.DataFrame(self.info_data, columns=self.info_cols)
        if preprocess is not None:
            feature_data = preprocess.fit_transform(feature_data)

        feature_frame = pd.DataFrame(feature_data, columns=self.bins)
        df = pd.concat([info_frame, feature_frame], axis=1)
        df = df[mz]
        
        print(df.describe())
        return df.corr(), mz, df

    def __str__(self) -> str:
        ''' Message given when printing the class object (i.e. "print(some_MSFileReader)")'''
        return f"""
    MSFileReader(file_name = {self.file_name})
        CSV data shape: {self.file_frame.shape}
        Feature data shape: {np.shape(self.feature_data)}
        Labels data shape: {np.shape(self.class_labels)}
        Class Encoding: {self.my_encoder.transform(self.my_encoder.classes_)} from {self.my_encoder.classes_}
        """