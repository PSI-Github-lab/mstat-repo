try:
    from numpy.core.function_base import linspace
    import os
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    from datetime import *
    from mstat.dependencies.ms_data.MSDataAnalyser import MSDataAnalyser
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def labelled_scatter(x, y, colormap, line_width, title, x_label, y_label, encoder, fig=None, clm_offset=0):
        if fig is None:
            fig = plt.figure(figsize=(9,9))
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid()

        target_names = encoder.inverse_transform(np.unique(y))
        colors = colormap(range(clm_offset, len(target_names) + clm_offset))

        for color, i, target_name in zip(colors, np.unique(y), target_names):
            try:
                plt.scatter(x[y ==i,0], x[y == i, 1], color=color, alpha=.8, lw=line_width,
                        label=target_name)
            except IndexError:
                plt.scatter(x[y ==i,0], [0]*len(x[y ==i,0]), color=color, alpha=.8, lw=line_width,
                        label=target_name)
        
        plt.legend(loc='best', shadow=False, scatterpoints=1)

        return fig

def labelled_scatter_3d(x, y, colormap, ax=None, clm_offset=0):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plt.grid()

        ax.scatter(x[:,0], x[:,1], x[:,2], color=colormap(y+clm_offset))
        #plt.legend(loc='best', shadow=False, scatterpoints=1)

        return ax

class AnalysisVis:
    '''
    This class handles plotting operations for the PCA and LDA data
    Dependencies: matplotlib, datetime
    '''
    result_folder: str
    analyser: MSDataAnalyser
    encoder: LabelEncoder

    def __init__(self, analyser, encoder, result_folder, clm) -> None:
        self.analyser = analyser
        self.encoder = encoder
        self.result_folder = result_folder
        self.colormap = clm

        try:
            if result_folder != '':
                os.mkdir(self.result_folder)
        except FileExistsError:
            pass

    def labelled_scatter(self, x, y, colormap, line_width, title, x_label, y_label):
        fig = plt.figure(figsize=(10,10))

        target_names = np.unique(y)
        colors = colormap(range(len(target_names)))

        for color, i, target_name in zip(colors, np.unique(y), target_names):
            try:
                plt.scatter(x[y ==i,0], x[y == i, 1], color=color, alpha=.8, lw=line_width,
                        label=target_name)
            except IndexError:
                plt.scatter(x[y ==i,0], [0]*len(x[y ==i,0]), color=color, alpha=.8, lw=line_width,
                        label=target_name)

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()

        return fig

    def labelled_scatter_3d(self, x, y, colormap):
        fig = plt.figure()
        self.ax1 = fig.add_subplot(projection='3d')

        
        self.ax1.scatter(x[:,0], x[:,1], x[:,2], color=colormap(y))
            

        #plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.grid()

        return fig, self.ax1

    def visualisePCA(self, save_fig=True):
        '''Visualize the results of PCA analysis and save figure to image file'''
        line_width = 2
        pca = self.analyser.dimr_pipeline['pca']

        exp1 = float('%.4g' % (100*pca.explained_variance_ratio_[0]))
        try:
            exp2 = float('%.4g' % (100*pca.explained_variance_ratio_[1]))
        except:
            exp2 = 0

        title = f'PCA-LDA capturing {exp1 + exp2}% data variance'
        x_label = f'PC 1 ({exp1}%)'
        y_label = f'PC 2 ({exp2}%)'

        fig = self.labelled_scatter(self.analyser.dimr_data, self.analyser.label_data, self.colormap, line_width, title, x_label, y_label)
        #plt.scatter(self.analyser.lda_data[-1,0], self.analyser.lda_data[-1, 1], color='red', alpha=.8, lw=line_width)

        now = str(datetime.now()).replace(' ','_').replace(':','').rsplit('.',1)[0]
        #plt.show()
        if save_fig:
            plt.savefig(f'{self.result_folder}/pca_{now}.png')

    def visualisePCA3D(self, save_fig=True):
        '''Visualize the results of PCA analysis and save figure to image file'''
    
        if self.analyser.dimr_data.shape[1] > 2:
            fig, ax1 = self.labelled_scatter_3d(self.analyser.dimr_data, self.analyser.label_data, self.colormap)
            #plt.scatter(self.analyser.lda_data[-1,0], self.analyser.lda_data[-1, 1], color='red', alpha=.8, lw=line_width)
            #plt.show()
            #if save_fig:
                #plt.savefig(f'{self.result_folder}/pca_{now}.png')
            return fig, ax1

    def visualiseLDA(self, save_fig=True):
        '''Visualize the results of LDA analysis and save figure to image file'''
        line_width = 2
        lda = self.analyser.class_pipeline['lda']

        exp1 = float('%.4g' % (100*lda.explained_variance_ratio_[0]))
        try:
            exp2 = float('%.4g' % (100*lda.explained_variance_ratio_[1]))
        except:
            exp2 = 0

        title = f'LDA capturing {exp1 + exp2}% data variance'
        x_label = f'LD 1 ({exp1}%)'
        y_label = f'LD 2 ({exp2}%)'
        
        fig = self.labelled_scatter(self.analyser.final_transform_data, self.analyser.label_data, self.colormap, line_width, title, x_label, y_label)
        #plt.scatter(self.analyser.lda_data[-1,0], self.analyser.lda_data[-1, 1], color='red', alpha=.8, lw=line_width)

        now = str(datetime.now()).replace(' ','_').replace(':','').rsplit('.',1)[0]
        #plt.show()
        plt.savefig(f'{self.result_folder}/pca_{now}.png')
