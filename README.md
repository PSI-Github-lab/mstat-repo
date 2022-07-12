# PSI Mass Spectrometer Statistical Learning Package (MStat)

This project has a description.

## 1. Python Version & Dependencies

<i>Version: Python 3.8+</i><br />
NOTE: The Python environment needs to be included in the windows PATH variable to use the python command in a console window from any directory on a computer. See this guide [this guide](https://datatofish.com/add-python-to-windows-path/) for more info.

## 2. MStat GUI

This GUI has instructions.

### Getting Started

GUI is split into three main segments: data selection, data prep & modelling, and data visualization

work across the screen from left to right

top right in menu bar show file options. you can choose training & testing folders to find your data.

once selected, training & testing folders are displayed in a directory tree format. Data can be selected by clicking the checkbox next to a directory of interest.

When directories have been selected they will appear in the middle-center where important file information can be confirmed (class name, number of samples, etc.)

Above the list of selected data are the PCA-LDA modelling options. These include the m/z bin settings (limits and bin size), number of PCA dimensions for the LDA classifier, and functions for viewing and saving a trained model.

Data will be displayed on the cartesian graph on the right side of the window once a model has been trained. Plotting options exist under the graph for selecting what type of scores and the x- & y- axes.

### Converting Data

When directories have been selected they will appear in the middle-center where important file information can be confirmed (class name, number of samples, etc.)

Data is processed by the program through `.npy` files. These can be created by following the file conversion prompt shown on the status bar at the bottom of the program window. MZML files can also be generated, if desired.

### Creating a PCA-LDA Model

Above the list of selected data are the PCA-LDA modelling options. These include the m/z bin settings (limits and bin size), number of PCA dimensions for the LDA classifier, and functions for viewing and saving a trained model.

1/5 rule for choosing number of PCA dimensions

links to information about sci-kit learn and PCA-LDA

### Exploring the Model

Data will be displayed on the cartesian graph on the right side of the window once a model has been trained. Plotting options exist under the graph for selecting what type of scores and the x- & y- axes.

scores plots will indicate how identified clusters will classify

loadings plots will show which m/z features are most important for classification

## 3. Other Scripts

### 3.1 script_name

`Here is some code font`

## 4. CSV Data Format

| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |