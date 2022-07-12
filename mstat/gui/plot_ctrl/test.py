import os
import random
import matplotlib
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(__file__)
logo_path = "../../img/POINTLOGOarrowhead_128x128.gif"
abs_logo_path = os.path.join(script_dir, logo_path)

print('IMAGE TESTING'.center(80, '*'))
with cbook.get_sample_data(abs_logo_path) as file:
    logo = image.imread(file)
print(np.array(logo).shape)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

with cbook.get_sample_data(abs_logo_path) as file:
    raw_im = image.imread(file)
    im = []
    for row in raw_im:
        new_row = []
        for item in row:
            #print(item)
            if (item[0]==0) and (item[1]==0) and (item[2]==0) and (item[3]==255):
                new_row.append(np.array([item[0], item[1], item[2], 0]).astype('uint8'))
            else:
                new_row.append(np.array(item).astype('uint8'))
        im.append(new_row)
    print(type(raw_im[0][0][3]), type(im[0][0][3]))
    print(np.array(raw_im).shape, np.array(im).shape)

fig, ax = plt.subplots()

ax.plot(np.sin(10 * np.linspace(0, 1)), '-o', ms=20,
         alpha=0.7, mfc='orange')
fig.figimage(im, 100, 50, zorder=3, alpha=.5)

plt.show()