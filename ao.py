#!/usr/bin/env python2.7
import numpy as np
from scipy import misc
from scipy.sparse import lil_matrix
#import matplotlib.pyplpip ot as plt
from PIL import Image
import matplotlib.pyplot as plt



#importing TIFF image into python and converting to array
image = Image.open("./Images/beads 10  0.tiff")
imarray = np.asarray(image)

#converting the image to greyscale and probing data

#plt.imshow(imarray, cmap = plt.cm.gist_gray)
print('max pixel value is ',np.max(imarray))
print('min pixel value is ',np.min(imarray))
print('shape of array is ',np.shape(imarray))


image_index = [str(' 0'),str(' 4'),str(' 8'),12,16,20,28]
intensity_vals =[]

# plotting the mean intensity of each image as we cycle through images
for i in image_index:
	now_pic = Image.open("./Images/beads 10 "+str(i)+".tiff")
	mean_pixel = np.mean(now_pic)
	intensity_vals.append(mean_pixel)
print(intensity_vals)

plt.plot([1, 2, 3, 4, 5, 6, 7], intensity_vals, 'ro')
plt.show()top
