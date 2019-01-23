#!/usr/bin/env python2.7

%matplotlib inline
from PIL import Image
import os

import math
import numpy as np
import matplotlib.pyplot as plt


from skimage import measure
from skimage import io

mode=np.arange(3,13+1,1)
print(mode)
rad=np.arange(0,28+4,4)
print(rad)
#currMode=mode[0]

#################################################################
## Get plots

# Upper quartile
print(len(mode))
print(len(rad))
vals=np.zeros((len(mode),len(rad)))
#print(vals)
#Load images
for j in range(len(mode)):
    for i in range(len(rad)):
        if (i<3) and (j<7):
            fileDir='Images/beads  %s  %s.tiff' %(str(mode[j]), str(rad[i]))
        elif i>=3 and j<7:
            fileDir='Images/beads  %s %s.tiff' %(str(mode[j]), str(rad[i]))
        elif i>=3 and j>=7:
            fileDir='Images/beads %s %s.tiff' %(str(mode[j]), str(rad[i]))
        else:
            fileDir='Images/beads %s  %s.tiff' %(str(mode[j]), str(rad[i]))

        #print(fileDir)

        image = Image.open(fileDir)
        imarray=np.asarray(image)
        #print(imarray.shape)
        #plt.imshow(image,cmap='gray')
        #label_image = io.imread("./images/label-img.png")

        imvals=imarray.ravel()
        imvals=np.sort(imvals)[::-1]
        #print(imvals)

        # Upper 25% intensity values
        count=0.1*len(imvals) #0.01 0.25 0.005 0.5 0.0025 10%, 1% similar results
        count=round(count)
        #print(len(imarray.ravel()))
        print(len(imvals[0:count+1]))
        #print(imvals[0:count+1])
        quantile=imvals[0:count+1]
        #print(quantile.mean())
        vals[j][i]=quantile.mean()


#print(vals)
#print(vals[0][:])

#plt.hist(imarray, bins=10); plt.show()
xrange=np.linspace(-1.4,1.4,len(rad))
xrange2=np.linspace(-1.4,1.4,100)
#print(xrange2)
fig, ax=plt.subplots(figsize=(15, 20))
for i in range(len(mode)):
    plt.subplot(3,4,i+1)
    #print(range(len(rad)))
    #plt.imshow(

    ## fit quadratic polynomial function
    fitquad=np.polyfit(xrange,vals[i][:],2)
    p=np.poly1d(fitquad)
    plt.plot(xrange,vals[i][:],'rx',xrange2,p(xrange2),'k')#)
    #plt.plot(np.polyfit(xrange,vals[i][:],3))
    plt.title('Mode %d' %(mode[i]))
    plt.xticks(xrange)
    plt.xlabel('radians')
    plt.ylabel('Intensity')
plt.show()


print(len(mode))
print(np.linspace(-1.4,1.4,8))
print(vals)
# get top 5% and then get the bins of range 5%
