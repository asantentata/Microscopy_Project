#!/usr/bin/env python2.7
import numpy as np
from scipy import misc, fftpack, ndimage
from scipy.sparse import lil_matrix
#import matplotlib.pyplpip ot as plt
from PIL import Image
import matplotlib.pyplot as plt
#plt.use("TkAgg")

from matplotlib.colors import LogNorm

def MakeMask(R, Cut):
            radius = int(R)
            mask = np.sqrt((np.arange(-radius,radius)**2).reshape((radius*2,1)) + (np.arange(-radius,radius)**2)) < Cut

            return(mask)

def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def av_freq(cut):
	mode = [str(' 3 ')]#,str(' 4 '),str(' 5 '),str(' 6 '),str(' 7 '),str(' 8 '),str(' 9 '),str('10 '),str('11 '),str('12 '),str('13 ')]
	radians = [str(' 0'),str(' 4'),str(' 8'),str('12'),str('16'),str('20'),str('24'),str('28')]
	#Variables for mask
	radius = 256
	#cut = 80
	out_mask = MakeMask(radius,cut)
	in_mask = (MakeMask(radius, cut-15))
	ring_mask = out_mask*((in_mask-1)*-1)
	#empty array for freq values
	avfreq_vals =[]
	#plotting the mean intensity of each image as we cycle through images
	mode_array = [[] for i in range(len(mode))]
	mode_count = 0
	for image in mode:

		avfreq_vals =[]
	# plotting the mean intensity of each image as we cycle through images
		count = 1
		for i in radians:
			image = Image.open("./Images/beads "+ str(mode[mode_count]) +str(i)+".tiff")
			imarray = np.asarray(image, dtype = 'uint16')
			fft2 = fftpack.fft2(imarray)
			masked_image = fftpack.fftshift(fft2)*ring_mask
			mean_pixel = np.mean(abs(masked_image))
			avfreq_vals.append(mean_pixel)
			count = count + 1
		mode_array[mode_count] = avfreq_vals
		#print(mode_count)
		#print(mode_array[mode_count])
		mode_count = mode_count + 1
	#plot_spectrum(masked_image)


	return mode_array


radius = 256
cut = 256
out_mask = MakeMask(radius,cut)
in_mask = (MakeMask(radius, cut-15))
ring_mask = out_mask*((in_mask-1)*-1)

image = Image.open("./Images/beads 10 20.tiff")
imarray = np.asarray(image, dtype = 'uint16')
fft2 = fftpack.fft2(imarray)
masked_image = fftpack.fftshift(fft2)*ring_mask
#plot_spectrum(masked_image)



########### Loop to obtain the average pixel intensity as radius of anulus increases
pix_in_rad = []
pix_check = 100
for cut in range(80,80 + pix_check):
	pix_in_rad.append(av_freq(cut))
flattened  = [val for sublist in pix_in_rad for val in sublist]
mode_by_rad = [list(x) for x in zip(*flattened)]
mode_by_rad = np.reshape(mode_by_rad,(8,pix_check))



fig, ax = plt.subplots(nrows=2, ncols=4)
#mode=np.arange(3,13+1,1)
x = range(80,80+pix_check)
xrange2=np.linspace(80,80+pix_check,100)
for collumn in ax:
	col_count = 1
	for col in collumn:

		y = mode_by_rad[col_count]
		fitquad=np.polyfit(x,y,3)
		p=np.poly1d(fitquad)
		#plt.xlabel('radians')
		#plt.ylabel('Intensity')
		col.plot(x,y,'rx',xrange2,p(xrange2),'k')
		#col.set_ylabel('Intensity')
		#col.set_xlabel('Radians')
		col_count = col_count + 1
#plt.show()
plt.show()
