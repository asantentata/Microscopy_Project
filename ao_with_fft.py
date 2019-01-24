#!/usr/bin/env python2.7
import numpy as np
from scipy import misc, fftpack, ndimage
from scipy.sparse import lil_matrix
#import matplotlib.pyplpip ot as plt
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


#importing TIFF image into python and converting to array
image = Image.open("./Images/beads 10 20.tiff")
imarray = np.asarray(image, dtype = 'uint16')



def av_intensity():
	mode = [str(' 3 '),str(' 4 '),str(' 5 '),str(' 6 '),str(' 7 '),str(' 8 '),str(' 9 '),str('10 '),str('11 '),str('12 '),str('13 ')]
 	radians = [str(' 0'),str(' 4'),str(' 8'),str('12'),str('16'),str('20'),str('24'),str('28')]
	mode_array = [[] for i in range(11)]
	mode_count = 0
	for image in mode:
		intensity_vals =[]
	# plotting the mean intensity of each image as we cycle through images
		count = 1
		for i in radians:
			now_pic = Image.open("./Images/beads "+ str(mode[mode_count]) +str(i)+".tiff")
			mean_pixel = np.mean(now_pic)
			intensity_vals.append(mean_pixel)
			count = count + 1
			#print(mode[mode_count])
			#print(intensity_vals)

		mode_array[mode_count] = intensity_vals
		mode_count = mode_count + 1
	return mode_array


#function for plotting the fourier transform of an image
def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()


#radius = 256
#cut = 200
#function for making a circular mask
def MakeMask(R, Cut):
            radius = int(R)
            mask = np.sqrt((np.arange(-radius,radius)**2).reshape((radius*2,1)) + (np.arange(-radius,radius)**2)) < Cut

            return(mask)

#Creating a circular mask for fourier transform image


def av_freq():
	mode = [str(' 3 '),str(' 4 '),str(' 5 '),str(' 6 '),str(' 7 '),str(' 8 '),str(' 9 '),str('10 '),str('11 '),str('12 '),str('13 ')]
	radians = [str(' 0'),str(' 4'),str(' 8'),str('12'),str('16'),str('20'),str('24'),str('28')]
	#Variables for mask
	radius = 256
	cut = 30
	out_mask = MakeMask(radius,cut)
	in_mask = (MakeMask(radius, cut*0.6))
	ring_mask = out_mask*((in_mask-1)*-1)
	#empty array for freq values
	avfreq_vals =[]
	#plotting the mean intensity of each image as we cycle through images
	mode_array = [[] for i in range(11)]
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
	plot_spectrum(masked_image)


	return mode_array
#Code to make subplots
x = range(10)
y = range(10)

'''
fig, ax = plt.subplots(nrows=2, ncols=2)
#mode=np.arange(3,13+1,1)
xrange=np.arange(-1.4,0.6,0.4)
xrange2=np.linspace(-1.4,0.2,100)
for row in ax:
	count_row = 1
	for col in row:
		y = (np.squeeze(av_freq()[count_row][2:7], axis=0))
		fitquad=np.polyfit(xrange,y,2)
		p=np.poly1d(fitquad)
		#plt.xlabel('radians')
		#plt.ylabel('Intensity')
		col.plot(xrange,y,'rx',xrange2,p(xrange2),'k')#)
		col.set_ylabel('Intensity')
		col.set_xlabel('Radians')
		count_row = count_row + 1
#plt.show()
'''
#print(range(1,9))
#print((av_freq()[1][:]))
#These are the functions that return the 11x8 arrays.
av_freq()
av_intensity()
#print("shape of feq array is ",np.shape(av_freq()))
'''
out_mask = MakeMask(radius,cut)
in_mask = (MakeMask(radius, cut*0.6))
ring_mask = out_mask*((in_mask-1)*-1)
fft2 = fftpack.fft2(imarray)
'''
#masked_image = fftpack.fftshift(fft2)*ring_mask
#plt.title('Fourier transform')
plt.show()
