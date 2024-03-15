#First you will need to import the modules we will need
from astropy.io import fits  # astropy for the fits handling 
import numpy as np   # numpy for the arrays and fast calculations
import os
import matplotlib.pyplot as plt #For the plotting of the images

data_q4a = fits.open('image3a_2024.fits')[0].data #open the data 
data_q4b = fits.open('image3b_2024.fits')[0].data
data_q4c = fits.open('image3c_2024.fits')[0].data

#Problem 4a
data_q4a_sum = np.sum(data_q4a,axis=0) #sum up the five exposures taken at the first position 

plt.imshow(np.log10(data_q4a_sum))
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('Summed Image at the first position, log scale')
plt.colorbar()
plt.show()

"""
One of the dominant structures we can see in the image is some form of (thermal) glow, kind off in the shape of a cross in the image.
The second seems to be a great amount of hot pixels scattered across the image, with an elevated count level. 
"""

#Write the result to a fits file 
fits.writeto('prob4a_vanVeenhuyzen.fits', data_q4a_sum, overwrite=True)

#Problem 4b
#First we combine all 15 frames by concatenating 
concat = np.concatenate((data_q4a,data_q4b),axis=0)
merged_data = np.concatenate((concat,data_q4c),axis=0)

#Then, we take the median of each pixel in the 128 by 128 image, so in total we take the median of 15 frames per pixel 
#This gives us the filtered straylight image to clean the data with 
median_pixel = np.median(merged_data,axis=0)

plt.imshow(median_pixel)
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('Filtered straylight image')
plt.colorbar()
plt.show()

#Write the result to a fits file 
fits.writeto('prob4b_vanVeenhuyzen.fits', median_pixel, overwrite=True)

#Problem 4c
flat = fits.open('flat3_2024.fits')[0].data

#Compute the mean
flat_mean = np.mean(flat)
#Normalize the image by the mean: 
flat_norm = flat/flat_mean
#Finally, divide the normalized flat into all the images in the image cubes 
data_q4a_div = data_q4a/flat_norm
data_q4b_div = data_q4b/flat_norm
data_q4c_div = data_q4c/flat_norm

plt.imshow(flat_norm)
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('The normalized flat field image')
plt.colorbar()
plt.show()

#Write the result to a fits file 
fits.writeto('prob4c_vanVeenhuyzen.fits', flat_norm, overwrite=True)

#Problem 4d

#First we subtract the median pixel like in problem b
data_q4a_sub = data_q4a - median_pixel
data_q4b_sub = data_q4b - median_pixel
data_q4c_sub = data_q4c - median_pixel

#Do the flat subtraction like in problem c:
data_q4a_subdiv = data_q4a_sub/flat_norm
data_q4b_subdiv = data_q4b_sub/flat_norm
data_q4c_subdiv = data_q4c_sub/flat_norm

#Now, we combine the relevant regions from the sky from all 15 frames using proper indexing. 
#We only select the central 68 pixels in the middle image since that part of the sky is covered by all three images
data_q4a_slice = data_q4a_subdiv[:,:,:-60]
data_q4b_slice = data_q4b_subdiv[:,:,30:-30]
data_q4c_slice = data_q4c_subdiv[:,:,60:]

#Finally, we concatenate the images an take the mean of these images to obtain our final result 
conc = np.concatenate((data_q4a_slice,data_q4b_slice),axis=0)
data_combined = np.concatenate((conc,data_q4c_slice),axis=0)
#Take the mean of all 15 (cleaned) frames 
data_coadd_comb = np.mean(data_combined,axis=0)

#Display the result 
plt.imshow(data_coadd_comb,vmin=-0.6,vmax=1.3,cmap='hsv')
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('The final image from the observations')
plt.colorbar()
plt.show()

"""
It seems that we can (barely) make the brown dwarf, more easily in DS9, at around pixel coordinates (x,y) = (36,75).
In DS9, using a linear zscale with b color allows you to see it best!
"""
fits.writeto('prob4d_vanVeenhuyzen.fits', data_coadd_comb, overwrite=True)