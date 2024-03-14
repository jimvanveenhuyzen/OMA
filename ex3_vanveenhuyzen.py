#First you will need to import the modules we will need
from astropy.io import fits  # astropy for the fits handling 
import numpy as np   # numpy for the arrays and fast calculations
import os
import matplotlib.pyplot as plt #For the plotting of the images

data_q3 = fits.open('image2_2024.fits')[0].data #open the data 

#Lets first inspect the image before cleaning
plt.imshow(data_q3)
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('The image before cleaning')
plt.colorbar()
plt.show()

"""
We clearly see the periodic noise pattern talked about in the assignment. We can actually easily remove this noise pattern
by taking the Fourier Transform of the image, from which we can identify the noise component since it is periodic in nature,
and the Fourier Transform shows exactly that: the periodic components inside an image. Luckily, I am also taking HCI 
and they use a nice function to FT and inverse FT an image, which Im assuming we are allowed to use. 
"""

#Copy the FFT functions from the course "High Contrast Imaging"
def FFT(c):
    """FFT(c) - carry out a complex Fourier transform
            c - the input 2D Complex numpy array
            Returns the `Complex` FFT padded array"""
    from numpy.fft import fft2,fftshift,ifft2,ifftshift
    psfA = fftshift(fft2(ifftshift(c)))
    return psfA

def IFFT(c):
    """IFFT(c) - carry out the complex Fourier transform
            and return the FFT padded array"""
    from numpy.fft import fft2,fftshift,ifft2,ifftshift
    psfA = fftshift(ifft2(ifftshift(c)))
    return psfA

data_q3_fft = FFT(data_q3) #obtain the FFT
fft_mag = np.abs(data_q3_fft)**2 #calculate the magnitude and plot it to inspect the image 

plt.imshow(np.log10(np.abs(fft_mag))) #plot on logaritmic scale so that the noise is easier to spot
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('The magnitude of the FFT of the image, log scale')
plt.colorbar()
plt.show()

#We can easily identify the noise to be stemming from a horizontal and vertical line in Fourier space, which we can 
#filter out by setting those values to 0 
data_q3_fft[256,:] = 0
data_q3_fft[:,256] = 0

#Now, we can convert back to real space using the inverse FFT function
data_q3_filter = np.abs(IFFT(data_q3_fft))

plt.imshow(data_q3_filter)
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.title('The image after cleaning/filtering of the periodic noise')
plt.colorbar()
plt.show()

# write it back to a fits file
fits.writeto('prob3_vanVeenhuyzen.fits', data_q3_filter, overwrite=True)

"""
There does appear to be a faint source in the image, namely in the region around pixel (x,y) = (202,378). Best visible in DS9.
"""