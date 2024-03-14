#First you will need to import the modules we will need
from astropy.io import fits  # astropy for the fits handling 
import numpy as np   # numpy for the arrays and fast calculations
import os
import matplotlib.pyplot as plt #For the plotting of the images

data_q2 = fits.open('image1_2024.fits')[0].data #open the data 

split_data = np.zeros((32,128,4)) #create an empty array to fill with the data, sorted per channel 
columns = len(data_q2[0])
for j in range(columns):
    #Here we loop through the columns, adding the data to the array we pre-defined per interval of 4 columns
    #we can cleverly do this using the modulus of 4, since the channels alternate per 4 columns. 
    N = int(j//4)
    if j % 4 == 0: 
        split_data[N,:,0] = data_q2[j,:]
    elif j % 4 == 1: 
        split_data[N,:,1] = data_q2[j,:]
    elif j % 4 == 2: 
        split_data[N,:,2] = data_q2[j,:]
    else: 
        split_data[N,:,3] = data_q2[j,:]
        
offsets = np.mean(split_data,axis=0)
#Print out the result 
print('The offsets in counts for the whole channels are\n',offsets)

print('The mean offset per channel in counts is\n',np.mean(offsets,axis=0))

data_sub = np.copy(data_q2)
for j in range(columns):
    #Now we loop through all the columns once again, subtracting the appropiate value of the offset per column (channel sorted)
    M = int(j%4)
    if j % 4 == 0: 
        data_sub[j,:] -= offsets[:,M] 
    elif j % 4 == 1: 
        data_sub[j,:] -= offsets[:,M] 
    elif j % 4 == 2: 
        data_sub[j,:] -= offsets[:,M] 
    else: 
        data_sub[j,:] -= offsets[:,M] 
    
#Compute the background of the image by taking the mean at a seemingly empty patch to improve visibility of the galaxy.
background = np.mean(data_sub[50:70,50:70])
data_sub -= background

# write it back to a fits file
fits.writeto('prob2b_vanVeenhuyzen.fits', data_sub, overwrite=True)

"""
The galaxy is now (somewhat) visible, it appears to be at around the pixel coordinates of (x,y) = (37,87)
"""