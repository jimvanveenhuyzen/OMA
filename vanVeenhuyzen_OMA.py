import numpy as np
import matplotlib.pyplot as plt

co21 = np.genfromtxt('co21.txt')

print(co21.shape)

co21_channels = co21[:,0]
co21_vel = co21[:,3]
co21_flux = co21[:,4] * 1e3

plt.plot(co21_vel,co21_flux)
plt.xlim(1000,1250)
plt.ylim(0,np.max(co21_flux)*1.1)
plt.xlabel('Velocity [km/s]') #CHANGE THIS TO KM/S
plt.ylabel('Integrated intensity [mJy/beam]')

plt.show()

#we can use np.trapz to integrate this, we just need a y[start:stop] and a dx (delta V), which we are going to need to define beforehand. 
dv = 10
print(co21_vel)

test = np.trapz(co21_flux[30:50],dx=dv)
print(test)