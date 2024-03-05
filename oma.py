import numpy as np
import matplotlib.pyplot as plt

co21 = np.genfromtxt('co21.txt')

co21agn = np.genfromtxt('co21agn.txt')
co21starburst = np.genfromtxt('co21starburst.txt')
co32agn = np.genfromtxt('co32agn.txt')
co32starburst = np.genfromtxt('co32starburst.txt')

print(co21.shape)

co21_channels = co21[:,0]
co21_vel = co21[:,3]
co21_flux = co21[:,4] * 1e3

def plotInt(vel,flux,title,fig):
    plt.plot(vel,flux*1e3) #multiply by 1e3 to convert to mJy, looks better in the plots
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Integrated intensity [mJy/beam]')
    plt.title(title)
    plt.savefig(fig)
    plt.show()

co21agn_vel = co21agn[:,3]
co21agn_flux = co21agn[:,4] 
co21starburst_vel = co21starburst[:,3]
co21starburst_flux = co21starburst[:,4] 
co32agn_vel = co32agn[:,3]
co32agn_flux = co32agn[:,4] 
co32starburst_vel = co32starburst[:,3]
co32starburst_flux = co32starburst[:,4] 

plotInt(co21agn_vel,co21agn_flux,'Flux (mJy) vs the velocity (km/s) of CO21: AGN','co21agn.png')
plotInt(co21starburst_vel,co21starburst_flux,'Flux (mJy) vs the velocity (km/s) of CO21: Starburst','co21starburst.png')
plotInt(co32agn_vel,co32agn_flux,'Flux (mJy) vs the velocity (km/s) of CO32: AGN','co32agn.png')
plotInt(co32starburst_vel,co32starburst_flux,'Flux (mJy) vs the velocity (km/s) of CO32: Starburst','co32starburst.png')

#we can use np.trapz to integrate this, we just need a y[start:stop] and a dx (delta V), which we are going to need to define beforehand. 
dv = 10 #using visual inspection of the velocity, it seems dv=10

co21agn_int = np.trapz(co21agn_flux,dx=dv)
co21starburst_int = np.trapz(co21starburst_flux,dx=dv)
co32agn_int = np.trapz(co32agn_flux,dx=dv)
co32starburst_int = np.trapz(co32starburst_flux,dx=dv)
print('The integrated intensity of the CO21 AGN region is {:.3f} Jy'.format(co21agn_int))
print('The integrated intensity of the CO21 Starburst region is {:.3f} Jy'.format(co21starburst_int))
print('The integrated intensity of the CO32 AGN region is {:.3f} Jy'.format(co32agn_int))
print('The integrated intensity of the CO32 Starburst region is {:.3f} Jy'.format(co32starburst_int))