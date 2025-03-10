import numpy as np
import matplotlib.pyplot as plt

#Pre-define some constants
c = 3e5 #km/s 
h = 6.626e-27 #erg*s
k = 1.38e-16 #erg/K 

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
    #plt.savefig(fig)
    #plt.show()
    plt.close()

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

def compute_beam(theta_maj,theta_min):
    #Convert first from arcsec to radians:
    arcsec2_to_sr = np.pi / (3600 * 180)
    return (np.pi*theta_maj*theta_min) / (4*np.log(2)) * arcsec2_to_sr**2

#Compute the starburst beams 
co21starburst_beam = compute_beam(0.12,0.12)
co32starburst_beam = np.copy(co21starburst_beam) #CO32 uses the same beam size 
print(co21starburst_beam)

#Compute the AGN beams 
co21agn_beam = compute_beam(0.025,0.025)
co32agn_beam = compute_beam(0.021,0.021)

#Use the CDMS catalogue to obtain these values:
#Einstein A coefficients
A_21 = 10**(-6.1605)
A_32 = 10**(-5.6026)

#Upper level energies: note CDMS lists lower level energies, so your upper level is 1 J higher! 
E_u21 = 11.5359  #cm^-1
E_u32 = 23.0695  #cm^-1



#Compute the statistical weights using g_J = 2J+1
g_21 = 2*2 + 1 
g_32 = 2*3 + 1

def Nthin_u(F_int,A,omega):
    #Compute the column density in Jy of the upper level in the optically thin emission case 
    energy_factor = 1e-23 #Go from Janksy to erg
    return (4*np.pi*F_int)/(A*omega*h*c) * energy_factor 

#Compute all the N_u (optically thin case and LTE)
co21starburst_Nu = Nthin_u(co21starburst_int,A_21,co21starburst_beam)
co32starburst_Nu = Nthin_u(co32starburst_int,A_32,co32starburst_beam)
co21agn_Nu = Nthin_u(co21agn_int,A_21,co21agn_beam)
co32agn_Nu = Nthin_u(co32agn_int,A_32,co32agn_beam)
print('The optically thin upper column density for CO21 starburst is: {:.3e} cm^-2'.format(co21starburst_Nu))
print('The optically thin upper column density for CO32 starburst is: {:.3e} cm^-2'.format(co32starburst_Nu))
print('The optically thin upper column density for CO21 AGN is: {:.3e} cm^-2'.format(co21agn_Nu))
print('The optically thin upper column density for CO32 AGN is: {:.3e} cm^-2'.format(co32agn_Nu))

co21starburst_Nu_std = 0.1*co21starburst_Nu
co21agn_Nu_std = 0.1*co21agn_Nu
co32starburst_Nu_std = 0.1*co32starburst_Nu
co32agn_Nu_std = 0.1*co32agn_Nu

def lnNu_error(Nu_error,F_int,A,omega):
    return Nu_error/F_int

co21starburst_lnNu_std = np.log(lnNu_error(co21starburst_Nu_std,co21starburst_int,A_21,co21starburst_beam)/g_21)
co32starburst_lnNu_std = np.log(lnNu_error(co32starburst_Nu_std,co32starburst_int,A_32,co32starburst_beam)/g_32)

print('The errors are:')
print(co21starburst_Nu_std)
print(lnNu_error(co21starburst_Nu_std,co21starburst_int,A_21,co21starburst_beam))
print(co21starburst_lnNu_std)

def E_u(E_cm):
    #Convert energies in cm^-1 to energies in K 
    return (E_cm*h*c)/k * 1e5 #factor 1e5 to convert c from km/s to cm/s 

T_21 = E_u(E_u21)
T_32 = E_u(E_u32)

print('The rotational temperature of the J=2 to J=1 transition of CO is {:.3f}'.format(T_21))
print('The rotational temperature of the J=3 to J=2 transition of CO is {:.3f}'.format(T_32))

from scipy.optimize import curve_fit

def func(x,a,b):
    return a*x + b

popt_starburst,pcov_starburst = curve_fit(func,[T_21,T_32],[np.log(co21starburst_Nu/g_21),np.log(co32starburst_Nu/g_32)],\
                                          sigma=[0.1*np.log(co21starburst_Nu/g_21),0.1*np.log(co32starburst_Nu/g_32)],\
                                            absolute_sigma=True)
popt_agn,pcov_agn = curve_fit(func,[T_21,T_32],[np.log(co21agn_Nu/g_21),np.log(co32agn_Nu/g_32)],\
                              sigma=[0.1*np.log(co21agn_Nu/g_21),0.1*np.log(co32agn_Nu/g_32)],\
                                absolute_sigma=True)

#p,cov = np.polyfit([T_21,T_32],[np.log(co21agn_Nu/g_21),np.log(co32agn_Nu/g_32)],deg=1,cov=True)
#print('Using np.polyfit gives:',p,cov)

print(popt_starburst)
print(popt_agn)

print('The fit-errors are:')
print(np.diag(pcov_starburst))
print(np.diag(pcov_agn))

rotT_starburst = -1/popt_starburst[0]
Ntot_overZ_starburst = np.exp(popt_starburst[1]) 

rotT_agn = -1/popt_agn[0]
Ntot_overZ_agn = np.exp(popt_agn[1])
print('The rotational temperature and ln(N_tot/Z(T)) for the starburst region is:')
print(rotT_starburst)
print('The rotational temperature and ln(N_tot/Z(T)) for the AGN is:')
print(rotT_agn)

#We can obtain the partition function Z using interpolation, values obtained from the CDMS catalogue: 
logZ = [0.1478,0.3389,0.5733,0.8526,1.1429,1.4386,1.7370,1.9123,2.0369,2.2584,2.5595]
T = [2.725,5,9.375,18.75,37.5,75,150,225,300,500,1000]

T_interp = np.linspace(2.725,1000,1000000)
logZ_interp = np.interp(T_interp,T,logZ)

T_starburst_idx = np.abs(T_interp-rotT_starburst).argmin()
T_agn_idx = np.abs(T_interp-rotT_agn).argmin()
logZ_starburst = logZ_interp[T_starburst_idx]
logZ_agn = logZ_interp[T_agn_idx]

print('The values of the partition functions log(Z) are:')
print(logZ_starburst,logZ_agn)

Ntot_starburst = Ntot_overZ_starburst*(10**logZ_starburst)
Ntot_agn = Ntot_overZ_agn*(10**logZ_agn)
print('The total column density of the starburst region is: {:.3e}'.format(Ntot_starburst))
print('The total column density of the agn is: {:.3e}'.format(Ntot_agn))

###########################
#SOME THINGS NOT RELEVANT PAST THIS: ONLY THE ROT PLOT 
###########################
print('-'*200)
T_21idx = np.abs(T_interp - T_21).argmin()
T_32idx = np.abs(T_interp - T_32).argmin()
print('Co21:')
print('The closest temperature is {:.3f} K'.format(T_interp[T_21idx]))
logZ_21 = logZ_interp[T_21idx]
print('The corresponding log10 value of the partition function is {:.3f}'.format(logZ_21))
print('Co32:')
print('The closest temperature is {:.3f} K'.format(T_interp[T_32idx]))
logZ_32 = logZ_interp[T_32idx]
print('The corresponding log10 value of the partition function is {:.3f}'.format(logZ_32))


plt.plot(T,logZ,label='data')
plt.plot(T_interp,logZ_interp,linestyle='dashed',color='black',label='interpolated')
plt.xlabel('Temperature')
plt.ylabel('Log10(Q(T))')
plt.title('Partition function values at various temperatures')
plt.legend()
plt.show()

def Ntot(Nupper_thin,g,Z,E_upper,T):
    exponent = (E_upper*h*c*1e5) / (k*T) #compute the exponent in exp 
    return (Nupper_thin/g) * (10**Z) * np.exp(exponent) #use 10**Z since Z is in log10 units

co21starburst_Ntot = Ntot(co21starburst_Nu,g_21,logZ_21,E_u21,T_21)
co32starburst_Ntot = Ntot(co32starburst_Nu,g_32,logZ_32,E_u32,T_32)
co21agn_Ntot = Ntot(co21agn_Nu,g_21,logZ_21,E_u21,T_21)
co32agn_Ntot = Ntot(co32agn_Nu,g_32,logZ_32,E_u32,T_32)
print('The total column density for CO21 starburst is: {:.3e} cm^-2'.format(co21starburst_Ntot))
print('The total column density for CO32 starburst is: {:.3e} cm^-2'.format(co32starburst_Ntot))
print('The total column density for CO21 AGN is: {:.3e} cm^-2'.format(co21agn_Ntot))
print('The total column density for CO32 AGN is: {:.3e} cm^-2'.format(co32agn_Ntot))

co21starburst_Nu_gu = np.log(co21starburst_Nu/g_21)
co32starburst_Nu_gu = np.log(co32starburst_Nu/g_32)
co21agn_Nu_gu = np.log(co21agn_Nu/g_21)
co32agn_Nu_gu = np.log(co32agn_Nu/g_32)

energy_array = np.array([T_interp[T_21idx],T_interp[T_32idx]])

plt.errorbar(energy_array,np.array([co21starburst_Nu_gu,co32starburst_Nu_gu]),\
             yerr=np.array([0.1,0.1]),color='black',zorder=100,fmt='o')
plt.errorbar(energy_array,np.array([co21agn_Nu_gu,co32agn_Nu_gu]),\
             yerr=np.array([0.1,0.1]),color='black',zorder=200,fmt='o')
#plt.errorbar(energy_array,np.array([co21agn_Nu_gu,co32agn_Nu_gu]),color='red',zorder=120)
plt.plot(energy_array,func(energy_array,*popt_starburst),label='Starburst',color='cyan')
plt.plot(energy_array,func(energy_array,*popt_agn),label='AGN',color='orange')
#plt.plot(energy_array,np.array([co21starburst_Nu_gu,co32starburst_Nu_gu]),label='Starburst',color='cyan')
#plt.plot(energy_array,np.array([co21agn_Nu_gu,co32agn_Nu_gu]),label='AGN',color='orange')
plt.xlabel('Energy of upper level [K]')
plt.ylabel('Upper level column density per transition: ln(N_u/g_u)')
plt.title('Rotational diagram of CO21 and CO32 transitions for starburst and AGN')
plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=False)
plt.legend()
plt.savefig('rotational_diagram_26032024.png')
plt.show()