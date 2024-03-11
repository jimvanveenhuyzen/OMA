import numpy as np
import matplotlib.pyplot as plt

#Problem 1a
def f(t,tc,x):
    B = 16.6
    return B * x**(-5/8) * (tc-t)**(-3/8)

time = np.linspace(0,1,100)

plt.plot(time,f(time,0.5,20),label='tc = 0.5, x=20')
plt.plot(time,f(time,0.5,30),label='tc = 0.5, x=30')
plt.plot(time,f(time,0.5,40),label='tc = 0.5, x=40')
plt.plot(time,f(time,0.8,20),label='tc = 0.8, x=20')
plt.plot(time,f(time,1,20),label='tc = 1, x=20')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('Frequency [Hz]')
plt.show()
plt.close()

#Problem 1b

def phi(t,tc,x):
    B = 16.6
    return 2 * np.pi * B * x**(-5/8) * (-3/8) * (tc-t)**(-5/8)

def h(t,tc,x):
    return f(t,tc,x)**(2/3) * np.cos(phi(t,tc,x))

time_new = np.linspace(-0.4,1,100)
plt.plot(time_new,h(time_new,0.5,15),label='tc = 0.5, x=15')
plt.plot(time_new,h(time_new,0.5,30),label='tc = 0.5, x=30')
plt.plot(time_new,h(time_new,0.5,44),label='tc = 0.5, x=45')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('Strain h(t)')
plt.show()
plt.close()

#Problem 2
def h_new(t,tc,x,phic):
    A = 1
    eta = 1 
    M = 1
    D = 1
    return (4 * A * eta * M * (np.pi*M)**(2/3))/D * (16.6 * x**(-5/8) * (tc-t)**(-3/8))**(2/3) * np.cos((-39.11 * x**(-5/8) * (tc-t)**(5/8))+phic)

time_ = np.linspace(-2*np.pi,2*np.pi,100)
plt.plot(time_,h_new(time_,0.5,20,0),label='phi=0')
plt.plot(time_,h_new(time_,0.5,20,1),label='phi=1')
plt.plot(time_,h_new(time_,0.5,20,2),label='phi=2')
plt.plot(time_,h_new(time_,0.5,20,3),label='phi=3')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('h(t)')
plt.show()

plt.plot(time_,h_new(time_,0.03,20,0),label='0.03')
plt.plot(time_,h_new(time_,0.1,20,0),label='0.1')
plt.plot(time_,h_new(time_,0.3,20,0),label='0.3')
plt.plot(time_,h_new(time_,0.99,20,0),label='0.99')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('h(t)')
plt.show()

plt.plot(time_,h_new(time_,0.5,10,0),label='20')
plt.plot(time_,h_new(time_,0.5,30,0),label='30')
plt.plot(time_,h_new(time_,0.5,40,0),label='40')
plt.plot(time_,h_new(time_,0.5,65,0),label='65')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('h(t)')
plt.show()