#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.integrate import quad


# This script aims to compute the shear stress on the interface of two droplets in close proximity and translating toward each other
# The method heavily relies on the work of Davis et al. (PoF 1989)

#logarithmic sampling of the radial direction
r = np.logspace(-2.33333, 0.63333, 87)

#criterion for the convergence of the numerical calculated integral (if not specified ~1e-8)
eps=2e-6

######################################################
######### Definitions (Nemer et al. JFM 2013) ########
######################################################

def zeta(r,R):
  return R/r
  
def k(r,R):
  return 1/(2*np.pi)*special.ellipk(4*zeta(r,R)/(1+zeta(r,R))**2)

def e(r,R):
  return 1/(2*np.pi)*special.ellipe(4*zeta(r,R)/(1+zeta(r,R))**2)

def phi(r,R):
  return (1+zeta(r,R)**2)/(1+zeta(r,R))*k(r,R)-(1+zeta(r,R))*e(r,R)

def singularity(r,R):
  return -1/2/np.pi*np.log(abs(R-r)/r)


######################################################
######### Dimensionless quantities ###################
######################################################

def h(R):
  return 1 + R**2

def u(R):
  return -R/h(R)

def dudR(R):
  return -1/h(R) + 2*R**2/h(R)**2

def d2udR2(R):
  return 6*R/h(R)**2-8*R**3/h(R)**3



##################################################################################
######### Calculations of the shear stress f = f_s + f_ns (dimensionless) ########
##################################################################################

#singular part of the shear stress
def f_s(r):
  return -(2+2*r**2+np.pi*r**3+3*np.pi*r+4*np.log(r))/(np.pi*(1+r**2)**2)

def integrand(R,r):
  return 4*(phi(r, R)+1/2/np.pi*np.log(abs(R-r)/r))*(u(R)/R**2 - dudR(R)/R - d2udR2(R))

#non-singular part of the shear stress
def f_ns(r):
  return quad(integrand, 0, np.inf, args=(r))[0]

#singular part of the shear stress without the logarithm singularity
def f_s_withoutsing(r):
  return -(2*r**2+np.pi*r**3+3*np.pi*r)/(np.pi*(1+r**2)**2)

#non singular part of the shear stress without the logarithm singularity
def f_ns_withoutsing(r):
  return quad(integrand, 0, np.inf, args=(r))[0]-(4*np.log(r)+2)/np.pi/(1+r**2)**2


vec_f_ns = np.vectorize(f_ns)
vec_f_ns_withoutsing = np.vectorize(f_ns_withoutsing)



###################################################################################
######### Calculations of the pressure within the film p = -2 \int f/h dr  ########
###################################################################################

def p_s_withoutsing(r):
  return 2*(2*r**2+np.pi*r**3+3*np.pi*r)/(np.pi*(1+r**2)**2)/(1+r**2)

def p_ns_withoutsing(r):
  return -2*(quad(integrand, 0, np.inf, args=(r),epsabs=eps)[0]-(4*np.log(r)+2)/np.pi/(1+r**2)**2)/(1+r**2)

def p(r):
  return p_s_withoutsing(r)+p_ns_withoutsing(r)

pressure = np.zeros(len(r))

for i in range(len(r)):
  pressure[i] = quad(p,r[i], np.inf,epsabs=eps)[0]



######################################################################################
######### Calculations of the force on the droplet F = 2 \pi \int f/h r^2 dr  ########
######################################################################################
def integrand_F(r):
  return 2*np.pi*(f_s_withoutsing(r)+f_ns_withoutsing(r))*r**2/(1+r**2)

vec_integrand_F = np.vectorize(integrand_F)


Force_trap=np.trapz(2*np.pi*(f_s(r)+vec_f_ns(r))*r**2/(1+r**2), r)
Force = quad(integrand_F,0.9e-3,np.inf,epsabs=eps)

print('Force (bruteforce trapezoidal calculation using r sampling), Force (proper integration), error',Force_trap,Force[0],Force[1])



###########################
######### Figures #########
###########################

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.semilogx(r,f_s(r),'-',linewidth=1)
plt.semilogx(r, vec_f_ns(r),'--',linewidth=1)
plt.semilogx(r, (4*np.log(r)+2)/np.pi/(1+r**2)**2,'-.',linewidth=1)
plt.semilogx(r, -(4*np.log(r)+2)/np.pi/(1+r**2)**2,'-.',linewidth=1)
plt.xlabel('$r^*$', fontsize=24)
plt.ylabel('$f^*$', fontsize=24)
plt.savefig('fpart.eps', format='eps', dpi=400,bbox_inches = 'tight')


plt.figure()

plt.semilogx(r,f_s_withoutsing(r),'-',linewidth=1)
#plt.semilogx(r, vec_f_ns(r)-(4*np.log(r)+2)/np.pi/(1+r**2)**2,'--',linewidth=1)
plt.semilogx(r, vec_f_ns_withoutsing(r),'--',linewidth=1)
plt.xlabel('$r^*$', fontsize=24)
plt.ylabel('$f^*$', fontsize=24)
plt.savefig('fpart_without_sing.eps', format='eps', dpi=400,bbox_inches = 'tight')


shearDavis = np.loadtxt("davis_shear.csv")

plt.figure()
plt.xlabel('$r^*$', fontsize=24)
plt.ylabel('$f^*$', fontsize=24)
plt.plot(r,-f_s(r)-vec_f_ns(r),'-',linewidth=1)
plt.plot(shearDavis[:,0]/2**0.5,shearDavis[:,1]*2,'o',linewidth=1)
plt.xlim(0,4)
plt.ylim(0,2)
plt.savefig('f.eps', format='eps', dpi=400,bbox_inches = 'tight')


plt.figure()

pressureDavis = np.loadtxt("davis_pressure.csv")
plt.plot(r,pressure)
plt.plot(pressureDavis[:,0]/2**0.5,pressureDavis[:,1]*2**0.5,'o',linewidth=1)
plt.xlabel('$r^*$', fontsize=24)
plt.ylabel('$p^*$', fontsize=24)
plt.savefig('pressure.eps', format='eps', dpi=400,bbox_inches = 'tight')









