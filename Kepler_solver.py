from __future__ import division
import numpy as np

#period is related to semi-major axis T^2 / a^3 = 4pi^2 / G(M+m)

G = 6.67e-11
Msun = 1.99e30
Mearth = 5.97e24
MJupiter = 1.89e27

def km2sr(km): #coverts kilometers to solar radii
    return km/695500

def sr2AU(sr,radius=215.1): #converts stellar radii to AU.
    return sr/radius

def AU2sr(AU, radius=215.1): #converts AUs to stellar radii
    return AU*radius

def year2sec(year): #converts years to seconds
    return year*365*24*60*60

def sec2year(sec): #converts seconds to years
    return sec/(60*60*24*365)

def sec2day(sec): #converts seconds to days
    return sec/(60*60*24)

def m2AU(meters): #converts meters to AU
    return meters/1.496e11

def AU2m(AU): #converts AU to meters
    return AU*1.496e11

def redmass(mstar, mplanet):
    return (mstar*mplanet) / (mstar + mplanet)

def period(a,mstar,mplanet, fmt='seconds'): #semi-major axis in AU
    a = AU2m(a)
    seconds =  np.sqrt(( (a**3)*(4*np.pi**2))/(G*(mstar+mplanet)))
    if fmt == 'years':
        return sec2year(seconds)
    elif fmt == 'days':
        return sec2day(seconds)
    else:
        return seconds

def a(T,mstar,mplanet,fmt='AU'): #period in years, masses in kg
    T = year2sec(T)
    semimajor = (((T**2)*(G*(mstar+mplanet))) / (4*np.pi**2))**(1/3)
    #print "semi-major axis = "+str(semimajor)+" meters."
    #print "= "+str((semimajor)/(1.496e11))+" AU."
    #return semimajor
    if fmt=='meters':
        return semimajor
    else:
        return semimajor/(1.496e11)

print " "
print " "
print "* * * * * KEPLER SOLVER * * * * *"
print "Mearth is 5.97e24"
print "Msun  is 1.99e30"
print "MJupiter = 1.89e27"
print " "
#print " "
print "1.) a(period, mstar, mplanet, fmt='AU'): "
print "    #calculates semi-major axis (period in years, mass in kg). fmt='AU' or 'meters'  (default AU)."
print " "
print "2.) period(a, mstar, mplanet, fmt='seconds'): "
print "    #calculates the orbital period (a in AU, mass in kg). fmt='seconds', 'days', or 'years' (default seconds)."  
print " "
print "3.) AU2sr(AU,radius=215.1): "
print "    #converts AUs to solar radii. radius=ratio of R* to AU (default solar)."
print " "
print "4.) sr2AU(sr, radius=215.1): "
print "    #converts solar radii to AU. radius=ratio of R* to AU (default solar)."
print " "
print "5.) km2sr(km): #coverts kilometers to solar radii"
#print " "
print "6.) year2sec(year): #converts years to seconds"
#print " "
print "7.) sec2year(sec): #converts seconds to years"
#print " "
print "8.) sec2day(sec): #converts seconds to days"
#print " "
print "9.) m2AU(meters): #converts meters to AU"

print "10.) AU2m(AU): #converts AU to meters"

print "11.) redmass(mstar, mplanet): #calculates reduced mass (in kg)"
print " "


