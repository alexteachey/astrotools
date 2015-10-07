from __future__ import division
import batman
import numpy as np
import matplotlib.pyplot as plt



#THIS CODE MAKES USE OF THE BATMAN TRANSIT MODEL CODE DEVELOPED BY LAURA KREIDBERG (UChicago). 
#DOCUMENTATION FOUND AT http://astro.uchicago.edu/~kreidberg/batman/index.html


#user specified parameters
user_spec = raw_input('Do you want to specify your own parameters? y/n: ')
if user_spec == 'y':
        try:
                execfile('Kepler_solver.py')
        except:
                pass

        timewindow = float(raw_input("How long before and after inferior conjunction do you want to model? (in hours): "))
        timewindow_frac = timewindow/24 #converts hours to fractions of a day.
        timesteps = timewindow*2*60 #creates one minute timesteps for the user specified time window.
        
	IC = float(raw_input('what is the time of inferior conjunction?: '))
	OP = float(raw_input('What is the orbital period in days? '))
        try:
                OP_years = OP/365
                SMA =a(OP_years, Msun, Mearth) #calculates semi-major axis in AU from the period
                SMA = AU2sr(SMA) #calculates semi-major axis in solar radii
                print "computed semi-major axis in stellar radii based M_Sun, M_Earth, and R_Sun." 
                print " "
        except:
                SMA = float(raw_input("What is the semi-major axis in units of stellar radii? "))
	PR = float(raw_input('What is the planet radius in units of stellar radii? '))
	OI = float(raw_input('What is the orbital inclination in degrees? '))
	ECC = float(raw_input('What is the orbital eccentricity? (0 - 1): '))
	PERI = float(raw_input('What is the longitude of periastron in degrees? (90 for a circle): '))
        print " "
        print "Limb darkening models are 'uniform', 'linear', 'quadratic', 'square-root', 'logarithmic', 'exponential', or 'nonlinear'."
        LDF = raw_input("What is the limb darkening model? ")
        print " "
        LDC = []
        if LDF == 'linear':
                LDC1 = float(raw_input("What is the limb darkening coefficient? "))
                LDC.append(LDC1)
        elif LDF == 'quadratic' or LDF == 'square-root' or LDF == 'logarithmic' or LDF == 'exponential':
                LDC1 = float(raw_input('What is the first limb darkening coefficient? '))
                LDC.append(LDC1)
                LDC2 = float(raw_input('What is the second limb darkening coefficient? '))
                LDC.append(LDC2)
        elif LDF == 'nonlinear':
                LDC1 = float(raw_input('What is the first limb darkening coefficient? '))
                LDC.append(LDC1)
                LDC2 = float(raw_input('What is the second limb darkening coefficient? '))
                LDC.append(LDC2)
                LDC3 = float(raw_input('What is the third limb darkening coefficient? '))
                LDC.append(LDC3)
                LDC4 = float(raw_input("What is the fourth limb darkening coefficient? "))
                LDC.append(LDC4)
        else:
                LDF = 'quadratic'
                LDC = [0.1, 0.3]
                print "Failed to select one of the options. Limb darkening will be quadratic with [c1,c2] = [0.1, 0.3]"

                

else:
        print "Going with the Hot Jupiter model (T=11.55 days, a = 0.1 AU).. See code for details. "
	IC = 0 #inferior conjunction
        OP = 11.55 #orbital period of a planet orbiting the Sun at 0.1 AU
	PR = 0.1 #planet radius in stellar radii
        SMA = 21.51 #semi-major axis of Earth in stellar radii
	OI = 90 #orbital inclination
	ECC = 0 #eccentricity
	PERI = 90 #longitude of periastron
	LDC = [0.1, 0.3] 
	LDF = 'quadratic'

        timewindow = 12
        timewindow_frac = timewindow/24 #converts hours to fractions of a day.
        timesteps = timewindow*2*60 #creates one minute timesteps for the user specified time window.


#specify parameters
params = batman.TransitParams()
params.t0 = IC                       #time of inferior conjunction
params.per = OP                      #orbital period
params.rp = PR                      #planet radius (in units of stellar radii)
params.a = SMA                       #semi-major axis (in units of stellar radii)
params.inc = OI                     #orbital inclination (in degrees)
params.ecc = ECC                      #eccentricity
params.w = PERI                       #longitude of periastron (in degrees)
params.u = LDC                #limb darkening coefficients
params.limb_dark = LDF       #limb darkening model



onemin = (0.5/720) #0.5 is half a day or 12 hours. There are 720 minutes in that time.
fifteenmin = onemin*15



#PART 1: generate "true" transit model (approximately instantaneous fluxes).
t = np.linspace(-timewindow_frac, timewindow_frac, timesteps)
m = batman.TransitModel(params, t)    #initializes model
flux = m.light_curve(params)          #calculates light curve



#PART 2: simulating long cadence data. makes every step a 30 minute exposure (averaging over half an hour).
binned_fluxes = []
for i in t:
        newfluxes = flux[np.where((t>=i-fifteenmin) & (t<= i+fifteenmin))]
        binned_fluxes.append(np.mean(newfluxes))



#PART 3: resampling the long cadence data. cut every "long cadence" data point up into 30 little pieces:
exploded_fluxes = []
exploded_times = []
for i in t:   #for each time step
        tempt = np.arange(i-fifteenmin, i+fifteenmin, onemin) #create 30 1-minute time steps.
        tempm = batman.TransitModel(params, tempt) #initialize the model for these times.
        tempflux = tempm.light_curve(params) #calculate the model fluxes for these times.
        for f,ti in zip(tempflux, tempt):
                exploded_fluxes.append(f)
                exploded_times.append(ti)

#NOW THESE HAVE MANY MORE DATA POINTS THAN YOUR ORIGINAL SAMPLE.
exploded_times = np.array(exploded_times)
exploded_fluxes = np.array(exploded_fluxes)



#PART 4: Re-bin each data point back to the original timesteps (30 minute integration times).
rebinned_fluxes = []
for i in t:      #along the original time array
        tempidx = np.where((exploded_times >=i-fifteenmin) & (exploded_times <= i+fifteenmin)) #grabbing all the indices around the data point, in the long cadence.
        tempidx = tempidx[0]
        tempfluxes = exploded_fluxes[tempidx]
        rebinned_fluxes.append(np.mean(tempfluxes)) #taking the mean of all these new fluxes.




#plot figure
plt.plot(t, flux, c='b', label='short cadence (true)', linewidth = 4, alpha=0.6)
plt.plot(t, binned_fluxes, c='r', label='simulated long cadence', linewidth = 7, alpha=0.6)
plt.plot(t, rebinned_fluxes, c='k', label='resampled fluxes', linewidth=1)
plt.errorbar(t[np.argmin(rebinned_fluxes)],np.amin(rebinned_fluxes)+0.005, c='k',xerr=fifteenmin, label='long cadence') #plots the size of the 30-minute cadence.
plt.xlabel("Time from central transit (days)") 
plt.ylabel("Relative flux") 
plt.grid() 
plt.legend()
plt.ylim(np.amin(flux)-0.05, np.amax(flux)+0.05) 
try:
        execfile('Kepler_solver.py')
        SMA_AU = sr2AU(SMA)
        plt.title('Hot Jupiter (T = '+str(OP)+'days, a = '+str(round(SMA_AU,3))+' AU)')
except:
        plt.title('Hot Jupiter (T = '+str(OP)+'days, a = '+str(round(SMA,3))+' solar radii)')
plt.show()
