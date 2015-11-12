from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

#posterior plotter.


os.system('ls')

p1 = np.load(raw_input('First planet file? '))
p3 = np.load(raw_input('Third planet file? '))
p4 = np.load(raw_input('Fourth planet file? '))
allps = np.load(raw_input('All planets file? '))

for i in [0,1,2]:
	n1, bins1, patches1 = plt.hist(p1.T[i], bins=50, color='r', histtype='step', label='Planet 1')
	n3, bins3, patches3 = plt.hist(p3.T[i], bins=50, color='b', histtype='step', label='Planet 3')
	n4, bins4, patches4 = plt.hist(p4.T[i], bins=50, color='g', histtype='step', label='Planet 4')
	na, binsa, patchesa = plt.hist(allps.T[i], bins=50, color='k', histtype='step', label='All Three Planets')

	plt.legend()
	if i == 0:
		plt.xlabel('Log(Stellar Density)')
		plt.title('Stellar Density Posterior (No Burn-In)')
	elif i == 1:
		plt.xlabel('q1')
		plt.title('q1 Posterior (No Burn-In)')
	elif i == 2:
		plt.xlabel('q2')
		plt.title('q2 Posterior (No Burn-In)')
	
	plt.show()