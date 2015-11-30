from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import os


### FUNCTION DEFINITIONS
def len_file(fname):
	with open(fname) as f:
		for i,l in enumerate(f):
			pass
	return i+1

### USER SPECIFICATIONS
ndesired = float(raw_input("How many transits do you want per epoch? "))


### FILE READING
bgbz_numlines = len_file('../HEK/bigboys_sandbox.txt')
bgbzfile = open('../HEK/bigboys_sandbox.txt', mode='r')


### DIRECTORIES
S2urveydir = '../S2urvey_initial_sandbox/'







### START THE BIG LOOP -- RUNNING CALCULATIONS FOR EACH PLANET.
for i in np.arange(0,bgbz_numlines, 1): 

	print "linenumber = ", i 
	line = bgbzfile.readline()
	line_list = line.split()
	HCAname = 'HCA-'+str(line_list[0])
	print "HCAname = ", HCAname


	### STEP 1 -- read in number of transits in bigboys.txt.
	### Calculate the number of transits in each segment.

	ntransits = float(line_list[1])
	noot = ntransits #number of "out of transit" events. Just doing this in case we need it later.
	print "ntransits = ", ntransits

	nsegments = np.around(ntransits/ndesired)
	print "nsegments = ", nsegments

	if nsegments == 0:
		print "fewer transits than ndesired transits per segment."
		pass
	else:
		n_per_segment = np.around(ntransits / nsegments)
		print "n_per_segment = ", n_per_segment
	

		segment_transit_numbers = []
	
		for i in np.arange(0,nsegments,1):
			segment_transit_numbers.append(n_per_segment)
	
		while np.sum(segment_transit_numbers) != ntransits:
			if np.sum(segment_transit_numbers) > ntransits:
				segment_transit_numbers[-1] = segment_transit_numbers[-1]-1
			elif np.sum(segment_transit_numbers) < ntransits:
				segment_transit_numbers[-1] = segment_transit_numbers[-1]+1
	
		if np.sum(segment_transit_numbers) == ntransits:
			print "GOOD TO GO."
			print "segment_transit_numbers = ", segment_transit_numbers
		else:
			raise Exception('Transit per segment calculation failed.')






	### STEP 2 -- for each data point in seriesP.dat, calculate transit (epoch) number, and t_fold.
	### CHANGE FROM ORIGINAL PLAN -- do these calculations BEFORE cloning, to save on computing time.

	try:
		HCAdir = S2urveydir+str(HCAname)
		COFIAM_file = open(HCAdir+'/CoFiAM_target_info.txt', mode='r')
		COFIAM_params = COFIAM_file.readline()
		COFIAM_params = COFIAM_params.split()

		period = float(COFIAM_params[1])
		print "period = ", period

		midtime = float(COFIAM_params[0])
		tau = midtime+54833
		print "midtime = ", midtime 

		first_midtime = midtime+54833 #adjusts for the mission elapsed time.
		
		all_midtimes = []
		for ts in np.arange(0,ntransits*5,1): #assuming many transits are lost. Is this good?
			try:
				all_midtimes.append(all_midtimes[-1]+period)   ### IS THIS RIGHT?
			except:
				all_midtimes.append(first_midtime)
		all_midtimes = np.array(all_midtimes)

		len_seriesPfile = len_file(HCAdir+'/example_TTVplan/seriesP.dat')
		seriesPfile = open(HCAdir+'/example_TTVplan/seriesP.dat', mode='r')
		seriesPmodfile = open(HCAdir+'/example_TTVplan/seriesP_mod.dat', mode='w')



		#create list of fluxes and tfolds, see if you can generate a light curve now!
		tfold_list = []
		dataflux_list = []
		dataerror_list = []
		epoch_list = []

		for i in np.arange(0,len_seriesPfile, 1):
			seriesPline = seriesPfile.readline()
			seriesP_vals = seriesPline.split()
			datatime = float(seriesP_vals[0])
			dataflux = float(seriesP_vals[1])
			dataerror = float(seriesP_vals[2])
			closest_midtime = all_midtimes[np.argmin(np.abs(datatime - all_midtimes))]

			#epoch = np.around ( (datatime - closest_midtime) / period )
			epoch = np.abs(np.around ( (datatime - tau) / period))   #epoch is the assigned transit number!

			if epoch in epoch_list:
				pass
			else:
				epoch_list.append(epoch)

			tfold = datatime - tau - period*epoch

			tfold_list.append(tfold)
			dataflux_list.append(dataflux)
			dataerror_list.append(dataerror)

			if epoch > ntransits:
				raise Exception("epoch number greater than the number of planet transits.")


			seriesP_vals.append(str(epoch)) #adds the epoch number onto the end of seriesPline

			seriesPmodfile.write(str(seriesPline[:-1])+'\t'+str(epoch)+'\n')



		COFIAM_file.close()
		seriesPfile.close()
		seriesPmodfile.close()
	except:
		print "couldn't calculate all midtimes."
		pass


	### PLOTTING THE LIGHT CURVE.
	plt.scatter(tfold_list, dataflux_list, s=10, alpha=0.5)
	plt.errorbar(tfold_list, dataflux_list, yerr=dataerror_list, fmt="none", alpha=0.5)
	plt.grid()
	plt.xlim(np.amin(tfold_list), np.amax(tfold_list))
	plt.ylim(np.amin(dataflux_list)-0.001, 1.001)
	plt.title(HCAname)
	plt.xlabel('Time From Mid-Transit')
	plt.ylabel('Relative Flux')
	plt.show()




	### STEP 3 -- RECALCULATE THE NUMBER OF SEGMENTS USING THE EPOCHS YOU'VE CALCULATED!!!!
	ntransits_old = ntransits
	nsegments_old = nsegments 

	ntransits = len(epoch_list)
	noot = ntransits #number of "out of transit" events. Just doing this in case we need it later.
	print "ntransits = ", ntransits

	nsegments = np.around(ntransits/ndesired)
	print "nsegments = ", nsegments

	if nsegments == 0:
		print "fewer transits than ndesired transits per segment."
		pass
	else:
		n_per_segment = np.around(ntransits / nsegments)
		print "n_per_segment = ", n_per_segment
	

		segment_transit_numbers = []
		for i in np.arange(0,nsegments,1):
			segment_transit_numbers.append(n_per_segment)
	
		while np.sum(segment_transit_numbers) != ntransits:
			if np.sum(segment_transit_numbers) > ntransits:
				segment_transit_numbers[-1] = segment_transit_numbers[-1]-1
			elif np.sum(segment_transit_numbers) < ntransits:
				segment_transit_numbers[-1] = segment_transit_numbers[-1]+1
	
		if np.sum(segment_transit_numbers) == ntransits:
			print "GOOD TO GO."
			print "segment_transit_numbers = ", segment_transit_numbers
		else:
			raise Exception('Transit per segment calculation failed.')




	### STEP 4 --- Sort the epochs into segments
	epoch_groups = []
	segment_startstop = [0]
	for i in np.cumsum(segment_transit_numbers):
		segment_startstop.append(i)

	for n in np.arange(0,len(segment_startstop), 1):
		try:
			segment_start = segment_startstop[int(n)]
			segment_stop = segment_startstop[int(n+1)]
			print "segment_start = ", segment_start
			print "segment_stop = ", segment_stop
			temp_epoch_list = epoch_list[int(segment_start):int(segment_stop)]
			epoch_groups.append(temp_epoch_list)
		except:
			pass
	print "epoch_groups = ", epoch_groups






	### STEP 5 --- ADD SEGMENT NUMBER TO THE seriesP_mod.dat file lines
	len_seriesPmodfile = len_file(HCAdir+'/example_TTVplan/seriesP_mod.dat')
	seriesPmodfile = open(HCAdir+'/example_TTVplan/seriesP_mod.dat', mode='r')
	seriesPmodmodfile = open(HCAdir+'/example_TTVplan/seriesP_modmod.dat', mode='w')

	for i in np.arange(0, len_seriesPmodfile, 1):
		seriesPmodfileline = seriesPmodfile.readline()
		seriesPmodfilelist = seriesPmodfileline.split()
		seriesPmodepoch = seriesPmodfilelist[-1]
		for n in np.arange(0,len(epoch_groups),1):
			if float(seriesPmodepoch) in epoch_groups[n]:
				#print "found the segment number!"
				seriesPsegment = n
		#seriesPmodfilelist.append(seriesPsegment)
		seriesPmodmodfile.write(str(seriesPmodfileline[:-1])+'\t'+str(seriesPsegment)+'\n')
	seriesPmodfile.close()
	seriesPmodmodfile.close()



	### STEP 6 -- CREATE THE CLONE FILES.
	example_subdirs = []
	for ns in np.arange(0,nsegments,1):
		Segplandir = HCAdir+'/exampleSEGplan_'+str(int(ns))
		os.system('cp -rf '+str(HCAdir)+'/example_TTVplan '+str(HCAdir)+'/exampleSEGplan_'+str(int(ns)))
		example_subdirs.append(HCAdir+'/exampleSEGplan_'+str(int(ns)))

	
		### eliminate the transits in seriesPmodmod.dat that aren't right for this segment
		Segplan_seriesP = open(HCAdir+'/exampleSEGplan_'+str(int(ns))+'/seriesP_modmod.dat', mode='r')
		Segplan_seriesPmod = open(HCAdir+'/exampleSEGplan_'+str(int(ns))+'/seriesP_segmod.dat', mode='w')
		for i in np.arange(0, len_seriesPmodfile, 1):
			Segplan_seriesPline = Segplan_seriesP.readline()
			Segplan_seriesPlist = Segplan_seriesPline.split()
			if int(Segplan_seriesPlist[-1]) == int(ns):
				Segplan_seriesPmod.write(Segplan_seriesPline)
			else:
				pass
		Segplan_seriesP.close()
		Segplan_seriesPmod.close()




		### STEP 7 --- remove all the superfluous intermediate seriesP files
		os.system('mv '+Segplandir+'/seriesP_segmod.dat '+Segplandir+'/seriesP.dat')
		os.system('rm -rf '+Segplandir+'/seriesP_modmod.dat')
		os.system('rm -rf '+Segplandir+'/seriesP_mod.dat')
		os.system('rm -rf '+Segplandir+'/seriesP_segmod.dat')










	### KEEP THIS AT THE VERY END OF THE FOR LOOP.
	print " "
	print " "







bgbzfile.close()
