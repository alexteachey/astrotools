from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab as P
import matplotlib.cm as cm
try:
    import corner
except:
    pass
import os
from mpl_toolkits.mplot3d import Axes3D


ln = np.log

max_it = 1e3

xvals = []
yvals = []
noisy_yvals = []
sigmas = []
noise_additions = []
function_value_matrix = []
loglike_list = []
loglike_full_list = []
reject_list = []
random_draw_list = []
color_list = []

loglike_list_post_burn = []
index_post_burn = []

#param_number = [] 
param_number = 3 #trying this to see if the issue is resolved -- November 9, 2015
param_options = []
exponents = []



#### FUNCTION DEFINITIONS ####


def file_len(fname):
    with open(fname) as f:
        for i,l in enumerate(f):
            pass
    return i+1



filelength = file_len('../Data/KOI490_Archive/planet1.dat') #should be more dynamic.

def mad(dataset):
    datamedian = np.median(dataset)
    dataset = np.array(dataset)
    datamedian = np.array(datamedian)
    deviation = np.abs(datamedian-dataset)
    return np.median(deviation)




def read_data(txtfile, planetnum):
    #LISTS
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []
    #col9 = []



    #data = open('../Data/chains/'+str(txtfile), 'r')
    data = open('../Data/KOI490_Archive/'+str(txtfile), 'r')
    for i in np.arange(0,filelength,1): #not flexible right now
        linedata = data.readline().split()


        ### THIS REFLECTS THE NEW DATA CHAINS DAVID SENT ME - NOVEMBER 4TH
        try:
            col1.append(float(linedata[0]))
            col2.append(float(linedata[1]))
            col3.append(float(linedata[2]))
            col4.append(float(linedata[3]))
            col5.append(float(linedata[4]))
            col6.append(float(linedata[5]))
            col7.append(float(linedata[6]))
            col8.append(float(linedata[7]))
            #col9.append(float(linedata[8]))
        except:
            pass

    col1 = np.array(col1)
    col2 = np.array(col2)
    col3 = np.array(col3)
    col4 = np.array(col2)
    col5 = np.array(col5)
    col6 = np.array(col6)
    col7 = np.array(col7)
    col8 = np.array(col8) 
    #col9 = np.array(col9)

    #cols = [col1,col2,col3,col4,col5,col6,col7,col8,col9]
    cols = [col1, col2, col3, col4, col5, col6, col7, col8]
    #colnames = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']
    colnames = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']
    for c,cn in zip(cols, colnames):
        try:
            column_dictionary[cn] = c
        except:
            column_dictionary = {cn : c}
    return column_dictionary




def X2(data, model, error): #args should be arrays
    return np.sum((data - model)**2 / (error)**2)

    
uber_column_dictionary = []
    


numfiles = 3 #could make dynamic -- this is for planet 1, 3 and 4.
datatype = raw_input("Do you want to generate 'f'ake data, or use 'r'eal data? ")
if datatype == 'r':
    print " "
    print " "
    os.system('ls ../Data/KOI490_Archive/')
    print " "
    print " "
    for n in [1,3,4]:
        yourfile = raw_input('What is the name of the file you want to load? ')
        uber_column_dictionary.append(read_data(yourfile, n))
    
else:
    make_data()



print "uber column dictionary = ", uber_column_dictionary
uber_column_dictionary = np.array(uber_column_dictionary)



#use col8, fit col2, col6, col7
start_col2 = np.median([np.median(uber_column_dictionary[0]['col2']), np.median(uber_column_dictionary[1]['col2']), np.median(uber_column_dictionary[2]['col2'])])
start_col6 = np.median([np.median(uber_column_dictionary[0]['col6']), np.median(uber_column_dictionary[1]['col6']), np.median(uber_column_dictionary[2]['col6'])])
start_col7 = np.median([np.median(uber_column_dictionary[0]['col7']), np.median(uber_column_dictionary[1]['col7']), np.median(uber_column_dictionary[2]['col7'])])

jumpsize_col2 = 1.4 * np.amin([mad(uber_column_dictionary[0]['col2']), mad(uber_column_dictionary[1]['col2']), mad(uber_column_dictionary[2]['col2'])])
#jumpsize_col6 = 1.4 * np.amin([mad(uber_column_dictionary[0]['col6']), mad(uber_column_dictionary[1]['col6']), mad(uber_column_dictionary[2]['col6'])])
#jumpsize_col7 = 1.4 * np.amin([mad(uber_column_dictionary[0]['col7']), mad(uber_column_dictionary[1]['col7']), mad(uber_column_dictionary[2]['col7'])])

jumpsize_col2 = jumpsize_col2/10
jumpsize_col6 = 0.01
jumpsize_col7 = 0.01

#trying this scaling november 10th 2015
#jumpsize_col2 = jumpsize_col2/100
#jumpsize_col6 = jumpsize_col6/100
#jumpsize_col7 = jumpsize_col7/100


print "jumpsize_col2 = ", jumpsize_col2
print "jumpsize_col6 = ", jumpsize_col6
print "jumpsize_col7 = ", jumpsize_col7







# # # # # MCMC CODE # # # # # 
burning = True
for i in np.arange(0,max_it,1): #steps
    if i >=1:
        burning = False
    try:
        gocode = 1
        print " "
        print " "
        print "i = ", i
        print "burning = ", burning
        
        if i == 0:
            ### INITIAL RUN.
            color_list.append(i/max_it)
            if datatype == 'r':
                start_params = np.array([start_col2, start_col6, start_col7])
                print "start_params (start_params) = ", start_params
            else:
                start_params = np.random.choice(param_options, size=param_number)
            print "start_params = ", start_params
            #if datatype == 'f':
                #print "true_params = ", true_params

            param_guess_matrix = start_params
            param_full_matrix = start_params



            ### NEW FUNCTIONALITY TO BRING IN THE ND INTERPOLATOR:
            #execfile("NDmcmc.py") #change that code to make it work better for this one.
            execfile("NDmcmc_noprint.py")
            intmode = 'mcmc'
            test_type = 'real'
            loglikes = []
            for p in np.arange(0,uber_column_dictionary.shape[0],1): #for each of the planets (3 in your case)
                testpoint = start_params
                print "testpoint = ", testpoint
                loglikes.append(interp(testpoint, uber_column_dictionary[p])) ### NEED TO WRITE THE NDint code to handle calling from the uber_column_dictionary

            total_loglike = np.sum(loglikes)
            print "total_loglike = ", total_loglike

            function_1st_guess = total_loglike #INITIAL FUNCTION VALUE.


            try:
                if norm_yvals == 'y':
                    function_1st_guess = function_1st_guess/np.amax(function_1st_guess)
            except:
                pass
            


            new_loglike = total_loglike #trying this November 9, 2015



            loglike_list.append( new_loglike  )


            if burning == False:
                loglike_list_post_burn.append(new_loglike)
                try:
                    index_post_burn.append(index_post_burn[-1]+1)
                except:
                    index_post_burn.append(0)

            loglike_full_list.append(new_loglike)
            
            function_value_matrix = function_1st_guess #APPENDING THE FUNCTION VALUE.
                      



            
        else: # ALL SUBSEQUENT RUNS.
            if gocode == 1:
                if i == 1:
                    start2 = np.random.normal(loc=param_guess_matrix[0], scale=jumpsize_col2) #trying this November 10 2015
                    start6 = np.random.normal(loc=param_guess_matrix[1], scale=jumpsize_col6)
                    start7 = np.random.normal(loc=param_guess_matrix[2], scale=jumpsize_col7)
                    start_params = np.array([start2, start6, start7])
                    print "start_params = ", start_params 
                    print "param_guess_matrix shape = ", param_guess_matrix.shape


                else:
                    if param_guess_matrix.shape == (param_number,):
                        start2 = np.random.normal(loc=param_guess_matrix[0], scale=jumpsize_col2)
                        start6 = np.random.normal(loc=param_guess_matrix[1], scale=jumpsize_col6)
                        start7 = np.random.normal(loc=param_guess_matrix[2], scale=jumpsize_col7)
                        start_params = np.array([start2, start6, start7])
                        print "start_params = ", start_params    

                    else:
                        start2 = np.random.normal(loc=param_guess_matrix[-1][0], scale=jumpsize_col2)
                        start6 = np.random.normal(loc=param_guess_matrix[-1][1], scale=jumpsize_col6)
                        start7 = np.random.normal(loc=param_guess_matrix[-1][2], scale=jumpsize_col7)
                        start_params = np.array([start2, start6, start7])
                        print "start_params = ", start_params

                            
                    print "param_guess_matrix shape = ", param_guess_matrix.shape

                """
                try:
                    print "true_params = ", true_params
                except:
                    pass
                """


            else: #if gocode == 0:
                if i == 1:
                    start_params = param_guess_matrix
                else:
                    start_params = param_guess_matrix[-1]
                gocode = 1
                
                
            param_full_matrix = np.vstack((param_full_matrix, start_params))
        
            #NEW ITERATION CALCULATION

            #execfile("NDmcmc.py") #change that code to make it work better for this one.
            execfile("NDmcmc_noprint.py")
            intmode = 'mcmc'
            test_type = 'real'

            loglikes = []
            for p in np.arange(0,uber_column_dictionary.shape[0],1):
                testpoint = start_params
                loglikes.append(interp(testpoint, uber_column_dictionary[p])) ### NEED TO WRITE THE NDint code to handle calling from the uber_column_dictionary
            
            total_loglike = np.sum(loglikes)
            print "total_loglike = ", total_loglike

            try:
                print "maximum loglike so far = ", np.amax(loglike_full_list)
                print "percent of max (this iteration) = ", total_loglike/np.amax(loglike_full_list)
            except:
                pass

            y_guess = total_loglike


            
            function_value_matrix = np.vstack((function_value_matrix, np.array(y_guess))) 
            new_loglike = total_loglike


            """if datatype == 'm': #what is this?
                new_loglike = total_loglike
            else:
                new_loglike = total_loglike
            """


            loglike_full_list.append(new_loglike)
            
            #burning = False  ### NEEDS TO CHANGE!

            
            
            
            if new_loglike > loglike_list[-1]:    ### THIS MARKS AN IMPROVEMENT
                print "accepted."
                print "new_loglike = ", new_loglike
                print " "
                color_list.append(i/max_it) 
                try:
                    param_guess_matrix = np.vstack((param_guess_matrix, start_params))
                except:
                    pass

                loglike_list.append(new_loglike)

                if burning == False:
                    loglike_list_post_burn.append(new_loglike)

                    try:
                        index_post_burn.append(index_post_burn[-1]+1)
                    except:
                        index_post_burn.append(0)
                    
                    try:
                        param_guess_post_burn = np.vstack((param_guess_post_burn, start_params))
                    except:
                        param_guess_post_burn = start_params
            

                
                                              
            else:  ### IF loglike isn't BETTER.
                normloglike = new_loglike/np.amax(loglike_list)
                normlastloglike = loglike_list[-1]/np.amax(loglike_list)



                ### THIS IS PROBLEMATIC... THE FIRST WAY BELOW ALWAYS RESULTS IN PROB = 0. THE SECOND WAS IS ALWAYS PROB = 0.999.

                #prob = np.exp(-((loglike_list[-1] - new_loglike)**2)/2) # THIS MAY NEED TO CHANGE.
                #prob = np.exp(-((normlastloglike - normloglike)**2)/2) #TRYING THIS NOVEMBER 9 2015
                #prob = np.exp( -( (1 / ((new_loglike)/loglike_list[-1]) ) ) )
                prob = np.exp(0.5 * (new_loglike - loglike_list[-1])) #DAVID'S FORMULATION -- NOVEMBER 10th 2015
                print 'prob = ', prob 



                random_draw = np.random.normal(loc=0.5, scale=0.1)
                print "random_draw = ", random_draw

                if prob > random_draw: # IF PROB GIVES YOU A SHOT
                    color_list.append(i/max_it)
                    try:
                        param_guess_matrix = np.vstack((param_guess_matrix, start_params))
                    except:
                        print "param_guess_matrix not stackable."
                        raise Exception('param_guess_matrix not stackable.')
                        #param_guess_matrix = start_params


                    loglike_list.append(new_loglike)


                    if burning == False:
                        loglike_list_post_burn.append(new_loglike)
                        try:
                            index_post_burn.append(index_post_burn[-1]+1)
                        except:
                            index_post_burn.append(0)
                        
                        try:
                            param_guess_post_burn = np.vstack((param_guess_post_burn, start_params))
                        except:
                            print "param_guess_matrix not stackable."
                            #raise Exception('param_guess_matrix not stackable.')
                            param_guess_post_burn = start_params

                    
                    print "loglike worse, prob better than random normal draw."
                    print "prob = ", prob
                    print "random draw = ", random_draw
                    print " "
                    random_draw_list.append(random_draw)                  


                    
                else:  ## IF PROB DOESN"T WORK.
                    reject_list.append(i)
                    print "REJECTED. loglike worse, prob worse than random normal draw."
                    print "prob = ", prob
                    print "random draw = ", random_draw
                    random_draw_list.append(random_draw)
                    gocode = 0
                    print "gocode = 0"
                    print " "
                    #continue
                    
                    
        print "new_loglike = ", new_loglike
        print " "
        print " "
    
    except KeyError:
        print "Couldn't find a necessary coordinate for the interpolation."
        continue
    
    
    
loglike_list_post_burn = np.array(loglike_list_post_burn)
index_post_burn = np.array(index_post_burn).astype(int)
    

print "len(loglike_list) =", len(loglike_list)
print "len(loglike_list_post_burn) = ", len(loglike_list_post_burn)
print "np.mean(loglike_list) = ", np.mean(loglike_list)
print "np.amax(loglike_list) = ", np.amax(loglike_list)
print "np.amin(loglike_list) = ", np.amin(loglike_list) 
    
    
    
    
    
    
### PLOTTING #### 

## RANDOM DRAW HISTOGRAM -- CENTERED ON 0.5
#n,bins,patches = P.hist(random_draw_list, 50, histtype='bar')
#P.show()


### PLOT THE DATA AND THE BEST FIT
"""
plt.plot(xvals, function_1st_guess, c='r', label='First Guess') ### 1ST GUESS
if datatype == 'f':
    plt.plot(xvals, yvals, c='b', label='True Solution', linewidth=3, alpha=0.5)  ### THE REAL ANSWER
plt.scatter(xvals, noisy_yvals, c='b', s=10)  ### THE NOISY DATA
plt.errorbar(xvals, noisy_yvals, yerr=sigmas, c='b', fmt=None)
plt.plot(xvals, function_value_matrix[np.argmin(loglike_full_list)], c='k', label='Best Guess', linewidth=2)
plt.title('MCMC Sandbox')
plt.legend()
plt.grid()
#plt.savefig('MCMC_sandbox.pdf', dpi=100, clobber=True)
plt.show()
"""


### RANDOM WALK VISUALIZATION
plt.scatter(param_guess_matrix.T[param_number-2], param_guess_matrix.T[param_number-1], s=20, c=color_list, alpha=0.2, label='Random Walk')
plt.scatter(param_guess_matrix[0][param_number-2], param_guess_matrix[1][param_number-1], s=80, label='Start Value', c='g', marker='s')
try:
    plt.scatter(true_params[param_number-2], true_params[param_number-1], label='True Value', c='r', marker='*', s=200)
except:
    pass
plt.scatter(param_guess_matrix.T[param_number-2][np.argmax(loglike_list)], param_guess_matrix.T[param_number-1][np.argmax(loglike_list)], s=200, label='Maximum Likelihood', c='g', marker='v')
plt.legend(scatterpoints=1, loc=0)
plt.grid()
plt.title('Random Walk')
#plt.savefig('MCMC_random_walk.pdf', dpi=100, clobber=True)
plt.show()



### TRYING 3D RANDOM WALK VISUALIZATION
"""
try:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(param_guess_matrix.T[param_number-2], param_guess_matrix.T[param_number-1], zs=param_guess_matrix.T[param_number], s=20, c=color_list, alpha=0.2)
    ax.scatter(param_guess_matrix[0][param_number-2], param_guess_matrix[1][param_number-1], zs=param_guess_matrix[2][param_number], s=80, c='g', marker='s')
    ax.scatter(param_guess_matrix.T[param_number-2][np.argmax(loglike_list)], param_guess_matrix.T[param_number-1][np.argmax(loglike_list)], zs=param_guess_matrix.T[param_number][np.argmax(loglike_list)], s=200, c='g', marker='v')
    ax.title('Random Walk')
    ax.show()
except:
    pass
"""

### RANDOM WALK VISUALIZATION (POST BURN)
"""
post_burn_color_list = index_post_burn/len(index_post_burn)
plt.scatter(param_guess_matrix.T[param_number-2][index_post_burn], param_guess_matrix.T[param_number-1][index_post_burn], s=20, c=post_burn_color_list, alpha=0.2, label='Random Walk')
plt.scatter(param_guess_matrix[0][param_number-2], param_guess_matrix[1][param_number-1], s=80, label='Start Value', c='g', marker='s')
try:
    plt.scatter(true_params[param_number-2], true_params[param_number-1], label='True Value', c='r', marker='*', s=200)
except:
    pass
plt.scatter(param_guess_matrix.T[param_number-2][np.argmin(loglike_list)], param_guess_matrix.T[param_number-1][np.argmin(loglike_list)], s=200, label='Final Value', c='g', marker='v')
plt.legend(scatterpoints=1, loc=0)
plt.grid()
#plt.savefig('MCMC_random_walk.pdf', dpi=100, clobber=True)
plt.title("Random Walk (Post Burn)")
plt.show()
"""



##PLOT loglike_list
plt.plot(np.arange(0,len(loglike_list), 1), loglike_list)
#plt.ylim(0,np.amax(loglike_list))
plt.title('log(likelihood) Values vs Accepted Iteration')
plt.grid()
#plt.savefig('MCMC_X2.pdf', dpi=100)
plt.show()


###PLOT loglike_full_list
plt.plot(np.arange(0,len(loglike_full_list), 1), loglike_full_list)
plt.title('All log(likelihood) values')
plt.grid()
plt.show()


#posterior distributions
"""
for param in np.arange(0,param_number,1):
    postburn =int( max_it/5)
    n, bins, patches = P.hist(param_full_matrix.T[param][postburn:], 50)
    P.ylim(0,np.amax(n)+100)
    stddev = np.std(param_full_matrix.T[param][postburn:])
    median = np.median(param_full_matrix.T[param][postburn:])
    P.plot(np.linspace(median, median, 100), np.linspace(0,np.amax(n)+100,100), c='r', label='Median')
    P.plot(np.linspace(param_full_matrix.T[param][np.argmax(loglike_full_list)], param_full_matrix.T[param][np.argmax(loglike_full_list)], 100), np.linspace(0,np.amax(n)+100, 100), c='b', label='Maximum Likelihood')
    P.plot(np.linspace(median-stddev, median-stddev,100), np.linspace(0,np.amax(n)+100,100), c='g', label=r'1 $\sigma$')
    P.plot(np.linspace(median+stddev, median+stddev,100), np.linspace(0,np.amax(n)+100,100), c='g')
    P.legend()
    if param == 0:
        P.title('Stellar Density, median = '+str(median))
    elif param == 1:
        P.title('q1, median = '+str(median))
    elif param == 2:
        P.title('q2, median = '+str(median))
    #P.title('Coefficient '+str(param)+' = '+str(median))
    P.grid()
    P.show()
"""

### USING ONLY ACCEPTED TRIALS -- FORGET ABOUT THE POSTBURN.
for param in np.arange(0,param_number,1):
    postburn =int( max_it/10)
    n, bins, patches = P.hist(param_guess_matrix.T[param][postburn:], 50)
    P.ylim(0,np.amax(n)+100)
    stddev = np.std(param_guess_matrix.T[param][postburn:])
    median = np.median(param_guess_matrix.T[param][postburn:])
    P.plot(np.linspace(median, median, 100), np.linspace(0,np.amax(n)+100,100), c='r', label='Median')
    P.plot(np.linspace(param_guess_matrix.T[param][np.argmax(loglike_list)], param_guess_matrix.T[param][np.argmax(loglike_list)], 100), np.linspace(0,np.amax(n)+100, 100), c='b', label='Maximum Likelihood')
    P.plot(np.linspace(median-stddev, median-stddev,100), np.linspace(0,np.amax(n)+100,100), c='g', label=r'1 $\sigma$')
    P.plot(np.linspace(median+stddev, median+stddev,100), np.linspace(0,np.amax(n)+100,100), c='g')
    P.legend()
    if param == 0:
        P.title('Stellar Density, median = '+str(median))
    elif param == 1:
        P.title('q1, median = '+str(median))
    elif param == 2:
        P.title('q2, median = '+str(median))
    #P.title('Coefficient '+str(param)+' = '+str(median))
    P.grid()
    P.show()



#samples = np.vstack((m_list[300:], b_list[300:], c_list[300:]))
X2_array = np.array(loglike_list)
Reasonable_X2_idx = np.where(X2_array < 30)


## DON'T HAVE CORNER LOADED ON LINUX.
"""
try:
    samples = np.vstack((param_guess_matrix.T[param_number-1][Reasonable_X2_idx], param_guess_matrix.T[param_number-2][Reasonable_X2_idx], param_guess_matrix.T[param_number-3][Reasonable_X2_idx]))
except:
    samples = np.vstack((param_guess_matrix.T[param_number-1], param_guess_matrix.T[param_number-2], param_guess_matrix.T[param_number-3]))
samples = samples.T
figure = corner.corner(samples, labels = ['a', 'b', 'c'])
figure.savefig('MCMC_corner.png', clobber=True)
figure.show()
"""


### CALCULATE BIC
print " "
print " "
best_loglike = np.amax(loglike_list)
best_params = param_guess_matrix[np.argmax(loglike_list)]


#Likelihood = best_loglike + len(xvals)*np.log(param_number) # WHAT DAVID TOLD ME TO DO
#Likelihood = best_loglike + param_number*np.log(len(xvals)) # reversing the meanings of k and N.

#NEW BIC FORMULA FROM DAVID:
BIC = best_loglike + param_number*(np.log(len(xvals)))

#BIC = -2 * np.log(Likelihood) + param_number*np.log(len(xvals))  ### VERY SUSPICIOUS ABOUT THIS.

#print "Likelihood = ", Likelihood
print "BIC = ", BIC
print "best log(likelihood) = ", best_loglike
print "parameters are = ", best_params

