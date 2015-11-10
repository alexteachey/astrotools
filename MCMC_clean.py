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
    col9 = []

    data = open('../Data/chains/'+str(txtfile), 'r')
    for i in np.arange(0,31566,1): #not flexible right now
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
            col9.append(float(linedata[8]))
        except:
            pass

    col1 = np.array(col1)
    col2 = -0.5*np.array(col2) #MAKE SURE YOU"RE ONLY DOING THIS ONCE!
    col3 = np.array(col3)
    col4 = np.array(col4)
    col5 = np.array(col5)
    col6 = np.array(col6)
    col7 = np.array(col7)
    col8 = np.array(col8)
    col9 = np.array(col9)

    cols = [col1,col2,col3,col4,col5,col6,col7,col8,col9]
    colnames = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']
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
    os.system('ls ../Data/chains/')
    print " "
    print " "
    for n in [1,3,4]:
        yourfile = raw_input('What is the name of the file you want to load? ')
        uber_column_dictionary.append(read_data(yourfile, n))
    
else:
    make_data()



print "uber column dictionary = ", uber_column_dictionary
uber_column_dictionary = np.array(uber_column_dictionary)



#use -0.5*col2, fit col4, col8, col9
start_col4 = np.median([np.median(uber_column_dictionary[0]['col4']), np.median(uber_column_dictionary[1]['col4']), np.median(uber_column_dictionary[2]['col4'])])
start_col8 = np.median([np.median(uber_column_dictionary[0]['col8']), np.median(uber_column_dictionary[1]['col8']), np.median(uber_column_dictionary[2]['col8'])])
start_col9 = np.median([np.median(uber_column_dictionary[0]['col9']), np.median(uber_column_dictionary[1]['col9']), np.median(uber_column_dictionary[2]['col9'])])

jumpsize_col4 = 1.4 * np.amin([mad(uber_column_dictionary[0]['col4']), mad(uber_column_dictionary[1]['col4']), mad(uber_column_dictionary[2]['col4'])])
jumpsize_col8 = 1.4 * np.amin([mad(uber_column_dictionary[0]['col8']), mad(uber_column_dictionary[1]['col8']), mad(uber_column_dictionary[2]['col8'])])
jumpsize_col9 = 1.4 * np.amin([mad(uber_column_dictionary[0]['col9']), mad(uber_column_dictionary[1]['col9']), mad(uber_column_dictionary[2]['col9'])])







# # # # # MCMC CODE # # # # # 
burning = True
for i in np.arange(0,max_it,1): #steps
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
                start_params = np.array([start_col4, start_col8, start_col9])
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
                    start4 = np.random.normal(loc=param_guess_matrix[0], scale=jumpsize_col4)
                    start8 = np.random.normal(loc=param_guess_matrix[1], scale=jumpsize_col8)
                    start9 = np.random.normal(loc=param_guess_matrix[2], scale=jumpsize_col9)
                    start_params = np.array([start4, start8, start9])
                    print "start_params = ", start_params 
                    print "param_guess_matrix shape = ", param_guess_matrix.shape


                else:
                    if param_guess_matrix.shape == (param_number,):
                        start4 = np.random.normal(loc=param_guess_matrix[0], scale=jumpsize_col4)
                        start8 = np.random.normal(loc=param_guess_matrix[1], scale=jumpsize_col8)
                        start9 = np.random.normal(loc=param_guess_matrix[2], scale=jumpsize_col9)
                        start_params = np.array([start4, start8, start9])
                        print "start_params = ", start_params    

                    else:
                        start4 = np.random.normal(loc=param_guess_matrix[-1][0], scale=jumpsize_col4)
                        start8 = np.random.normal(loc=param_guess_matrix[-1][1], scale=jumpsize_col8)
                        start9 = np.random.normal(loc=param_guess_matrix[-1][2], scale=jumpsize_col9)
                        start_params = np.array([start4, start8, start9])
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

            y_guess = total_loglike


            
            function_value_matrix = np.vstack((function_value_matrix, np.array(y_guess))) 
            new_loglike = total_loglike


            """if datatype == 'm': #what is this?
                new_loglike = total_loglike
            else:
                new_loglike = total_loglike
            """


            loglike_full_list.append(new_loglike)
            
            burning = False  ### NEEDS TO CHANGE!

            
            
            
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
                #prob = np.exp(-((loglike_list[-1] - new_loglike)**2)/2) # THIS MAY NEED TO CHANGE.
                prob = np.exp(-((normlastloglike - normloglike)**2)/2) #TRYING THIS NOVEMBER 9 2015
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
plt.scatter(param_guess_matrix.T[param_number-2][np.argmin(loglike_list)], param_guess_matrix.T[param_number-1][np.argmin(loglike_list)], s=200, label='Final Value', c='g', marker='v')
plt.legend(scatterpoints=1, loc=0)
plt.grid()
plt.title('Random Walk')
#plt.savefig('MCMC_random_walk.pdf', dpi=100, clobber=True)
plt.show()


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



##PLOT X2 Test
plt.plot(np.arange(0,len(loglike_list), 1), loglike_list)
plt.ylim(0,np.amax(loglike_list))
plt.title(r'$\chi^2$ Values vs Accepted Iteration')
plt.grid()
#plt.savefig('MCMC_X2.pdf', dpi=100)
plt.show()


#posterior distributions
for param in np.arange(0,param_number,1):
    postburn =int( max_it/5)
    n, bins, patches = P.hist(param_full_matrix.T[param][postburn:], 50)
    P.ylim(0,np.amax(n)+100)
    stddev = np.std(param_full_matrix.T[param][postburn:])
    median = np.median(param_full_matrix.T[param][postburn:])
    P.plot(np.linspace(median, median, 100), np.linspace(0,np.amax(n)+100,100), c='r', label='Median')
    P.plot(np.linspace(param_full_matrix.T[param][np.argmin(loglike_full_list)], param_full_matrix.T[param][np.argmin(loglike_full_list)], 100), np.linspace(0,np.amax(n)+100, 100), c='b', label='Best')
    P.plot(np.linspace(median-stddev, median-stddev,100), np.linspace(0,np.amax(n)+100,100), c='g', label=r'1 $\sigma$')
    P.plot(np.linspace(median+stddev, median+stddev,100), np.linspace(0,np.amax(n)+100,100), c='g')
    P.title('Coefficient '+str(param)+' = '+str(median))
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
best_X2 = np.amin(loglike_list)
best_coefs = param_guess_matrix[np.argmin(loglike_list)]


#Likelihood = best_X2 + len(xvals)*np.log(param_number) # WHAT DAVID TOLD ME TO DO
#Likelihood = best_X2 + param_number*np.log(len(xvals)) # reversing the meanings of k and N.

#NEW BIC FORMULA FROM DAVID:
BIC = best_X2 + param_number*(np.log(len(xvals)))

#BIC = -2 * np.log(Likelihood) + param_number*np.log(len(xvals))  ### VERY SUSPICIOUS ABOUT THIS.

#print "Likelihood = ", Likelihood
print "BIC = ", BIC
print "best X2 = ", best_X2
print "coefficients are ", param_full_matrix[np.argmin(loglike_full_list)]

