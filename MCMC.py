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

max_it = 1e4

xvals = []
yvals = []
noisy_yvals = []
sigmas = []
noise_additions = []
y_matrix = []
X2_list = []
X2_full_list = []
reject_list = []
random_draw_list = []
color_list = []

X2_list_post_burn = []
index_post_burn = []

coef_number = [] 
coef_options = []
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
        #print 'linedata = ', linedata
        #xvals.append(float(linedata[0]))
        #yvals.append(float(linedata[1]))
        #noisy_yvals.append(float(linedata[1]))
        #sigmas.append(float(linedata[2]))

        ### THIS REFLECTS THE NEW DATA CHAINS DAVID SENT ME - NOVEMBER 4TH
        try:
            #print "linedata[0] = ", linedata[0]
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


    #THESE SHOULD PROBABLY GO AWAY -- NOT RELEVANT FOR OUR WORK.
    """
    coefs= int(raw_input('How many coefficients do you want? '))
    print "coef numbers appending"
    coef_number.append(coefs)
    print "coef_options appending"
    coef_options.append(np.linspace(-2,2,1000)) #NOTE THAT YOU HAVE TO GRAB [0]
    print "exponents appending"
    exponents.append(np.arange(0,coefs,1)) #NOTE THAT YOU HAVE TO GRAB [0] 
    """


"""
def make_data():
    coefs= int(raw_input('How many coefficients do you want? '))
    coef_number.append(coefs)
    coef_options.append(np.linspace(-2,2,1000)) #NOTE THAT YOU HAVE TO GRAB [0]
    exponents.append(np.arange(0,coef_number,1)) #NOTE THAT YOU HAVE TO GRAB [0] 
    true_coefs = np.random.choice(coef_options, size=coefs)

    norm_yvals = raw_input('Do you want to normalize y values? y/n: ')
    xvals = np.arange(-50,50,5)
    for x in xvals:
        yvals.append(np.sum(true_coefs*x**exponents))
    yvals = np.array(yvals)

    if norm_yvals == 'y':
        yvals = yvals/np.amax(yvals)

    for i in yvals:
        noise_term = np.random.normal(loc=i, scale=(0.36*i))
        sigmas.append(noise_term)
        noisy_yvals.append(i + noise_term)

"""





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


#raise Exception("This is all you want to do.")




## CRITICAL LOADING
"""
xvals = np.array(xvals)
yvals = np.array(yvals)
sigmas = np.array(sigmas)
noisy_yvals = np.array(noisy_yvals) ### IS THIS OK? OCT 2.
expected_X2 = len(xvals) - coef_number[0]

try:
    coef_number = coef_number[0]
except:
    print "could not properly grab coef_number."
try:
    coef_options = coef_options[0]
except:
    print "could not properly grab coef_options"
try:
    exponents = exponents[0]
except:
    print "could not properly grab exponents"



use_half = raw_input("Do you want to use '1'st half, '2'nd half, or 'a'll of the data? ")

if use_half == '1':
    xvals = xvals[0:len(xvals)/2]
    yvals = yvals[0:len(yvals)/2]
    noisy_yvals = noisy_yvals[0:len(noisy_yvals)/2]
    sigmas = sigmas[0:len(sigmas)/2]
    
elif use_half == '2':
    xvals = xvals[len(xvals)/2:]
    yvals = yvals[len(yvals)/2:]
    noisy_yvals = noisy_yvals[len(noisy_yvals)/2:]
    sigmas = sigmas[len(sigmas)/2:]
else:
    pass

"""



# # # # # MCMC CODE # # # # # 
burning = True
for i in np.arange(0,max_it,1): #steps
    gocode = 1
    print "i = ", i
    print "burning = ", burning
    
    if i == 0:
        ### INITIAL RUN.
        color_list.append(i/max_it)
        if datatype == 'r':
            #start_coefs = np.random.choice(coef_options, size=coef_number)
            start_coefs = np.array([start_col4, start_col8, start_col9]) #should rename this  --- start_params
            
            #exponents = np.arange(0,coef_number,1) #needs to change for variable size, same as the line above.
            print "start_coefs (start_params) = ", start_coefs
            #print "exponents = ", exponents
        else:
            start_coefs = np.random.choice(coef_options, size=coef_number)
        print "start_coefs = ", start_coefs
        if datatype == 'f':
            print "true_coefs = ", true_coefs

        coef_guess_matrix = start_coefs
        coef_full_matrix = start_coefs

        #y_1st_guess = [] #the function value is going to be the likelihoods we calculate.
        #for x in xvals:
            #y_1st_guess.append(np.sum(start_coefs*x**exponents))
        #y_1st_guess = np.array(y_1st_guess)


        ### NEW FUNCTIONALITY TO BRING IN THE ND INTERPOLATOR:
        execfile("NDmcmc.py") #change that code to make it work better for this one.
        intmode = 'mcmc'
        test_type = 'real'
        loglikes = []
        for p in np.arange(0,uber_column_dictionary.shape[0],1):
            testpoint = start_coefs
            loglikes.append(interp(testpoint, uber_column_dictionary[p])[0]) ### NEED TO WRITE THE NDint code to handle calling from the uber_column_dictionary
        
        total_loglike = np.sum(loglikes)
        y_1st_guess = total_loglike #INITIAL FUNCTION VALUE.


        try:
            if norm_yvals == 'y':
                y_1st_guess = y_1st_guess/np.amax(y_1st_guess)
        except:
            pass
        


        if datatype == 'f':
            new_X2 = X2(noisy_yvals, y_1st_guess, sigmas)
        else:
            #new_X2 = X2(yvals, y_1st_guess, sigmas)
            new_X2 = total_loglike #reflects the fact that our function value is the direct test of goodness of fit. MAKE SURE YOU ARE MAXING, NOT MINNING, THIS VALUE!

        old_X2 = new_X2 #just for comparison in subsequent iterations.
        #switching it up here to start throwing away the first 20% of the runs:
        if i > max_it / 5:
        #if new_X2 < expected_X2 + 10:
            burning = False



        
        X2_list.append( new_X2  )
        if burning == False:
            X2_list_post_burn.append(new_X2)
            #index_post_burn.append(i)
            try:
                index_post_burn.append(index_post_burn[-1]+1)
            except:
                index_post_burn.append(0)
        X2_full_list.append(new_X2)
        
        y_matrix = y_1st_guess #APPENDING THE FUNCTION VALUE.
                  



        
    else: # ALL SUBSEQUENT RUNS.
        if gocode == 1:
            if i == 1:
               # if norm_yvals == 'y':
                start4 = np.random.normal(loc=coef_guess_matrix[0], scale=jumpsize_col4)
                start8 = np.random.normal(loc=coef_guess_matrix[1], scale=jumpsize_col8)
                start9 = np.random.normal(loc=coef_guess_matrix[2], scale=jumpsize_col9)
                start_coefs = np.array([start4, start8, start9])
                print "start_coef = ", start_coefs 
                #else:
                    #start_coefs = np.random.normal(loc=coef_guess_matrix, scale=0.1)
                print "coef_guess_matrix shape = ", coef_guess_matrix.shape
            else:
                if coef_guess_matrix.shape == (coef_number,):
                    start4 = np.random.normal(loc=coef_guess_matrix[0], scale=jumpsize_col4)
                    start8 = np.random.normal(loc=coef_guess_matrix[1], scale=jumpsize_col8)
                    start9 = np.random.normal(loc=coef_guess_matrix[2], scale=jumpsize_col9)
                    #start_coefs = np.random.normal(loc=coef_guess_matrix, scale=0.1)
                    start_coefs = np.array([start4, start8, start9])
                    print "start_coefs = ", start_coefs                   
                else:
                    start4 = np.random.normal(loc=coef_guess_matrix[-1][0], scale=jumpsize_col4)
                    start8 = np.random.normal(loc=coef_guess_matrix[-1][1], scale=jumpsize_col8)
                    start9 = np.random.normal(loc=coef_guess_matrix[-1][2], scale=jumpsize_col9)
                    start_coefs = np.array([start4, start8, start9])
                    #start_coefs = np.random.normal(loc=coef_guess_matrix[-1], scale=0.1)
                    print "start_coefs = ", start_coefs

                        
                print "coef_guess_matrix shape = ", coef_guess_matrix.shape
            #print "start_coefs = ", start_coefs
            try:
                print "true_coefs = ", true_coefs
            except:
                pass


        else: #if gocode == 0:
            if i == 1:
                start_coefs = coef_guess_matrix
            else:
                #pass
                start_coefs = coef_guess_matrix[-1]
            gocode = 1
            
            
        coef_full_matrix = np.vstack((coef_full_matrix, start_coefs))
    
        #NEW ITERATION CALCULATION

        execfile("NDmcmc.py") #change that code to make it work better for this one.
        intmode = 'mcmc'
        test_type = 'real'
        loglikes = []
        for p in np.arange(0,uber_column_dictionary.shape[0],1):
            testpoint = start_coefs
            loglikes.append(interp(testpoint, uber_column_dictionary[p])[0]) ### NEED TO WRITE THE NDint code to handle calling from the uber_column_dictionary
        
        total_loglike = np.sum(loglikes)

        #y_1st_guess = total_loglike #INITIAL FUNCTION VALUE.

        y_guess = total_loglike
        #y_guess = []
        #for x in xvals:
            #y_guess.append(np.sum(start_coefs*x**exponents))
            #y_guess = 
        #y_guess = np.array(y_guess)

        #try:
            #if norm_yvals == 'y':
                #y_guess = y_guess/np.amax(y_guess)
        #except:
            #pass

        
        y_matrix = np.vstack((y_matrix, np.array(y_guess))) 
        if datatype == 'm':
            new_X2 = total_loglike
            #new_X2 = X2(noisy_yvals, y_guess, sigmas)
        else:
            #new_X2 = X2(yvals, y_guess, sigmas)
            new_X2 = total_loglike
        X2_full_list.append(new_X2)
        
        #if new_X2 < expected_X2 + 10:
        burning = False  ### NEEDS TO CHANGE!

        
        
        
        if new_X2 > X2_list[-1]:    ### THIS MARKS AN IMPROVEMENT
            color_list.append(i/max_it) 
            try:
                coef_guess_matrix = np.vstack((coef_guess_matrix, start_coefs))
            except:
                #coef_guess_matrix = start_coefs
                pass

            X2_list.append(new_X2)
            if burning == False:
                X2_list_post_burn.append(new_X2)
                try:
                    index_post_burn.append(index_post_burn[-1]+1)
                except:
                    index_post_burn.append(0)
                
                #index_post_burn.append(i)
            
            if burning == False:
                try:
                    coef_guess_post_burn = np.vstack((coef_guess_post_burn, start_coefs))
                except:
                    coef_guess_post_burn = start_coefs
                

                #X2_list_post_burn.append(new_X2)
                #index_post_burn.append(i)

            
                                          
        else:  ### IF X^2 isn't BETTER.
            prob = np.exp(-((X2_list[-1] - new_X2)**2)/2) # THIS MAY NEED TO CHANGE.
            random_draw = np.random.normal(loc=0.5, scale=0.1)
            
            if prob > random_draw: # IF PROB GIVES YOU A SHOT
                color_list.append(i/max_it)
                try:
                    coef_guess_matrix = np.vstack((coef_guess_matrix, start_coefs))
                except:
                    print "coef_guess_matrix not stackable."
                    coef_guess_matrix = start_coefs


                X2_list.append(new_X2)


                if burning == False:
                    X2_list_post_burn.append(new_X2)
                    try:
                        index_post_burn.append(index_post_burn[-1]+1)
                    except:
                        index_post_burn.append(0)
                    
                    try:
                        coef_guess_post_burn = np.vstack((coef_guess_post_burn, start_coefs))
                    except:
                        print "coef_guess_matrix not stackable."
                        coef_guess_post_burn = start_coefs

                
                print "X2 worse, prob better than random normal draw."
                print "prob = ", prob
                print "random draw = ", random_draw
                random_draw_list.append(random_draw)                  


                
            else:  ## IF PROB DOESN"T WORK.
                reject_list.append(i)
                print "REJECTED. X2 worse, prob worse than random normal draw."
                print "prob = ", prob
                print "random draw = ", random_draw
                random_draw_list.append(random_draw)
                gocode = 0
                print "gocode = 0"
                #continue
                
                
    print "new X2 = ", new_X2
    print " "
    print " "
    
    
    
    
    
X2_list_post_burn = np.array(X2_list_post_burn)
index_post_burn = np.array(index_post_burn).astype(int)
    
    
    
    
    
    
    
    
### PLOTTING #### 

## RANDOM DRAW HISTOGRAM -- CENTERED ON 0.5
#n,bins,patches = P.hist(random_draw_list, 50, histtype='bar')
#P.show()


### PLOT THE DATA AND THE BEST FIT
#for i in np.arange(0,y_matrix.shape[0],1):
    #plt.plot(xvals, y_matrix[i], c=cm.coolwarm(i/y_matrix.shape[0]), alpha=0.1)
plt.plot(xvals, y_1st_guess, c='r', label='First Guess') ### 1ST GUESS
if datatype == 'f':
    plt.plot(xvals, yvals, c='b', label='True Solution', linewidth=3, alpha=0.5)  ### THE REAL ANSWER
plt.scatter(xvals, noisy_yvals, c='b', s=10)  ### THE NOISY DATA
plt.errorbar(xvals, noisy_yvals, yerr=sigmas, c='b', fmt=None)
plt.plot(xvals, y_matrix[np.argmin(X2_full_list)], c='k', label='Best Guess', linewidth=2)
#for i in np.arange(0,len(m_list),1):
    #plt.plot(xvals, y_guess, c=cm.hot(i/1), alpha=0.2) #doesn't work right now... don't have a big list of guesses.
plt.title('MCMC Sandbox')
plt.legend()
plt.grid()
#plt.savefig('MCMC_sandbox.pdf', dpi=100, clobber=True)
plt.show()



### RANDOM WALK VISUALIZATION
plt.scatter(coef_guess_matrix.T[coef_number-2], coef_guess_matrix.T[coef_number-1], s=20, c=color_list, alpha=0.2, label='Random Walk')
plt.scatter(coef_guess_matrix[0][coef_number-2], coef_guess_matrix[1][coef_number-1], s=80, label='Start Value', c='g', marker='s')
try:
    plt.scatter(true_coefs[coef_number-2], true_coefs[coef_number-1], label='True Value', c='r', marker='*', s=200)
except:
    pass
plt.scatter(coef_guess_matrix.T[coef_number-2][np.argmin(X2_list)], coef_guess_matrix.T[coef_number-1][np.argmin(X2_list)], s=200, label='Final Value', c='g', marker='v')
plt.legend(scatterpoints=1, loc=0)
plt.grid()
plt.title('Random Walk')
#plt.savefig('MCMC_random_walk.pdf', dpi=100, clobber=True)
plt.show()


### RANDOM WALK VISUALIZATION (POST BURN)
"""
post_burn_color_list = index_post_burn/len(index_post_burn)
plt.scatter(coef_guess_matrix.T[coef_number-2][index_post_burn], coef_guess_matrix.T[coef_number-1][index_post_burn], s=20, c=post_burn_color_list, alpha=0.2, label='Random Walk')
plt.scatter(coef_guess_matrix[0][coef_number-2], coef_guess_matrix[1][coef_number-1], s=80, label='Start Value', c='g', marker='s')
try:
    plt.scatter(true_coefs[coef_number-2], true_coefs[coef_number-1], label='True Value', c='r', marker='*', s=200)
except:
    pass
plt.scatter(coef_guess_matrix.T[coef_number-2][np.argmin(X2_list)], coef_guess_matrix.T[coef_number-1][np.argmin(X2_list)], s=200, label='Final Value', c='g', marker='v')
plt.legend(scatterpoints=1, loc=0)
plt.grid()
#plt.savefig('MCMC_random_walk.pdf', dpi=100, clobber=True)
plt.title("Random Walk (Post Burn)")
plt.show()
"""



##PLOT X2 Test
plt.plot(np.arange(0,len(X2_list), 1), X2_list)
plt.ylim(0,np.amax(X2_list))
plt.title(r'$\chi^2$ Values vs Accepted Iteration')
plt.grid()
#plt.savefig('MCMC_X2.pdf', dpi=100)
plt.show()


#posterior distributions
for param in np.arange(0,coef_number,1):
    postburn =int( max_it/5)
    n, bins, patches = P.hist(coef_full_matrix.T[param][postburn:], 50)
    P.ylim(0,np.amax(n)+100)
    stddev = np.std(coef_full_matrix.T[param][postburn:])
    median = np.median(coef_full_matrix.T[param][postburn:])
    P.plot(np.linspace(median, median, 100), np.linspace(0,np.amax(n)+100,100), c='r', label='Median')
    P.plot(np.linspace(coef_full_matrix.T[param][np.argmin(X2_full_list)], coef_full_matrix.T[param][np.argmin(X2_full_list)], 100), np.linspace(0,np.amax(n)+100, 100), c='b', label='Best')
    P.plot(np.linspace(median-stddev, median-stddev,100), np.linspace(0,np.amax(n)+100,100), c='g', label=r'1 $\sigma$')
    P.plot(np.linspace(median+stddev, median+stddev,100), np.linspace(0,np.amax(n)+100,100), c='g')
    P.title('Coefficient '+str(param)+' = '+str(median))
    P.grid()
    P.show()



#samples = np.vstack((m_list[300:], b_list[300:], c_list[300:]))
X2_array = np.array(X2_list)
Reasonable_X2_idx = np.where(X2_array < 30)


## DON'T HAVE CORNER LOADED ON LINUX.
"""
try:
    samples = np.vstack((coef_guess_matrix.T[coef_number-1][Reasonable_X2_idx], coef_guess_matrix.T[coef_number-2][Reasonable_X2_idx], coef_guess_matrix.T[coef_number-3][Reasonable_X2_idx]))
except:
    samples = np.vstack((coef_guess_matrix.T[coef_number-1], coef_guess_matrix.T[coef_number-2], coef_guess_matrix.T[coef_number-3]))
samples = samples.T
figure = corner.corner(samples, labels = ['a', 'b', 'c'])
figure.savefig('MCMC_corner.png', clobber=True)
figure.show()
"""


### CALCULATE BIC
print " "
print " "
best_X2 = np.amin(X2_list)
best_coefs = coef_guess_matrix[np.argmin(X2_list)]


#Likelihood = best_X2 + len(xvals)*np.log(coef_number) # WHAT DAVID TOLD ME TO DO
#Likelihood = best_X2 + coef_number*np.log(len(xvals)) # reversing the meanings of k and N.

#NEW BIC FORMULA FROM DAVID:
BIC = best_X2 + coef_number*(np.log(len(xvals)))

#BIC = -2 * np.log(Likelihood) + coef_number*np.log(len(xvals))  ### VERY SUSPICIOUS ABOUT THIS.

#print "Likelihood = ", Likelihood
print "BIC = ", BIC
print "best X2 = ", best_X2
print "coefficients are ", coef_full_matrix[np.argmin(X2_full_list)]




### THIS IS ALL OLD.
"""
Likelihood_list = []
variance_list = []
mean_list = []
model_number = y_matrix.shape[0]
data_points_number = y_matrix.shape[1]
for i in np.arange(0, data_points_number, 1): # for each data point (~20)
    variance = np.var(y_matrix.T[i]) #variance of the model distribution at that point
    print "variance = ", variance
    variance_list.append(variance)
    mean = np.mean(y_matrix.T[i]) #mean of the model distribution at that point
    print "mean = ", mean
    mean_list.append(mean)
    
    Likelihood = (2*np.pi*variance)**(-model_number/2)  *  np.exp( (-1)/(2*variance)  * np.sum(y_matrix.T[i] - mean) )
    print "Likelihood = ", Likelihood
    print " "
    print " "
    

#for i in np.arange(0,model_number, 1): #for each model (~10000)
    #for d in np.arange(0, data_points_number, 1):
        #Likelihood = (2*np.pi)**(-data_points_number/2) * np.exp*( (-1/(2*variance_list[d])) * np.sum(y_matrix[i]
    
"""




    
    #THIS SHOULD BE VARIANCE AND MEAN OF THE POSTERIORS IN A GIVEN PARAMETER?
    #Likelihood = (2*np.pi*variance)**(-i/2) * np.exp((-1/(2*variance))) * np.sum((y_matrix[i] - mean)**2)
    #print "Likelihood = ", Likelihood


