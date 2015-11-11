from __future__ import division
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


#col1  #Rp/R*
#col2  #rho*
#col3  #impact parameter b
#col6  #q1..?
#col7  #q2..?
#col16 #log(likelihoods) 

#dim_choice = int(raw_input('how many dimensions do you want? '))
dim_choice = start_params.shape
#test_type = raw_input("Do you want to test 'a'ccuracy, or 's'catter plotting? ")
run_sintest = 'y'
#run_sintest = raw_input("Do you want to run a sintest? y/n: ")
#numpoints = int(raw_input('how many points do you want to generate in the random sample? '))



def sinfun(e1,e2,dim,e3=0,e4=0,e5=0):
    if dim == 5:
        return np.sin(e1)*np.cos(e2)*np.sin(e3)*np.cos(e4)*np.sin(e5)
    elif dim == 4:
        return np.sin(e1)*np.cos(e2)*np.sin(e3)*np.cos(e4)
    elif dim == 3:
        return np.sin(e1)*np.cos(e2)*np.sin(e3)
        #return (3*x**2) - (2*y**3) + (7*z**(-3))
        #return (3*(x-2)**2) - (7*y**3) - (14*z**(-4))
        #return (3*x) - (y-2)**(-2) + (1/2)*z + 4
    elif dim == 2:
        return np.sin(e1)*np.cos(e2)


"""
col1 = []
col2 = []
col3 = []
col4 = []
col5 = []
col6 = []
col7 = []
col8 = []
col9 = []
col10 = []
col11 = []
col12 = []
col13 = []
col14 = []
col15 = []
col16 = []
"""

#def read_data():
#data = open('../Data/post1.dat', mode='r') #reflects file location on Umbriel


"""
if run_sintest == 'n':
    data = open('../Data/LRVplan1-post_equal_weights.dat', mode='r')
    #for i in np.arange(0,35097,1): #not flexible right now
    for i in data:
        #linedata = data.readline().split()
        linedata = i.split()
        col1.append(float(linedata[0]))
        col2.append(float(linedata[1]))
        col3.append(float(linedata[2]))
        col4.append(float(linedata[3]))
        col5.append(float(linedata[4]))
        col6.append(float(linedata[5]))
        col7.append(float(linedata[6]))
        col8.append(float(linedata[7]))
        
        col9.append(float(linedata[8]))
        col10.append(float(linedata[9]))
        col11.append(float(linedata[10]))
        col12.append(float(linedata[11]))
        col13.append(float(linedata[12]))
        col14.append(float(linedata[13]))
        col15.append(float(linedata[14]))
        col16.append(float(linedata[15]))
        
        
    col1 = np.array(col1)
    col2 = np.log10(np.array(col2)) #per David's instructions, Oct 29.
    col3 = np.array(col3)
    col4 = np.array(col4)
    col5 = np.array(col5)
    col6 = np.array(col6)
    col7 = np.array(col7)
    col8 = np.array(col8)
    
    col9 = np.array(col9)
    col10 = np.array(col10)
    col11 = np.array(col11)
    col12 = np.array(col12)
    col13 = np.array(col13)
    col14 = np.array(col14)
    col15 = np.array(col15)
    col16 = np.array(col16)
    

elif run_sintest == 'y': #this will supercede what came before.
    colchoice = np.linspace(-np.pi,np.pi,100000)
    col1 = np.random.choice(colchoice, size=numpoints)
    col2 = np.random.choice(colchoice, size=numpoints)
    col3 = np.random.choice(colchoice, size=numpoints)
    col4 = np.random.choice(colchoice, size=numpoints)
    col5 = np.random.choice(colchoice, size=numpoints)
    col6 = np.random.choice(colchoice, size=numpoints)
    col7 = np.random.choice(colchoice, size=numpoints)
    #col16 = sinfun(col1,col2,col3) 3D
    if dim_choice == 2:
        col16 = sinfun(col2,col6, dim_choice) #was this the problem (nov 1)? was col2 and col7.
    elif dim_choice == 3:
        col16 = sinfun(col2,col6, dim_choice, e3=col7)
    elif dim_choice == 4:
        col16 = sinfun(col2, col6, dim_choice, e3=col7, e4=col4)
    elif dim_choice == 5:
        col16 = sinfun(col1, col2, dim_choice, e3=col3, e4=col6, e5=col7)

    col8 = col16 #takes care of using col8 as the results column.


    samplespace_density =len(col1)/ ((4*np.pi)**2)
    #print "sample space density = ", samplespace_density


    """







#USE THIS EVENTUALLY, WHEN YOU WORK OUT THE BUGS.
def coordpair(coordperm,ndim,nmatches):
    coordpairidx = [] #indices that have the same final values (i.e. x0,y0,z0 and x1,y0,yz)
    ## GENERALIZE:
    for i in np.arange(0,len(coordperm)-1,1):
        idxi,idxj = i,i+1
        try:
            temparray = coordinate_permutations[idxi][(ndim-nmatches):] == coordinate_permutations[idxj][(ndim-nmatches):] #for 5D, matching four points, this is [1:]
        except:
            break
        while False in temparray:
            idxj = idxj+1
            try:
                temparray = coordinate_permutations[idxi][(ndim-nmatches):] == coordinate_permutations[idxj][(ndim-nmatches):] #for 5D, matching fours, this is[1:]
            except:
                break
        if idxj != len(coordinate_permutations):
            coordpairidx.append(np.array([idxi, idxj]))
    

    return coordpairidx




def pointgen(numdims):
    if numdims == 2:
        #return np.array([np.random.choice(col1), np.random.choice(col3)])
        # REFLECTS OCT 28 DATA.
        return np.array([np.random.choice(col2), np.random.choice(col6)])
    elif numdims == 3:
        ### REFLECTS OCT 28 DATA.
        return np.array([np.random.choice(col2), np.random.choice(col6), np.random.choice(col7)])
        
    elif numdims == 4:
        return np.array([np.random.choice(col1), np.random.choice(col2), np.random.choice(col3), np.random.choice(col6)])
    elif numdims == 5:
        return np.array([np.random.choice(col1), np.random.choice(col2), np.random.choice(col3), np.random.choice(col6), np.random.choice(col7)])
    


## BEGIN FUNCTION ##


def interp(point,somedict): 
    numdims = len(point)
    if numdims == 2:
        rows = np.vstack((somedict['col6'],somedict['col7']))
    elif numdims == 3:
        rows = np.vstack((somedict['col2'],somedict['col6'],somedict['col7']))

    elif numdims == 4:
        rows = np.vstack((col1,col2,col3,col6))
    elif numdims == 5:
        rows = np.vstack((col1,col2,col3,col6,col7))


    #finding distance to each point
    distances = []
    for i in np.arange(0,len(somedict['col8']),1): #for each point in your sample, calculate a distance.
        distance = np.sqrt(np.sum( ( point - rows.T[i])**2) )
        ##print "distance = ", distance
        distances.append(distance)
    distances = np.array(distances)
    sorted_args = np.argsort(distances) 
    sorted_distances = distances[sorted_args]
    sorted_distances2 = np.sort(distances)
    """
    if False in (sorted_distances == sorted_distances2):
        #print "DETECTED A ERROR IN SORTED DISTANCES"
    else:
        #print "SORTED DISTANCES WORKING."
    ##print "len(sorted_distances) = ", len(sorted_distances)
    
    ##print "sorted distance test = ", sorted_distances == sorted_distances2
    """

    #nearest_value = col8[sorted_args][0]
    ##print "nearest_value = ", col8[sorted_args][0]
    ##print "nearest_point = ", col2[sorted_args][0], col6[sorted_args][0]
    ##print "test point = ", point
    ##print "distance to nearest point = ", sorted_distances[0]
    ##print "calculated distance to nearest point (invalid above 2D)= ", np.sqrt((point[0] - col2[sorted_args][0])**2 + (point[1] - col6[sorted_args][0])**2)

    



    ### GENERATING ALL THE COORDINATE PERMUTATIONS.
    coordinate_permutations = []
    if numdims == 2:
        for e1 in [0,1]:
            for e2 in [0,1]:
                coordinate_permutations.append((e1,e2))

    elif numdims == 3:
        for e1 in [0,1]:
            for e2 in [0,1]:
                for e3 in [0,1]:
                    coordinate_permutations.append((e1,e2,e3))

    elif numdims == 4:
        for e1 in [0,1]:
            for e2 in [0,1]:
                for e3 in [0,1]:
                    for e4 in [0,1]:
                        coordinate_permutations.append((e1,e2,e3,e4))

    elif numdims == 5:
        for e1 in [0,1]:
            for e2 in [0,1]:
                for e3 in [0,1]:
                    for e4 in [0,1]:
                        for e5 in [0,1]:
                            coordinate_permutations.append((e1,e2,e3,e4,e5))


    coordinate_permutations = np.array(coordinate_permutations)
    coordT = coordinate_permutations.T
    numslots = len(coordinate_permutations)
        
    #print "len(coordinate_permutations) = ", len(coordinate_permutations)
    #print "coordinate_permutations = ", coordinate_permutations




    # locating the four (or whatever)  nearest points that fit requirements
    # DON"T NEED TO TOUCH THIS, SHOULD SCALE FOR N-DIMENSIONS. NOT SURE IF IT DOES THOUGH.
    new = []
    found = []

    realcoords = [] #the actual coordinates in space.
    coords01 = [] #the binary coordinates.
    functionvals = []

    for p,a in zip(rows.T[sorted_args], sorted_args): ## LOSING COORDS SOMEHOW.
        spatial_differences = p - point # just preserves the spatial differences instead of replacing them with binary coordinates.
        differences = p - point  #calculates the differences to every point.

        np.place(differences, differences<=0, 0) # If the coordinate is less than the coordinate of the point in question, it's assigned a 0       
        np.place(differences, differences>0, 1) # If the coordinate is greater than the coordinate of the point in question, it's assigned a 1
        try:
            coordinate_dictionary[tuple(differences)] #test whether the key is already there. If it is do nothing.
            coordinate_differences[tuple(differences)]
            coordinate_indices[tuple(differences)]
            coordinate_vals[tuple(differences)]
            found.append(1)
        except:
            realcoords.append(p)
            functionvals.append(somedict['col8'][a])
            coords01.append(differences)
            #print "coord found = ", differences, tuple(p)

            new.append(1)
            try:
                coordinate_dictionary[tuple(differences)] = tuple(p) #these are all callable by (0,1) coordinate tuple.
                #coordinate_differences[tuple(differences)] = tuple(p - point)
                coordinate_differences[tuple(differences)] = np.sqrt(np.sum( (p-point)**2 ) )
                coordinate_indices[tuple(differences)] = a
                #coordinate_vals[tuple(differences)] = col16[a]
                coordinate_vals[tuple(differences)] = somedict['col8'][a]
            except:
                coordinate_dictionary= {tuple(differences) : tuple(p)}
                #coordinate_differences = {tuple(differences) : tuple(p - point)}
                coordinate_differences = {tuple(differences) : np.sqrt(np.sum( (p-point)**2 ) ) }
                coordinate_indices = {tuple(differences) : a}
                #coordinate_vals = {tuple(differences) : col16[a]}
                coordinate_vals = {tuple(differences) : somedict['col8'][a]}
        if len(coordinate_dictionary) == numslots:
            break

    #print "len(coordinate_dictionary) = ", len(coordinate_dictionary)
    #print " "
    #print "len(coordinate_differences) = ", len(coordinate_differences)
    #print " "
    #print "coordinate dictionary values = ", coordinate_dictionary.values()
    #print " "
    #print "coordinate_differences values = ", coordinate_differences.values()
    #print "coordinate_differences.items() ", coordinate_differences.items()
    #if dim_choice == 2:
        #print "coordinate_dictionary[(0,0)] = ", coordinate_dictionary[(0,0)]
    #elif dim_choice ==3:
        #print "coordinate_dictionary[(0,0,0)] = ", coordinate_dictionary[(0,0,0)]
    #print "point = ", point


    coord_to_coord_distance = []
    associated_permutation = []
    
    for perm in coordinate_permutations:
        #associated_permutations.append(perm)
        if dim_choice == 2:
            c2cd = np.sqrt((coordinate_dictionary[tuple(perm)][0] - point[0])**2 + (coordinate_dictionary[tuple(perm)][1] - point[1])**2)
        elif dim_choice == 3:
            c2cd = np.sqrt((coordinate_dictionary[tuple(perm)][0] - point[0])**2 + (coordinate_dictionary[tuple(perm)][1] - point[1])**2 + (coordinate_dictionary[tuple(perm)][2] - point[2])**2)
        elif dim_choice == 4:
            c2cd = np.sqrt((coordinate_dictionary[tuple(perm)][0] - point[0])**2 + (coordinate_dictionary[tuple(perm)][1] - point[1])**2 + (coordinate_dictionary[tuple(perm)][2] - point[2])**2 + (coordinate_dictionary[tuple(perm)][3] - point[3])**2)
        elif dim_choice == 5:
            c2cd = np.sqrt((coordinate_dictionary[tuple(perm)][0] - point[0])**2 + (coordinate_dictionary[tuple(perm)][1] - point[1])**2 + (coordinate_dictionary[tuple(perm)][2] - point[2])**2 + (coordinate_dictionary[tuple(perm)][3] - point[3])**2 + (coordinate_dictionary[tuple(perm)][4] - point[4])**2)



        ##print "c2cd = ", c2cd
        #coord_to_coord_distance.append(c2cd)
    #c2cmean = np.mean(coord_to_coord_distance)
    
        




    ### RUN THIS FOR EACH ROUND
    ### THIS FINDS THE COORDINATE PAIRS.
    pairpoint_function = {'pairpoint' : 'funval'}
    roundnums = np.arange(0,numdims,1)    
    for roundnum in roundnums:
        #print "round number = ", roundnum
        pairpoints = []
        xds = []
        funvals = []
        coordpairidx = [] #indices that have the same final values (i.e. x0,y0,z0 and x1,y0,yz)

        ### COORDPAIR OPERATION ### 
        if roundnum != 0:
            coordinate_permutations = np.array(tempcoords)

        for i in np.arange(0,len(coordinate_permutations)-1,1):
            idxi,idxj = i,i+1
            try:
                temparray = coordinate_permutations[idxi][1:] == coordinate_permutations[idxj][1:]
            except:
                break
            while False in temparray:
                idxj = idxj+1
                try:
                    temparray = coordinate_permutations[idxi][1:] == coordinate_permutations[idxj][1:]
                except:
                    break
            if idxj != len(coordinate_permutations):
                coordpairidx.append(np.array([idxi, idxj]))    
        #print " "
        #print "coordpairidx indices: "
        #for i in np.arange(0,len(coordpairidx),1):
            #print coordpairidx[i]
        #for i in coordpairidx:
            #print tuple(coordinate_permutations[i][0]), tuple(coordinate_permutations[i][1])



        # 1.) CALCULATE THE POINT FOR THE NEXT ROUND.
        tempcoords = []
        for i in coordpairidx: 
            #print "len(coordpairidx) = ", len(coordpairidx)
            #print "coordpairidx = ", i
            coord1,coord2 = tuple(coordinate_permutations[i][0]), tuple(coordinate_permutations[i][1]) #coords of points to draw a line between.
            firstval,secondval = coordinate_dictionary[tuple(coord1)], coordinate_dictionary[tuple(coord2)]            
            #print "coord1, coord2 = ", coord1,coord2
            templine = []
            for t in np.linspace(0,1,1000):
                templine.append(np.array(coordinate_dictionary[coord1]) + (np.array(coordinate_dictionary[coord2])-np.array(coordinate_dictionary[coord1])) *t) #creates a parametric line
            templine = np.array(templine)
            #print "point[roundnum] = ", point[roundnum]
            pairpointarg = np.argmin(np.abs(point[roundnum] - templine.T[roundnum])) #finds the point on the line with the nearest xval to the point.
            pairpoint = templine[pairpointarg]
            #print "pairpoint = ", pairpoint
            pairpoints.append(pairpoint)


            ### BIG CHANGE TO WEIGHTING (OCT 27) -- was the first line of code below.
            ### SWITCHING TO THE DISTANCE TO THE POINT, NOT FROM X-COORDS.
            ### THE TWO EXPRESSIONS FOR XD ARE EQUIVALENT.
            xdold = (pairpoint[roundnum] - firstval[roundnum]) / (secondval[roundnum] - firstval[roundnum])
            xd = xdold
            """
            dppx0 = np.array(pairpoint) - np.array(firstval) #should switch a tuple to an array.
            dppx0 = dppx0**2 #squares every component
            dppx0 = np.sqrt(np.sum(dppx0))
            dx1x0 = np.array(secondval) - np.array(firstval)
            dx1x0 = dx1x0**2
            dx1x0 = np.sqrt(np.sum(dx1x0))
            xd = dppx0/dx1x0
            """

            #print "firstval = ", np.array(firstval)
            #print "secondval = ", np.array(secondval)


            ##print "pairpoint[roundnum] = ", pairpoint[roundnum]
            ##print "firstval[roundnum] = ", firstval[roundnum]
            ##print "secondval[roundnum] = ", secondval[roundnum]
            ##print "x - x0 = ", pairpoint[roundnum] - firstval[roundnum]
            ##print "x1 - x0 = ", secondval[roundnum] - firstval[roundnum]

            #print "xd (old way) = ", xdold 
            #print "xd = ", xd
            xds.append(xd)

            # 2.) CALCULATE THE FUNCTION VALUES.
            funval = coordinate_vals[tuple(coord1)]*(1-xd) + coordinate_vals[tuple(coord2)]*xd
            #print "function val = ", funval
            funvals.append(funval)
            tempcoords.append(coord1[1:])
            #print "appending ", coord1[1:], "to tempcoords."
            #print "len(tempcoords) = ", len(tempcoords)
            #print "tempcoords = ", tempcoords
            coordinate_dictionary[tuple(coord1[1:])] = tuple(pairpoint) #these are all callable by (0,1) coordinate tuple.
            coordinate_vals[tuple(coord1[1:])] = funval
            #print " "
     
    final_value = funval
    #print "final value = ", final_value
    ##print "nearest value = ", nearest_value
    #print "average of all vals = ", np.mean(functionvals)

    """
    if dim_choice == 2:
        actual_value = sinfun(point[0], point[1], dim_choice)
        #print "actual value = ", actual_value
    elif dim_choice == 3:
        actual_value = sinfun(point[0], point[1], dim_choice, e3=point[2])
    elif dim_choice == 4:
        actual_value = sinfun(point[0], point[1], dim_choice, e3=point[2], e4=point[3])
    elif dim_choice == 5:
        actual_value = sinfun(point[0], point[1], dim_choice, e3=point[2], e4=point[3], e5=point[4])

        #print "actual value = ", actual_value
    #print " "
    #print "X X X X X X X X X X"
    #print " "
    #print " "
    
    
    #DAVID"S TEST PER OCT 30 E-MAIL
    final_minus_true = np.abs(final_value - actual_value)
    final_minus_nearest = np.abs(final_value - nearest_value)
    """
    
    
    #### RETURN SECTION ####
    if test_type == 'a':
        return final_value, actual_value, c2cmean, final_minus_true, final_minus_nearest, nearest_value, sorted_distances[0], col8[sorted_args][0], col6[sorted_args][0], point[0], point[1]
    elif test_type == 's':
        if dim_choice == 2:
            return point[0], point[1], final_value, final_value/np.amax(col8), nearest_value, np.mean(functionvals), final_minus_true, final_minus_nearest
        elif dim_choice == 3:
            return point[0], point[1], point[2], final_value, final_value/np.amax(col8), nearest_value, np.mean(functionvals), final_minus_true, final_minus_nearest

    if test_type == 'real':
        return final_value




    









### ALL TESTS BELOW THIS POINT -- DON'T NEED TO WORRY ABOUT THEM.




###### SINUNSOIDAL TEST FUNCTION #############

def sintest():
    test_type = 'a'
    interpvals = []
    actvals = []
    nearest_distances = []
    meandistances = []
    final_minus_nearest = []
    final_minus_true = []
    nearest_values = []
    nearest_x = []
    nearest_y = []
    point_x = []
    point_y = []
    for i in np.arange(0,10000,1):
        try:
            if dim_choice == 2:
                function_output = interp(pointgen(2))
            elif dim_choice == 3:
                function_output = interp(pointgen(3))
            elif dim_choice == 4:
                function_output = interp(pointgen(4))
            elif dim_choice == 5:
                function_output = interp(pointgen(5))
            #try:
            interpvals.append(function_output[0])
            actvals.append(function_output[1])
            meandistances.append(function_output[2])
            final_minus_true.append(function_output[3])
            final_minus_nearest.append(function_output[4])
            nearest_values.append(function_output[5])
            nearest_distances.append(function_output[6])
            nearest_x.append(function_output[7])
            nearest_y.append(function_output[8])
            point_x.append(function_output[9])
            point_y.append(function_output[10])
            
            test_final_minus_true = np.abs(function_output[0] - function_output[1])
            if final_minus_true != test_final_minus_true:
                #print "final_minus_true = ", final_minus_true
                #print "test_final_minus_true = ", test_final_minus_true
                raise Exception('final_minus_true != test_final_minus_true')
                
            #except:
                #raise Exception("There's something not right here. I feel cold, death.")
        except:
            pass
        #print "len(interpvals) = ", len(interpvals)
        #print "len(actvals) = ", len(actvals)

    interpvals = np.array(interpvals)
    actvals = np.array(actvals)
    meandistances = np.array(meandistances)
    #print "len(interpvals) = ", len(interpvals)
    #print "len(actvals) = ", len(actvals)
    #print "len(meandistances) = ", len(meandistances)

    value_differences = actvals - interpvals
    valstd = np.std(value_differences)
    negvalstd = np.mean(value_differences) - valstd
    posvalstd = np.mean(value_differences) + valstd
    #print "len(value_differences) = ", len(value_differences)


    weird_fmn = np.where(final_minus_nearest > 0.2) #this is peculiar to the morning of nov 3 run.
    #print "mean distances for weird fmn vals = ", meandistances[weird_fmn]
    

    
    ### PLOT THE ERROR / DISTANCE HISTOGRAM
    n1,bins1,patches1 = plt.hist(value_differences/meandistances, 200)
    eoverd_std = np.std(value_differences/meandistances)
    fracerr_negvalstd = np.mean(value_differences/meandistances) - eoverd_std
    fracerr_posvalstd = np.mean(value_differences/meandistances) + eoverd_std
    plt.plot(np.linspace(fracerr_negvalstd, fracerr_negvalstd, 10), np.linspace(0,10000,10),c='r', linestyle='--')
    plt.plot(np.linspace(fracerr_posvalstd, fracerr_posvalstd, 10), np.linspace(0,10000,10),c='r', linestyle='--')
    plt.xlabel('Error / Distance')
    plt.title('Error / Distance, stddev = '+str(eoverd_std))
    plt.grid()
    plt.show()


    ### PLOT THE HISTOGRAM
    n,bins,patches = plt.hist(value_differences, 200)

    if dim_choice == 2:
        plt.title('sin(e1)*cos(e2), stddev = '+str(valstd))
    elif dim_choice == 3:
        plt.title('sin(e1)*cos(e2)*sin(e3), stddev = '+str(valstd))
    elif dim_choice == 4:
        plt.title('sin(e1)*cos(e2)*sin(e3)*cos(e4), stddev = '+str(valstd))
    elif dim_choice == 5:
        plt.title('sin(e1)*cos(e2)*sin(e3)*cos(e4)*sin(e5), stddev = '+str(valstd))

    plt.xlabel('Actual Minus Interpolated')
    plt.grid()
    plt.plot(np.linspace(negvalstd, negvalstd, 10), np.linspace(0,10000,10),c='r', linestyle='--')
    plt.plot(np.linspace(posvalstd, posvalstd, 10), np.linspace(0,10000,10),c='r', linestyle='--')
    plt.show()




    # SCATTER PLOT, WITH AVGS AND ERRORBARS.    
    errorbins = np.linspace(np.amin(meandistances), np.amax(meandistances), 20)
    #print "errorbins = ", errorbins
    errorbinsidx = np.digitize(meandistances, bins=errorbins) #each point is assigned a bin index. should be same length as meandistances array.
    #print "errorbinsidx = ", errorbinsidx
    erroravgs = []
    errorstds = []
    for eb in range(0,len(errorbins)): #for each bin
        #print "eb = ", eb
        #print "np.where(errorbinsidx = eb) ", np.where(errorbinsidx == eb)
        ebincount = len(np.where(errorbinsidx == eb)) #return the number of points in each bin
        #print "ebincount = ", ebincount
        errors = value_differences[np.where(errorbinsidx == eb)] #np.where should return indices of the points in the given bin.
        #print "errors = ", errors
        avgerror = np.mean(errors)
        #print 'avgerror = ', avgerror
        errorstd = np.std(errors)
        #print "errorstd = ", errorstd
        erroravgs.append(avgerror)
        errorstds.append(errorstd)

    try:
        plt.scatter(meandistances, value_differences, s=8, color = value_differences/np.amax(np.abs(value_differences)), alpha=0.08)
    except:
        plt.scatter(meandistances, value_differences, s=8, alpha=0.08)
    plt.errorbar(errorbins, erroravgs, yerr=errorstds, fmt=' ', color='r')
    plt.scatter(errorbins, erroravgs, s=30, c='r')
    plt.xlabel('mean distance from point to interpolation points')
    plt.ylabel('value differences (actual minus interpolation-derived)')
    plt.ylim(-1,1)
    plt.grid()
    plt.title(str(dim_choice)+'D Error Function')
    plt.show()




    # NEAREST_DISTANCES SCATTER PLOT, WITH AVGS AND ERRORBARS.    
    errorbins = np.linspace(np.amin(nearest_distances), np.amax(nearest_distances), 20)
    #print "errorbins = ", errorbins
    errorbinsidx = np.digitize(nearest_distances, bins=errorbins) #each point is assigned a bin index. should be same length as meandistances array.
    #print "errorbinsidx = ", errorbinsidx
    erroravgs = []
    errorstds = []
    for eb in range(0,len(errorbins)): #for each bin
        #print "eb = ", eb
        #print "np.where(errorbinsidx = eb) ", np.where(errorbinsidx == eb)
        ebincount = len(np.where(errorbinsidx == eb)) #return the number of points in each bin
        #print "ebincount = ", ebincount
        errors = value_differences[np.where(errorbinsidx == eb)] #np.where should return indices of the points in the given bin.
        #print "errors = ", errors
        avgerror = np.mean(errors)
        #print 'avgerror = ', avgerror
        errorstd = np.std(errors)
        #print "errorstd = ", errorstd
        erroravgs.append(avgerror)
        errorstds.append(errorstd)

    try:
        plt.scatter(nearest_distances, value_differences, s=8, color = value_differences/np.amax(np.abs(value_differences)), alpha=0.08)
    except:
        plt.scatter(nearest_distances, value_differences, s=8, alpha=0.08)
    plt.errorbar(errorbins, erroravgs, yerr=errorstds, fmt=' ', color='r')
    plt.scatter(errorbins, erroravgs, s=30, c='r')
    plt.xlabel('distance from nearest point to interpolation points')
    plt.ylabel('value differences (actual minus interpolation-derived)')
    plt.ylim(-1,1)
    plt.grid()
    plt.title(str(dim_choice)+'D Error Function')
    plt.show()





    ### DAVID'S REQUESTED PLOT PER OCT 30 E-MAIL.
    plt.scatter(final_minus_true, final_minus_nearest, s=10, alpha=0.5, c=(nearest_distances/np.amax(nearest_distances)))
    plt.plot(np.linspace(0,2,10), np.linspace(0,2,10), c='k')
    plt.xlabel('| Derived Function Value - True Function Value |')
    plt.ylabel('| Derived Function Value - Nearest Neighbor Function Value |')
    plt.grid()
    #plt.xlim(0,2)
    #plt.ylim(0,2)
    #H, xedges, yedges = np.histogram2d(final_minus_nearest, final_minus_true, bins=400)
    #plt.colorbar()
    #plt.imshow(H, interpolation='none', origin='low')
    plt.title(str(dim_choice)+'D Interpolation Error')
    plt.xlim(0,2)
    plt.ylim(0,2)

    plt.show()
    
    
    #H, xedges, yedges = np.histogram2d(final_minus_nearest, final_minus_true, bins=(xedges, yedges), interpolation='none')
    #plt.imshow(H)
    #plt.show()
    
    
    #PLOT NEAREST DISTANCES HISTOGRAM
    n, bins, edges = plt.hist(nearest_distances, bins=50)
    plt.xlabel('Distance to Nearest Neighbor')
    #nearest_distance_stddev = np.std(nearest_distances)
    #ndnegstd = np.mean(n)-nearest_distance_stddev
    #ndposstd = np.mean(n)+nearest_distance_stddev
    #plt.plot(np.linspace(ndnegstd,ndnegstd,10), np.linspace(0,np.amax(n)+100,10), c='r', linestyle='--')
    #plt.plot(np.linspace(ndposstd,ndposstd,10), np.linspace(0,np.amax(n)+100,10), c='r', linestyle='--')
    plt.grid()
    plt.title('Nearest Neighbor Distances ('+str(len(col1))+' points)')
    plt.show()


    f = open('sample_points.txt', 'w')

    
    
    #WRITE THIS STUFF TO A FILE.
    for x,y,sc in zip(col2,col6,col8):
        f.write(str(x)+' '+str(y)+' '+str(sc)+' \n')

    f.close()


    t = open('interpolation_results.txt', 'w')
    for tx,ty,nnx,nny,nnv,dv,av in zip(point_x, point_y, nearest_x, nearest_y, nearest_values, interpvals, actvals):
        t.write(str(tx)+' '+str(ty)+' '+str(nnx)+' '+str(nny)+' '+str(nnv)+' '+str(dv)+' '+str(av)+' \n')

    t.close()

####### END SINUSOIDAL TEST FUNCTION #################        



    

def bigshow(dims):
    test_type = 's'
    xcoord = []
    ycoord = []
    zcoord = []
    normvals = []
    function_vals  = []
    nearestvals = []
    avgallvals = []
    for i in np.arange(0,len(col1),1): #creating nearly as many points as in the real sample.xs
    #for i in np.arange(0,50000,1):
        try:
            #print "round = ", i
            try:
                allvals = interp(pointgen(dims))
                xcoord.append(allvals[0]) #the x-coordinate of your test point
                ycoord.append(allvals[1]) #the y-coordinate of your test point
                if dim_choice == 2:
                    function_vals.append(allvals[2])
                    normvals.append(allvals[3]) #the function value divided by the maximum function value.
                    nearestvals.append(allvals[4])
                    avgallvals.append(allvals[5])
                elif dim_choice == 3:
                    zcoord.append(allvals[2])
                    function_vals.append(allvals[3])
                    normvals.append(allvals[4]) #the function value divided by the maximum function value.
                    nearestvals.append(allvals[5])
                    avgallvals.append(allvals[6])
            except:
                #print "looks like we couldn't find a coordinate."
                continue
        except KeyboardInterupt:
            break
    plt.scatter(xcoord,ycoord,c=normvals, alpha=0.5, s=50)
    plt.grid()
    plt.show()

    if dim_choice == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xcoord,ycoord,zcoord, c=normvals, alpha=0.3, s=20)
        plt.show()

    
    ## SHOWING THE DEVIATION FROM NEAREST NEIGHBOR INTERPOLATION -- A STRAIGHT DIAGONAL WOULD BE GOOD AGREEMENT. NOT THAT WE WANT GOOD AGREEMENT NECESSARILY.
    plt.scatter(nearestvals, function_vals, s=10, alpha=0.1, c='g')
    plt.plot(np.arange(np.amin(function_vals),np.amax(function_vals),1), np.arange(np.amin(function_vals), np.amax(function_vals), 1), c='k')
    plt.xlabel('Nearest Coordinate Function Value')
    plt.ylabel('Derived Value')
    plt.grid()
    plt.title('Deviation from Nearest Neighbor Interpolation')
    plt.show()
    
    #COULD ALSO PLOT DEVIATION FROM NEAREST NEIGHBOR AS A FUNCTION OF DISTANCE FROM THAT POINT.



def testshow():
    plt.scatter(col2,col6, c=(col8/np.amax(col8)), alpha=0.5, s=50)
    plt.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(col2,col6,col7, c=(col8/np.amax(col8)), alpha=0.3, s=20)
    plt.show()




