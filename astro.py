# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 14:08:10 2016

@author: jdc14
"""

from astropy.io import fits
import numpy as np
from scipy import stats #Used to find the mode. stats.mode(nostar)
import scipy.optimize #Linear curve fit of galaxy numbers.
import matplotlib.pyplot as plt
import csv
import time #used to uniquely name the catalogue files.

def scale_lower(pic, dimmer=3600): 
    """Using for loop to replace bright pixels with dimmer value.
    Input:
        pic - numpy.array(), the image to be dimmed
        dimmer - int, how low should the "lowpass cutoff" be.
    """
    #most_common = stats.mode(pic)
    for rowno in range(pic.shape[0]):
        for colno in range(pic.shape[1]):
            if pic[rowno][colno] > dimmer:
                pic[rowno][colno] = dimmer

    return pic

def aperture_photometry(radius, centrex, centrey, pic):
    """
    This function counts the total intensity within a circle,
    then finds the average background for the annulus surrounding that circle,
    and subtracts the background off to give a corrected intensity.
    Note that for extended objects, a larger radius might have a greater corrected intensity.
    
    Inputs,
        radius - int, how large a circle of the image do we want to measure?
        centrex - int, ordinate of centre of circle and annulus.
        centrey - int, abscissa of centre of circle and annulus.
        pic - 2d numpy masked array.
    
    Returns,
        correct_intensity, - int, how bright is the galaxy?
        circle_background -int, how much brightness correction was subtracted
                                to give the corrected intensity count?
    """
    
    def magnitude(intensity, int_sd, magzpt, magzrr):
        """
        Returns the magnitudes of every element of the input.
        Inputs:
            intensity - numpy masked array.
            int_sd - standard deviation of number of counts.
            magzpt - float, calibration magnitude.
            magzrr - float, magnitude zero error.
        
        Returns:
            mag - output magnitude masked array.
            mag_sd - magnitude standard deviation array.
        """
        
        mag_i = -2.5 * np.log10(intensity)
        err_i = np.absolute((2.5*int_sd)/(intensity*np.log(10)))
        
        mag = magzpt + mag_i
        mag_sd = np.sqrt(magzrr*magzrr + err_i*err_i)
                
        return mag, mag_sd
    
    assert type(pic) == np.ma.core.MaskedArray
    
    rows = [] #These two lists are used with advanced array slicing
    cols = [] #to pick out the bit of the masked array we want.
    
    inner_ring = []
    
    outer_ring_rows = []
    outer_ring_cols = []
    #This approach ensures we don't include masked values in our calculations.
    
    smaller_circle_coords = ((x,y) for x, y in circles(radius, centrex, centrey) \
              if 0 <= x < pic.shape[1] if 0<= y < pic.shape[0])
    
    bigger_circle_coords =  ((x,y) for x, y in circles(radius * 2, centrex, centrey) \
                       if 0 <= x < pic.shape[1] if 0<= y < pic.shape[0])
    
    for x, y in smaller_circle_coords:
        rows.append(y)
        cols.append(x)
        inner_ring.append((x,y))
    
    for x,y in bigger_circle_coords:
        if (x,y) not in inner_ring:
            outer_ring_rows.append(y)
            outer_ring_cols.append(x)
    
    #Now, we have build our advanced slicing list indices.
    #We can make a masked array view with only the pixels we want 
    #using pic[rows, cols]
    circle_array = pic[rows, cols]
    annulus_array = pic[outer_ring_rows, outer_ring_cols]
    
    total_count = circle_array.sum()
    background_per_pixel = annulus_array.mean()
    circle_background = annulus_array.mean() * circle_array.count()
    correct_intensity = total_count - circle_background 
    
    #If we assume the only source of noise in our counts is shot noise.
    #Ignore Bias structure.
    #Ignore dark current.
    #Ignore flat fielding.
    #Ignore interpolation errors.
    #Ignore Charge transfer efficiency problems.
    #Ignore charge skimming.
    #Ignore shutter vignetting.
    circle_variances = circle_array.copy() #Variance = mean (mu) for poisson data.
    overall_variance = circle_variances.sum() #not including annular background correction errors yet.
    
    #How much error is in our background annulus?
    background_variances = annulus_array.copy()
    background_combined_variance = background_variances.sum()/(background_variances.count()*background_variances.count())
    circle_background_variance = circle_array.count() * background_combined_variance
    background_sd = np.sqrt(circle_background_variance) #standard deviation of circle bg correction.
    
    correct_variance = overall_variance + circle_background_variance
    correct_sd = np.sqrt(correct_variance)
     
    magzpt = hdulist[0].header['MAGZPT']
    magzrr = hdulist[0].header['MAGZRR']
    
    mag, mag_sd = magnitude(correct_intensity, correct_sd, magzpt, magzrr)

    return correct_intensity, correct_sd, circle_background, background_sd, mag, mag_sd #This result scales with aperture size ^2. 

def cover_box(xmin, xmax, ymin, ymax, pic):
    """
    Paints a rectangle of median value over the image
    from xmin (inclusive) to xmax (exclusive) and
    from ymin (inclusive) to ymax (exclusive).
    """
    #First, Sanitise our inputs.
    assert type(pic) == np.ma.core.MaskedArray
    if xmin < 0:
        xmin = 0
    if xmax > pic.shape[1]:
        xmax = pic.shape[1]
    if ymin < 0:
        ymin = 0
    if ymax > pic.shape[0]:
        ymax = pic.shape[0]
    
    new = pic.copy()
     
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            
            new[j][i] = np.ma.masked
    return new

def cover_circle(radius, centrex, centrey, pic):
    """
    Paints a circle of median value over the image.
    This function should be used for preprocessing.
    radius - int, size of circle to paint median.
    """
    assert type(pic) == np.ma.core.MaskedArray
    new = pic.copy()
    coords = ((x,y) for x, y in circles(radius, centrex, centrey) if 0 <= x < pic.shape[1] if 0<= y < pic.shape[0])
    for i in coords:
        x, y = i
        new.mask[y][x] = True
    return new

def circles(radius, centrex, centrey):
    """
    Generates a list of co-ordinates which lie 
    within a certain distance from a point.
    
    radius - int. How big
    centre - 2-element tuple/list, (centre_x, centre_y)
    """
    for width in range(-1* radius, radius+1):
        y_diff = int(round(np.sqrt(radius * radius - width * width),0))
        for height in range(-1 * y_diff, y_diff + 1):
            yield((centrex + width,centrey+height))

def mask_to_cat(pic, aperture = 6, out = None):
    """
    Take the brightest galaxy remaining in the image,
    add the row number, column number, intensity, and background 
    to the dictionary.
    return pic with that galaxy masked by circle.

    Inputs:
        pic - 2d array, image of sky.
        aperture - int, how big should the mask circle radius be?
        output - dict, where do we put this galaxy metadata?
    
    Returns:
        mask pic - 2d array, picture with the galaxy masked.
        out - dict, information about one galaxy.
    """
    assert type(pic) == np.ma.core.MaskedArray
    
    if out == None: # This stops the problem with default mutable arguments
        out = {}    # mutating between function calls.
    gal_rowcol = np.unravel_index(pic.argmax(),pic.shape)
    #flatten array, find flat coord of max value, flat coord -> pic coord. 
    gal_int, gal_int_sd, gal_bg, gal_bg_sd, mag, mag_sd = aperture_photometry(aperture, gal_rowcol[1], gal_rowcol[0], pic)
    
    
    
    out['xy-coords'] = (gal_rowcol[1],gal_rowcol[0])
    out['intensity'] = gal_int
    out['intensity standard deviation'] = gal_int_sd
    out['background'] = gal_bg
    out['background standard deviation'] = gal_bg_sd
    out['magnitude'] = mag
    out['magnitude standard deviation'] = mag_sd
    mask_pic = cover_circle(aperture, gal_rowcol[1], gal_rowcol[0], pic)
    return mask_pic, out

def build_catalogue(pic, aperture = 6, fmin = 3500):
    """
    Uses mask to cat to get store all the galaxies in the sky in a list.
    
    Inputs:
        pic - 2d masked array, image of sky.
        aperture - int, optional, how big should the mask circle radius be?
        fmin - int, optional, how bright should the brightest unmasked galaxy be?
    
    Returns:
        galaxies, list of dicts, information about lots of galaxies.
        new_pic, picture with the galaxy masked.
    """
    assert type(pic) == np.ma.core.MaskedArray
    
    galaxies = []
    new_pic = pic.copy()
    former = new_pic.max() + 1
    current = new_pic.max()
    while new_pic.max() > fmin:
        if current < former:
            print ("max brightness at {} \n".format(current))
        former = new_pic.max()
        new_pic, gal_data = mask_to_cat(new_pic, aperture, None)
        current = new_pic.max()
        galaxies.append(gal_data)
    return galaxies, new_pic
    
def for_catalogue(pic, aperture = 6, how_many = 2000):
    """
    Builds a catalogue of the n brightest galaxies using a for loop.
    
    Inputs:
        pic - 2d masked array.
        aperture - how big should the masked circle radius be?
        
    Returns:
        galaxies - list of dicts of galaxy information.
            len(galaxies) returns the number of galaxies.
        new_pic - pic with circle masked
    """
    assert type(pic) == np.ma.core.MaskedArray
    
    galaxies = []
    new_pic = pic.copy()
    for i in range(how_many):
        print(i)
        new_pic, gal_data = mask_to_cat(new_pic, aperture, None)
        galaxies.append(gal_data)
    return galaxies, new_pic

def final(galaxies):
    """
    Returns the data needed to make a plot of number of galaxies brighter than a
    given magnitude.
    
    Input:
        galaxies - list of dictionaries, dictionaries have value called intensity.
    
    Returns:
        x, y, y_err, linear_fit_params
    """
    def number_of_gal_less_than_m(magnitudes, magnitudes_sd, benchmark):
        """
        Returns the humber of things less than a benchmark.
        Thing error is calculated by bootstrapping.
        
        Input:
            magnitudes - numpy 1d array of galaxy magnitudes.
            magnitudes_sd - numpy 1d array of galaxy standard deviations.
        
        Returns:
            how_many_things - numpy 1d array, number of things are less than benchmark.  
            thing_error - numpy 1d array, how big is the standard deviation on the number.
        """
        magplus1sigma = magnitudes + magnitudes_sd
        magminus1sigma = magnitudes - magnitudes_sd
        
        how_many_things_pessimistic = np.sum(magplus1sigma < benchmark)
        number_of_things = np.sum(magnitudes < benchmark)
        how_many_things_optimistic =  np.sum(magminus1sigma < benchmark)
        
        error_up = how_many_things_optimistic - number_of_things
        error_down = number_of_things - how_many_things_pessimistic
        
        #Unfortunately our curve fitting doesn't work with asymmetric error bars.
        #We combine the two by just finding the mean of the upper and lower bar.
        thing_error = np.maximum(1e-2, (np.absolute(error_up) + np.absolute(error_down)))
        
        return number_of_things, thing_error
       
    magnitudes = np.fromiter((i['magnitude'] for i in galaxies\
                              if i['magnitude'] != 0 \
                              and i['magnitude'] != np.inf),dtype = float)
    
    magnitudes_sd = np.fromiter((i['magnitude standard deviation'] for i in galaxies\
                                 if i['magnitude standard deviation'] != 0\
                                 and i['magnitude standard deviation'] != np.inf), dtype = float)
    
    x = np.arange(0,30,0.01)
    y_list =[]
    y_err_list = []
    
    for m in x:
        num, error = number_of_gal_less_than_m(magnitudes, magnitudes_sd, m)
        y_list.append(num)
        y_err_list.append(error)
        
    y = np.array(y_list)
    y_error = np.array(y_err_list)
    
    return x, y, y_error
    
def line(x, a, b):
    """
    literally a*x+b
    Inputs:
        x, float, independant variable.
        a, float, gradient
        b, float, intercept.
    Returns:
        y, float, height of the line
    """
    return a*x + b
    
def linear_fit(xdata, ydata, y_error):
    """
    Fits a straight line to a segment of the data.
    """
    f = line
    popt, pcov = scipy.optimize.curve_fit(f, xdata, ydata, p0=(0.38, -3.3), sigma = y_error, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov)) #Parameter standard deviation.

    return popt, perr

def prettify_error_data(error_data, zeroval = 0.01):
    """
    fills every 'zero' value with the value of the most recent non-zero value to the left. This copies it's neighbours.

    Inputs:
    error_data = np.array 1-d. Your errors, 

    Returns:
    pretty_error_data = np.array 1-d. Your prettier  plaigarised neighbourr errors
    """
    pretty_error_data = error_data.copy()
    most_recent = error_data[error_data != zeroval][0]
    had_nonzero_yet = False
    
    for error_idx in range(len(error_data)):
        if error_data[error_idx] == zeroval:    
            pretty_error_data[error_idx] = most_recent
        elif had_nonzero_yet and error_data[error_idx] == 0:
            pretty_error_data[error_idx] = most_recent
        else:
            most_recent = error_data[error_idx]
            had_nonzero_yet = True

    return pretty_error_data
        
"***************************main code***********************************"
if __name__ == "__main__":
    radius = 6
    fmin = 3500
    filestr = time.strftime("%Y%m%d-%H%M%S") \
                + 'r' + str(radius) + 'f' + str(fmin) + '.csv'
    
    #The .csv file name, filestr, contains the time, radius, minimum flux.
    
    with open(filestr, 'w') as f: #This .csv file will become our catalogue.
        #open and slice the fits data.
        hdulist = fits.open("C:\\Users\Joaquin\Documents\Year 3\Astronomical Image Processing\A1_mosaic.fits")
    
        pic = np.array(hdulist[0].data)
        pic = pic[890:4310,370:2150] #this is a better slice. Grabs a reasonable section.
        pic_mask = np.zeros(pic.shape, dtype = 'bool')
        pic = np.ma.array(pic, mask = pic_mask)
        
        #Preprocessing.
        print('Beginning preprocessing')
        new = pic.copy()
        #new = scale_lower(new)
        print('Removing artifacts.')
        remove_starline = cover_box(1050,1080,0,3419,new)
        nostar = cover_circle(250,1070,2320, remove_starline)
        nostar2 = cover_box(150,230,3180,3230,nostar)
        nostar3 = cover_box(350,470,2305,2535,nostar2)
        nostar4 = cover_box(480,600,1330,1470,nostar3)
        print('preprocessing 50% complete')
        nostar5 = cover_box(550,660,1800,1950,nostar4)
        nostar6 = cover_box(1690,1750,510,565,nostar5)
        nostar7 = cover_box(1710,1779,1380,1450,nostar6)
        nostar8 = cover_box(1710,1779,2815,2915,nostar7)
        
        #display image
        #plt.colorbar(plt.imshow(nostar8, origin = 'lower', cmap = 'gray'))
        #plt.yscale('log', nonposy='clip')
        #plt.hist(nostar)
        ##################Write to CSV#######################
        print('Building Catalogue')
        #galaxies, nostar9 = build_catalogue(nostar8, radius, fmin) 
        fieldnames = ['xy-coords','intensity','intensity standard deviation','background','background standard deviation','magnitude','magnitude standard deviation']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        
        writer.writeheader()
        writer.writerows(galaxies)
        
        #plt.colorbar(plt.imshow(nostar9, origin = 'lower', cmap = 'gray'))
       
        ####Plot Number of galaxies brighter than magnitude m########
        x_data, y_data, y_error = final(galaxies)
        #plt.errorbar(x_data, y_data, yerr = y_error)
        pretty_y_error = prettify_error_data(y_error)
        
        log_y_data = np.maximum(0, np.log10(y_data))
        log_y_error = np.abs(y_error/(y_data*np.log(10)))
        pretty_log_y_error = np.abs(pretty_y_error/(y_data*np.log(10)))
    
        popt, perr = linear_fit(x_data[960:1800], log_y_data[960:1800], log_y_error[960:1800]) 
        y_fit_list = []
        for x in x_data:
            y_fit_list.append(line(x, popt[0],popt[1]))
        
        plt.errorbar(x_data, log_y_data, log_y_error)    
        plt.plot(x_data[960:1800], y_fit_list[960:1800], linewidth = 2.5)
        
        
        new_popt, new_perr = linear_fit(x_data[960:1800], log_y_data[960:1800], pretty_log_y_error[960:1800]) 
        y_fit_list = []
        for x in x_data:
            y_fit_list.append(line(x, new_popt[0],new_popt[1]))
        
        plt.errorbar(x_data, log_y_data, pretty_log_y_error)    
        plt.plot(x_data[960:1800], y_fit_list[960:1800], linewidth = 2.5)
        
        plt.xlabel('Magnitude m')
        plt.ylabel('Logarithm of number of galaxies with magnitude less than m')
        
        plt.title('Galaxy number counts')
