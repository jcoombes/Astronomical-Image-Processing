# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 14:08:10 2016

@author: jdc14
"""

from astropy.io import fits
import numpy as np
from scipy import stats #Used to find the mode. stats.mode(nostar)
import matplotlib.pyplot as plt
import csv
import time #used to uniquely name the catalogue files.


def scale_lower(pic, dimmer=3600): 
    """Using for loop to replace bright pixels with dimmer value.
    Input:
        pic - numpy.array(), the image to be dimmed
        dimmer - int, how low should the "lowpass cutoff" be.
        
    Do I want to subtract off the Background count (=min(pic.flat)) before I replace with zero???!?!?!?!"""
    #most_common = stats.mode(pic)
    for rowno in range(pic.shape[0]):
        for colno in range(pic.shape[1]):
            if pic[rowno][colno] > dimmer:
                pic[rowno][colno] = dimmer

    return pic

#def bg_annulus(pic,radius,centrex,centrey,lower_limit_sigma=0.1):
#
#    mean = 3423.78
#    lower_limit_galaxy_brightness = mean+lower_limit_sigma*13.5
#    
#    small_circle_list=[]
#    large_circle_list=[]
#    annular_list=[]
#    small_circle= circles(radius,centrex,centrey)
#    large_circle= circles(2*radius,centrex,centrey)
#    for i in small_circle:
#        small_circle_list.append(i)
#    for i in large_circle:
#        large_circle_list.append(i)
#    for i in large_circle_list:
#        if i not in small_circle_list:
#            annular_list.append(i)
#    
#    local_background_pixel_values = []
#    pixel_values_galaxy_var= []
#
#    for i in annular_list:
#        if pic[i[1]][i[0]] >= lower_limit_galaxy_brightness:
#            local_background_pixel_values.append(pic[i[1]][i[0]])
#        else:
#            local_background_pixel_values.append(mean)
#    for i in small_circle_list:
#        pixel_values_galaxy_var.append(pic[i[1]][i[0]])
#    
#    mean_local_background = sum(local_background_pixel_values)/len(local_background_pixel_values)
#    
#    return mean_local_background      

def bg_annulus(pic, radius, centrex, centrey, lower_limit_sigma = 0.1):
    """
    Returns the 
    We expect annulus to be the diameter.
    """
    return None

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
    gal_int = pic.max()
    gal_rowcol = np.unravel_index(pic.argmax(),pic.shape)
    #flatten array, find flat coord of max value, flat coord -> pic coord. 
    gal_bg = bg_annulus(pic,aperture,gal_rowcol[1],gal_rowcol[0])
    
    out['rowcol-coords'] = gal_rowcol
    out['intensity'] = gal_int
    out['background'] = gal_bg
    mask_pic = cover_circle(aperture, gal_rowcol[1], gal_rowcol[0], pic)
    return mask_pic, out

def build_catalogue(pic, aperture = 6, fmin = 3500):
    """
    Uses mask to cat to get store all the galaxies in the sky in a ditionary.
    
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
  
def magnitude(pic,f2):
    """
    Inputs:
        pic - numpy array of brightness values.
        f2 - magnitude zero point, magzpt from fits header data.
        
    Outputs:
        numpy array of magnitude values.
    """    
    return -2.5 * np.log10(pic/f2)
    
    
"""
#Using list comprehensions to filter array elements of pictrue > 36,000
#Replace bright pixels with zero brightness
newpic = np.empty((4611, 2570), int)
for rowno in range(pic.shape[0]):
    row = [x if x < 36000 else 0 for x in pic[rowno]]
    newpic[rowno] = np.asarray(row)
"""

"***************************main code***********************************"
if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%H%M%S") + '.csv'
    with open(timestr, 'w') as f: #This .csv file will become our catalogue.
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
        print('rescaled. Removing artifacts.')
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
        
        #plt.yscale('linear', nonposy='clip')
        #plt.yscale('log', nonposy='clip')
        #plt.hist(nostar)
        ##################Write to CSV#######################
        print('Building Catalogue')
        galaxies, nostar9 = build_catalogue(nostar8, 6, fmin = 3440) 
        fieldnames = ['rowcol-coords','intensity','background']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        
        writer.writeheader()
        writer.writerows(galaxies)
        
        plt.colorbar(plt.imshow(nostar9, origin = 'lower', cmap = 'gray'))
