# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:11:04 2025

@author: Usuario
"""
# Masks vs True positive

import os #For importing images and saving figures
import errno # for checking whether the montage direcotry already exists
import matplotlib.pyplot as plt
import numpy as np

from skimage import filters, morphology


import csv #Save data as csv

#Functions

def set_kernel(img):
    '''
    Parameters
    ----------
    img : The image to be processed.

    Returns
    -------
    footprint : square structuring element based on the image's noise.

    '''
    global_sd = np.std(img)
    kernel_size = int(1.5*global_sd)
    
    #Kernel size ahs to be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
 
    footprint = np.ones((kernel_size, kernel_size), dtype=bool) 
    return footprint


def small_to_background(foreground, background, connectivity=5):
    '''
    Parameters
    ----------
    foreground : Binary image of the pixels above a threshold.
    background : Binary image of the pixels below a threshold.
    connectivity : maximum number of background pixels a foreground pixel can be in contact with.
    
    Returns
    -------
    fore : foreground pixels that were connected to more than the set amount of background pixels are set as background.
    back : updated background based on connectivity.

    '''
    fore = foreground.copy()
    back = background.copy()
    
    row_offsets = [-1, -1, -1, 0, 0, 1, 1, 1]
    col_offsets = [-1, 0, 1, -1, 1, -1, 0, 1]
    shape = fore.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            flag = 0
            for p in range(8):
                neighbour_row = i + row_offsets[p]
                neighbour_col = j + col_offsets[p]
                try:
                    value = foreground[neighbour_row, neighbour_col]
                except IndexError: # catch the error
                    pass # pass will basically ignore it
                    # and execution will continue on to whatever comes
                    # after the try/except block
                if not value:
                    flag += 1
            if flag >= connectivity:
                fore[i,j] = False
                back[i,j] = True
   
    return fore, back


# Makes a new direcory for the given path, unless it alredy exists
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise   
            
def Percent_True(num_detcted, num_corrected):
    if num_detcted > 0:
        if num_detcted > num_corrected:
            true = round((num_corrected / num_detcted) * 100, 2)
        else: 
            true = 100
    else:
        true = "None detected"
    return true
     
def summary_plot(name, DAPI, Tc, masks_Tc, Tc_mask, True_masks, threshold, coverage):
    '''
    Generates a figure (2x2) with the original image, the image overlayed with the amastigote or nuclear masks 
    and lines connceting the amastigotes to their assigned nucleus.
    Includes the number total number of amastigotes and nuclei that were detected.
    Shows the % of infected cells and the average amastigotes per infected cell.     

    Parameters
    ----------
    name: image name with all the conditions
    DAPI : Nuclear stain image used as input for the model.
    Tc: T. cruzi stain image.
    masks_Tc : The amastigotes that were masked in by the neural net in the image.
    Tc_mask : Thresholded image for the Tc stain.
    True_masks : The amastigote masks multiplied by the Tc stain binary image.

    Returns
    -------
    fig : A figure with 4 panels.

    '''
    
    true = Percent_True(np.max(masks_Tc), np.max(True_masks))
    
    fig, ax = plt.subplots(nrows=3, ncols=2)
    
    plt.rcParams['axes.titlesize'] = 5
    
    ax[0][0].imshow(DAPI, cmap = 'gray')
    ax[0][0].axis('off')
    ax[0][0].set_title('%s' %(name))
    
    ax[0][1].imshow(mask_overlay(DAPI, masks_Tc))
    ax[0][1].axis('off')
    ax[0][1].set_title('Overlay')
    
    ax[1][0].imshow(masks_Tc)
    ax[1][0].axis('off')
    ax[1][0].set_title('Detected amastigotes: %s' %(np.max(masks_Tc)))
    
    ax[1][1].imshow(Tc_mask)
    ax[1][1].axis('off')
    ax[1][1].set_title('Threshold anti-Tc (%s)' %(round(threshold, 2)))

    ax[2][0].imshow(True_masks)
    ax[2][0].axis('off')
    ax[2][0].set_title('True masks: %s' %(np.max(True_masks)))
    
    ax[2][1].imshow(mask_overlay(Tc, True_masks))
    ax[2][1].axis('off')
    ax[2][1].set_title('Percent true: %s. Coverage: %s' %(true, coverage))
    
    plt.subplots_adjust(top=1.1)
    plt.subplots_adjust(wspace=-0.6, hspace=0.2)
    
    return fig

#####
# Wroking direcotry (where program is saved)
wd = r'C:\Users\Usuario\Desktop\Joaquin\Test\2050710 - HeLa C Tul Dm'
os.chdir(wd)

# Montage direcotry
figuredir = os.path.join(wd, "Figures_True_Positive_Inspect")

make_sure_path_exists(figuredir)


# Folder where all the images are stored, each within a direcotry for all the images in a field
main_folder = r'C:\Users\Usuario\Desktop\Joaquin\Test\2050710 - HeLa C Tul Dm/Fotos'

# List of each folder containing the images for the fields of view
fields = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]


DAPI = [] #Lists of all the images (numpy arrays)
Tc =[]
for fov in range(len(fields)):
    #List of all images for a field
    campo = os.path.join(main_folder, fields[fov])
    fotos =  [f for f in os.listdir(campo) if os.path.isfile(os.path.join(campo, f))] # Excludes Metadatafolder
    fotos.sort()
    
    img = plt.imread(os.path.join(campo, fotos[0])) #Load DNA stains 
    DAPI.append(img)
    img = plt.imread(os.path.join(campo, fotos[1])) #Load Tc stains
    Tc.append(img)



#Columns for the output table of Average FOV measurments
#List of lists, each row corresponds to the measurments for a field of view

head = ["Cepa", "Linea", "Campo", "Amastigotes_totales", "Amstigotes_Reales", 
            "Amastigotes_Falsos", "Porcentaje", "Cobertura" ]

# First row is the name of the columns
results = [head]

'''
thresholds = []
for tc in range(len(Tc)):
    
    Tc_smooth = filters.gaussian(Tc[tc])
    Tc_Threshold = filters.threshold_otsu(Tc_smooth)
    thresholds.append(Tc_Threshold)

Tc_Threshold = np.mean(thresholds)
'''

import skimage.morphology as morph
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

masks_Tc = []
for im in range(len(DAPI)):  

    #image_conditions = fields.split("-")
    conditions = fields[im].split("-")
    
    img = DAPI[im]    
    
    ##Cell pipleine
    #Structuring element for nuclei
    footprint = set_kernel(img)

    blur = ndi.median_filter(img, size=15) #size 15 determined in paper
    threshold = threshold_otsu(blur)
    #threshold = threshold_local(blur, block_size=9, method='median', offset=0, mode='reflect', param=None, cval=0)
    binary = blur > threshold
    no_holes = binary_fill_holes(binary)
    closed_nuc = no_holes.copy()
    #closed_nuc = morph.closing(no_holes, footprint=footprint) #Final Nuclear mask

    
    ###Parasite pipeline
    #footprint_p = np.array([[1,1],[1,1]]) #Structuring element for parasites
    footprint_p = morph.disk(2) #disk works better than square
    
    #Invert the image
    invert = 255 - img 
    
    #Black top transform
    blurred = gaussian(invert) #Blurring to remove the noise, details are extracted better
    black_tophat = morph.black_tophat(blurred, footprint=footprint_p) #Small elements as brighter pixels (kinetoplast becomes more noticeable)
    
    #Substract the segmentation from the cell pipeline
    no_cell = black_tophat.copy()
    no_cell[closed_nuc] = 0 #removes pixels already assigned to cells
    
    #Setting the threshold to binarize tophat transform
    no_cell_mean = np.mean(no_cell)
    no_cell_sd = np.std(no_cell)    
    threshold_kineto = no_cell_mean + 4*no_cell_sd #3 standard deviations: top 0.27% of pixels from blackhat
    
    #Binarized images
    background = no_cell <= threshold_kineto
    foreground = no_cell > threshold_kineto #Image that will be used to detect amastigotes
    
    foreground_clean, background_2 = small_to_background(foreground, background) #Remove sparsly connected pixels from forgound and send them to background
                
    no_small = morph.remove_small_objects(foreground_clean, min_size = 12) #Remove small objects 
    labeled_kineto = label(no_small) #Labeled kinetoplasts
    
    masks_Tc.append(labeled_kineto)


from cellpose.plot import mask_overlay
for im in range(len(Tc)):  

    #image_conditions = fields.split("-")
    conditions = fields[im].split("-")
    
    Tc_smooth = filters.gaussian(Tc[im])
    Tc_Threshold = filters.threshold_otsu(Tc_smooth)
    Tc_mask = Tc_smooth > Tc_Threshold        
    
    Detected_Tc = int(np.max(masks_Tc[im]))
    
    True_masks = morphology.label(masks_Tc[im] * Tc_mask)
        
    True_detected = int(np.max(True_masks))    
    
    true = Percent_True(Detected_Tc, True_detected)
    
    true_total = True_masks > 0 
    coverage = round((np.sum(true_total) / np.sum(Tc_mask)) *100, 2 )
    
    row = [conditions[0], conditions[1], conditions[2], Detected_Tc, True_detected, Detected_Tc-True_detected, true, coverage]

    #Save current image's measurments
    results.append(row)
    
    
    #Plot
    fig_name = "%s.png" % (fields[im])  
    fig = summary_plot(fields[im], DAPI[im], Tc[im], masks_Tc[im], Tc_mask, True_masks, Tc_Threshold, coverage)
    plt.savefig(os.path.join(figuredir, fig_name), bbox_inches='tight', dpi = 300) #Saves the plot
    plt.close() #Closes the plot

#If the output file is not in the program's folder, try at C:/Users/Usuario
with open(os.path.join(wd,'truth_results_Inspect.csv'), 'w', newline='') as f: #Measurements for each field's average value
    writer = csv.writer(f)
    writer.writerows(results)
