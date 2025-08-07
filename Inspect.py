#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:09:21 2025

@author: joaquin
"""
import os
import errno # for checking whether the montage direcotry already exists
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.filters import threshold_local, threshold_otsu, gaussian
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
import math #Square root
import csv #Save data as csv


##Functions

# Makes a new direcory for the given path, unless it alredy exists
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise   

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


def find_infected(amastigote_props, Nuclei_props, params, conditions):
    '''
    Calculates the distance between each amastigote's centroid and each nucleus' centroid. 
    Assigns each amastigote to its closest nucleus.
    
    Parameters
    ----------
    amastigote_props : Region properties for the labeled masks generated for amastigotes from the input image
    Nuclei_props : Region properties for the labeled masks generated for nuclei from the input image
    params: the number of variables used to identify each image
    conditions: the specific conditions for each image (the value of the variables used to identify it)
    
    Returns
    -------
    amastigotes_in_cell : List of the number of amastigotes for each nucleus (each value corresponds to one nucleus).
    sc_results_partial : The measurments for each cell in the fov.
    fov_row : The summary parameters for the fov.
    pairs: The coordinates of each amastigote, associated to its corresponding nucleus.
    

    '''
    sc_results_partial = [] #List to save the measurments for each cell in the image
    infected_cells = [] #Nuclei which were the closest to an amastigote are considered to be part of infected cells
    pairs = [] #List of lists. Each list contains the coordinates for an amastigote and its assigned cell
    for tc in range(len(amastigote_props)):
        
        amastigote_coord = amastigote_props[tc].centroid #(y, x)
        
        pair = [amastigote_coord] #This amastigote's coordinate
        
        yk = int(amastigote_coord[0]) #amastigote y coordinate
        xk = int(amastigote_coord[1]) #amastigote x coordinate
    
        distances = []
        for nuc in range(len(Nuclei_props)):
           Nuclei_coord = Nuclei_props[nuc].centroid #(y, x)
           yn = int(Nuclei_coord[0]) #Nucleus y coordinate
           xn = int(Nuclei_coord[1]) #Nucleus x coordinate       
        
           d = math.sqrt((xn - xk)**2 + (yn - yk)**2) #Distance between amastigote and nucleus coordinates
            
           distances.append(d)
    
        infected_cells.append(distances.index(min(distances))) #The host cell's coordinate
        
        pair.append(Nuclei_props[distances.index(min(distances))].centroid)
        pairs.append(pair)
    
    amastigotes_in_cell = [] #saves for the current image how many amasitogtes are in each cell (index)      
    for cell in range(0, len(Nuclei_props)):
        amastigotes = 0
        for j in range(len(infected_cells)):
            if infected_cells[j] == cell:
                amastigotes = amastigotes + 1
        amastigotes_in_cell.append(amastigotes)
        
        #Add each condition to the row
        sc_row = []
        for var in range(params):
            sc_row.append(conditions[var])
        sc_row.append(cell+1) #Add the current cell to the row
        sc_row.append(amastigotes) #Add the number of amastigotes in that cell
        
        sc_results_partial.append(sc_row) #Appending each cell's results one by one
        
    total_amastigotes = np.sum(amastigotes_in_cell)
    total_cells = len(amastigotes_in_cell)
    amastigotes_per_cell = total_amastigotes / total_cells
    infected = list(filter(lambda x: x > 0, amastigotes_in_cell))
    percent_infected = (len(infected)/len(amastigotes_in_cell)) *100
    amastigotes_per_infected = np.mean(infected)
    
    #Add each condition to the row
    fov_row = []
    for var in range(params):
        fov_row.append(conditions[var])
    fov_row.append(total_amastigotes)#Append each measurment to the row
    fov_row.append(total_cells)
    fov_row.append(amastigotes_per_cell)
    fov_row.append(percent_infected)
    fov_row.append(amastigotes_per_infected)
    
    return amastigotes_in_cell, sc_results_partial, fov_row, pairs, percent_infected, amastigotes_per_infected

def summary_plot(name, image, masks_Tc, masks_Nuc, pairs, percent_infected, amastigotes_per_infected, amastigotes_in_cell):
    '''
    Generates a figure (2x2) with the original image, the image overlayed with the amastigote or nuclear masks 
    and lines connceting the amastigotes to their assigned nucleus.
    Includes the number total number of amastigotes and nuclei that were detected.
    Shows the % of infected cells and the average amastigotes per infected cell.     

    Parameters
    ----------
    image : original image used as input for the analysis.
    masks_Tc : The amastigotes that were masked in that image.
    masks_Nuc :The masks for the nuclei.
    pairs : A list of each amastigote's coordinate associated with its host cell's coordinate.
    percent_infected : The calculated infection rate.
    amastigotes_per_infected : The value for the average amasitogtes in infected cells.
    amastigotes_in_cell : List of the number of amastigotes for each nucleus (each value corresponds to one nucleus).

    Returns
    -------
    fig : A figure with 4 panels.

    '''
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0][0].imshow(image, cmap="gray")
    ax[0][0].axis('off')
    ax[0][0].set_title('%s' %(name))

    ax[0][1].imshow(masks_Tc)
    ax[0][1].axis('off')
    ax[0][1].set_title('Amastigote masks: %s' %(np.max(masks_Tc)))

    ax[1][0].imshow(masks_Nuc)
    ax[1][0].axis('off')
    ax[1][0].set_title('Nuclear masks: %s' %(np.max(masks_Nuc)))

    #Plot lines between each amastigote and each nucleus
    for pair in pairs:
        xs=(pair[0][1], pair[1][1])
        ys=(pair[0][0], pair[1][0])
        plt.plot(xs, ys, color="white", linewidth = 0.3)

    #Add total number of amastigotes assigned to each nucleus
    Nuclei_props = regionprops(masks_Nuc)
    for cell in range(len(amastigotes_in_cell)):
        xtext = Nuclei_props[cell].centroid[::-1][0]
        ytext = Nuclei_props[cell].centroid[::-1][1]
        plt.text(xtext, ytext, amastigotes_in_cell[cell], color = "red", size = 5)
              
    ax[1][1].set_title('Infected: %s%% \n %s per infected cell' %(round(percent_infected, 2), round(amastigotes_per_infected, 2)))
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    
    return fig



#####
# Wroking direcotry (where program is saved)
wd = '/home/joaquin/Desktop/Poster 2025/20250710 - THP1 y HeLa - Tul vs Dm'
os.chdir(wd)

# Montage direcotry
figuredir = os.path.join(wd, "INsPECT plots")

make_sure_path_exists(figuredir)


# Folder where all the images are stored, each within a direcotry for all the images in a field
main_folder = '/home/joaquin/Desktop/Poster 2025/20250710 - THP1 y HeLa - Tul vs Dm/Fotos'

# List of each folder containing the images for the fields of view
fields = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]


imgs = [] #Lists of all the images (numpy arrays)
for fov in range(len(fields)):
    #List of all images for a field
    campo = os.path.join(main_folder, fields[fov])
    fotos =  [f for f in os.listdir(campo) if os.path.isfile(os.path.join(campo, f))] # Excludes Metadatafolder
    fotos.sort()
    
    dapi = plt.imread(os.path.join(campo, fotos[0])) 
    imgs.append(dapi)

#Columns for the output table of Average FOV measurments
params = 3
#List of lists, each row corresponds to the measurments for a field of view
sc_head = ["Cepa", "Replica","Campo", "Célula", "Amastigotes"]

fov_head = ["Cepa", "Replica", "Campo", "Amastigotes", "Células", 
            "Amas por célula", "Porcentaje", "Amas por célula infectada" ]

# First row is the name of the columns
sc_results = [sc_head] 
fov_results = [fov_head]

for im in range(len(imgs)):  

    #image_conditions = fields.split("-")
    conditions = fields[im].split("-")
    
    img = imgs[im]    
    
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

    #Watershed to separarate nuclei that are touching
    distance = ndi.distance_transform_edt(closed_nuc)
    coords = peak_local_max(distance, footprint=footprint, labels=closed_nuc, min_distance = 20) #min _distance can be different dependeing con cell line or aquisition parameters
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labeled_nuc = watershed(-distance, markers, mask=closed_nuc) #Final labeled cells
    labeled_nuc = morph.remove_small_objects(labeled_nuc, min_size=50)
    labeled_nuc = label(labeled_nuc)
    
    
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
    
    
    ###
    Nuclei_props = regionprops(labeled_nuc)
    amastigote_props = regionprops(labeled_kineto)
    
    #Measure the distance of each amastigote to each nucleus, then save which nucleus is the closest to each amastigote
    amastigotes_in_cell, sc_results_partial, fov_row, pairs, percent_infected, amastigotes_per_infected = find_infected(amastigote_props, Nuclei_props, params, conditions)
    
    #Save current image's measurments
    fov_results.append(fov_row)
    sc_results = sc_results + sc_results_partial
    
    #Plot
    fig_name = "%s.png" % (fields[im])  
    fig = summary_plot(fields[im], imgs[im], labeled_kineto, labeled_nuc, pairs, percent_infected, amastigotes_per_infected, amastigotes_in_cell)
    plt.savefig(os.path.join(figuredir, fig_name), bbox_inches='tight', dpi = 300) #Saves the plot
    plt.close() #Closes the plot
    
    print("Done with %s" % (fields[im]))

#If the output file is not in the program's folder, try at C:/Users/Usuario
with open(os.path.join(wd,'sc_INsPECT.csv'), 'w', newline='') as f: #Measurements for each field's average value
    writer = csv.writer(f)
    writer.writerows(sc_results)
    
with open(os.path.join(wd,'fov_INsPECT.csv'), 'w', newline='') as f: #Measurements for each field's average value
    writer = csv.writer(f)
    writer.writerows(fov_results)
    
