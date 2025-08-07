# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 18:09:39 2025

@author: Usuario
"""

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

from cellpose import models, io #Cellpose ML model for single cell segmentation
from cellpose.io import imread 
from cellpose.plot import mask_overlay

import csv #Save data as csv

#Functions

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
    
    #true = Percent_True(np.max(masks_Tc), np.max(True_masks))
    
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
    #ax[2][0].set_title('True masks: %s' %(np.max(True_masks)))
    
    
    ax[2][1].imshow(DAPI)
    ax[2][1].axis('off')
    #ax[2][1].set_title('Percent true: %s. Coverage: %s' %(true, coverage))
    
    plt.subplots_adjust(top=1.1)
    plt.subplots_adjust(wspace=-0.6, hspace=0.2)
    
    return fig

#####
# Wroking direcotry (where program is saved)
wd = r'C:\Users\Usuario\Desktop\Joaquin\Test\20250624 - Muestras Dualseq Mayo y Junio'
os.chdir(wd)

# Montage direcotry
figuredir = os.path.join(wd, "Figures_True_Positive_Nuc_og")

make_sure_path_exists(figuredir)


# Folder where all the images are stored, each within a direcotry for all the images in a field
main_folder = r'C:\Users\Usuario\Desktop\Joaquin\Test\20250624 - Muestras Dualseq Mayo y Junio\Fotos'

# List of each folder containing the images for the fields of view
fields = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]


DAPI = [] #Lists of all the images (numpy arrays)
Tc =[]
for fov in range(len(fields)):
    #List of all images for a field
    campo = os.path.join(main_folder, fields[fov])
    fotos =  [f for f in os.listdir(campo) if os.path.isfile(os.path.join(campo, f))] # Excludes Metadatafolder
    fotos.sort()
    
    img = imread(os.path.join(campo, fotos[0])) #Load DNA stains 
    DAPI.append(img)
    img = imread(os.path.join(campo, fotos[1])) #Load Tc stains
    Tc.append(img)

## Cellpose setup
io.logger_setup()

#Pretrained model selection: model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

#Load the specifically trained model
model_Nuc = models.CellposeModel(gpu=True, pretrained_model="cpsam") #Model to detect amastigotes from DNA stain


#Running the model
masks_Nuc, flows_Nuc, styles_Nuc = model_Nuc.eval(DAPI)



#Columns for the output table of Average FOV measurments
#List of lists, each row corresponds to the measurments for a field of view

head = ["Cepa", "Linea", "Campo", "Nucleos_totales", "Nucleos_equivocados", 
            "Superposicion"]

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

for im in range(len(DAPI)):  

    #image_conditions = fields.split("-")
    conditions = fields[im].split("-")
    
    #Binary mask for antiTc channel
    Tc_smooth = filters.gaussian(Tc[im])
    Tc_Threshold = filters.threshold_otsu(Tc_smooth)
    Tc_mask = Tc_smooth > Tc_Threshold        
    
    #Total surface area for anti Tc signal
    Tc_area = np.sum(Tc_mask)
    
    #Binary mask for all nuclei
    Nuc_mask_binary = masks_Nuc[im] > 0
    
    #Total surface area for all nuceli
    Nuc_area = np.sum(Nuc_mask_binary)
    
    #Overlap between antiTc mask and nucelar mask
    overlapped = Nuc_mask_binary * Tc_mask
    overlap = np.sum(overlapped) #Area
    overlap_per = round((overlap/Nuc_area) * 100, 3)#Percentage overlap
    
    
    
    Detected_Nuc = int(np.max(masks_Nuc[im]))
    
    Nucleos =  masks_Nuc[im]
    equivocados = 0
    for nuc in range(1, Detected_Nuc):
        nucleo = Nucleos == nuc
        nuc_falso = nucleo * Tc_mask
        hay_Tc = np.sum(nuc_falso)
        if hay_Tc > 0:
            equivocados += 1
    
        
        
    '''
    False_masks = morphology.label(masks_Nuc[im] * Tc_mask)
        
    True_detected = Detected_Nuc - int(np.max(False_masks))    
    False_detected = int(np.max(False_masks)) 
    
    
    true_total = True_masks > 0 
    coverage = round((np.sum(true_total) / np.sum(Tc_mask)) *100, 2 )
    '''
    
    row = [conditions[0], conditions[1], conditions[2], Detected_Nuc, equivocados, overlap_per]

    #Save current image's measurments
    results.append(row)
    
    
    #Plot
    fig_name = "%s.png" % (fields[im])  
    fig = summary_plot(fields[im], DAPI[im], Tc[im], masks_Nuc[im], Tc_mask, overlapped, Tc_Threshold, overlap_per)
    plt.savefig(os.path.join(figuredir, fig_name), bbox_inches='tight', dpi = 300) #Saves the plot
    plt.close() #Closes the plot
    
    
#If the output file is not in the program's folder, try at C:/Users/Usuario
with open(os.path.join(wd,'truth_results_Nuc_og.csv'), 'w', newline='') as f: #Measurements for each field's average value
    writer = csv.writer(f)
    writer.writerows(results)
