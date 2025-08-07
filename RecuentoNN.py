# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:11:58 2025

@author: Usuario
"""
#Librarires
import os #For importing images and saving figures
import errno # for checking whether the montage direcotry already exists

from cellpose import models, io #Cellpose ML model for single cell segmentation
from cellpose.io import imread 
from cellpose.plot import mask_overlay

import numpy as np
import matplotlib.pyplot as plt

from skimage import  measure#Extract labeled object properties
import math #Square root

import csv #Save data as csv


#Functions

# Makes a new direcory for the given path, unless it alredy exists
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise   

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

##
def summary_plot(name, image, masks_Tc, masks_Nuc, pairs, percent_infected, amastigotes_per_infected):
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

    Returns
    -------
    fig : A figure with 4 panels.

    '''
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0][0].imshow(image, cmap="gray")
    ax[0][0].axis('off')
    ax[0][0].set_title('%s' %(name))

    ax[0][1].imshow(mask_overlay(image, masks_Tc))
    ax[0][1].axis('off')
    ax[0][1].set_title('Amastigote masks: %s' %(np.max(masks_Tc)))

    ax[1][0].imshow(mask_overlay(image, masks_Nuc))
    ax[1][0].axis('off')
    ax[1][0].set_title('Nuclear masks: %s' %(np.max(masks_Nuc)))

    #Plot lines between each amastigote and each nucleus
    for pair in pairs:
        xs=(pair[0][1], pair[1][1])
        ys=(pair[0][0], pair[1][0])
        plt.plot(xs, ys, color="white", linewidth = 0.3)

    #Add total number of amastigotes assigned to each nucleus
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
wd = r'C:\Users\Usuario\Desktop\Joaquin\Test\20250624 - Muestras Dualseq Mayo y Junio'
os.chdir(wd)

# Montage direcotry
figuredir = os.path.join(wd, "Figures")

make_sure_path_exists(figuredir)


# Folder where all the images are stored, each within a direcotry for all the images in a field
main_folder = r'C:\Users\Usuario\Desktop\Joaquin\Test\20250624 - Muestras Dualseq Mayo y Junio\Fotos'

# List of each folder containing the images for the fields of view
fields = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]


imgs = [] #Lists of all the images (numpy arrays)
for fov in range(len(fields)):
    #List of all images for a field
    campo = os.path.join(main_folder, fields[fov])
    fotos =  [f for f in os.listdir(campo) if os.path.isfile(os.path.join(campo, f))] # Excludes Metadatafolder
    fotos.sort()
    
    img = imread(os.path.join(campo, fotos[0])) 
    imgs.append(img)
    


## Cellpose setup
io.logger_setup()

#Pretrained model selection: model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

#Trained models
model_Tc = models.CellposeModel(gpu=True, pretrained_model="DAPI_Tc") #Model to detect amastigotes from DNA stain
model_Nuc= models.CellposeModel(gpu=True, pretrained_model="DAPI_Nuc")  #Model to detect mammalian Nuclei from DNA stain


#Running the models
masks_Tc, flows_Tc, styles_Tc = model_Tc.eval(imgs)
masks_Nuc, flows_Nuc, styles_Nuc = model_Nuc.eval(imgs)


#Columns for the output table of Average FOV measurments
#List of lists, each row corresponds to the measurments for a field of view
sc_head = ["Cepa","Replica","Campo", "Célula", "Amastigotes"]

fov_head = ["Cepa", "Replica", "Campo", "Amastigotes", "Células", 
            "Amas por célula", "Porcentaje", "Amas por célula infectada" ]

# First row is the name of the columns
sc_results = [sc_head] 
fov_results = [fov_head]

for im in range(len(imgs)):  

    #image_conditions = fields.split("-")
    conditions = fields[im].split("-")
    
    
    Nuclei_props = measure.regionprops(masks_Nuc[im])
    amastigote_props = measure.regionprops(masks_Tc[im])
    
    
    #Measure the distance of each amastigote to each nucleus, then save which nucleus is the closest to each amastigote
    amastigotes_in_cell, sc_results_partial, fov_row, pairs, percent_infected, amastigotes_per_infected = find_infected(amastigote_props, Nuclei_props, 3, conditions)
    
    #Save current image's measurments
    fov_results.append(fov_row)
    sc_results = sc_results + sc_results_partial
    
    #Plot
    fig_name = "%s.png" % (fields[im])  
    fig = summary_plot(fields[im], imgs[im], masks_Tc[im], masks_Nuc[im], pairs, percent_infected, amastigotes_per_infected)
    plt.savefig(os.path.join(figuredir, fig_name), bbox_inches='tight', dpi = 300) #Saves the plot
    plt.close() #Closes the plot


#If the output file is not in the program's folder, try at C:/Users/Usuario
with open(os.path.join(wd,'sc_results.csv'), 'w', newline='') as f: #Measurements for each field's average value
    writer = csv.writer(f)
    writer.writerows(sc_results)
    
with open(os.path.join(wd,'fov_results.csv'), 'w', newline='') as f: #Measurements for each field's average value
    writer = csv.writer(f)
    writer.writerows(fov_results)
    
