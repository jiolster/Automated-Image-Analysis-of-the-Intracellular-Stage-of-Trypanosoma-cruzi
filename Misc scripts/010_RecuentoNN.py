# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 19:32:53 2025

@author: Usuario
"""

#Run after main script 
#im = 4
import matplotlib.pyplot as plt

masks_Tc2 = masks_Tc.copy()
masks_Nuc2 =masks_Nuc.copy()

image = imgs[im]
name = fields[im]
masks_Tc = masks_Tc[im]
masks_Nuc = masks_Nuc[im]

dummy = np.zeros((512,512))

#♦Summary plot
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
plt.close()


#Plot amastigote centroids
fig, ax = plt.subplots()
plt.imshow(dummy, cmap='gray')
ax.set_facecolor('black')
ax.set_aspect(0.75)
ax.set_aspect('equal', adjustable='box')
for ama in pairs:
    x = ama[0][1]
    y = ama[0][0]
    ax.plot(x, y, marker = '.', markersize = 2)
plt.xticks([])
plt.yticks([])
plt.savefig('ama_centroids.svg', format='svg')

#Plot Nuc centroids
fig, ax = plt.subplots()
plt.imshow(dummy, cmap='gray')
ax.set_facecolor('black')
ax.set_aspect('equal', adjustable='box')
for ama in pairs:
    x = ama[1][1]
    y = ama[1][0]
    ax.plot(x, y, marker = 's', markersize = 2)
plt.xticks([])
plt.yticks([])
plt.savefig('Nuc_centroids.svg', format='svg')

#Centroides unidos
fig, ax = plt.subplots()
plt.imshow(dummy, cmap='gray')
ax.set_facecolor('black')
ax.invert_yaxis()
ax.set_aspect('equal', adjustable='box')
for pair in pairs:
    xs=(pair[0][1], pair[1][1])
    ys=(pair[0][0], pair[1][0])
    plt.plot(xs, ys, color="white", linewidth = 0.3)
for ama in pairs:
    x = ama[1][1]
    y = ama[1][0]
    ax.plot(x, y, marker = 's', markersize = 2)
for ama in pairs:
    x = ama[0][1]
    y = ama[0][0]
    ax.plot(x, y, marker = '.', markersize = 4)
plt.xticks([])
plt.yticks([])
plt.savefig('Joined_centroids.svg.svg', format='svg')



plt.figure(facecolor='white')
plt.imshow(masks_Nuc, cmap='inferno')
plt.xticks([])
plt.yticks([])

mask = masks_Tc > 0
edges = feature.canny(mask)
edge_coords = np.argwhere(edges)

masks_Tc2 = relabel_sequential(masks_Tc)

# Get coordinates of edge pixels
edge_coords = np.argwhere(edges)


fig, ax = plt.subplots()
ax.set_facecolor('black')
ax.invert_yaxis()
ax.set_aspect(0.75)
plt.imshow(mask, cmap='gray')
plt.plot(edge_coords[:, 1], edge_coords[:, 0], 'r.', markersize=0.1) # Plot edges in red

plt.axis('off') # Hide axes ticks and labels
plt.show()


#Randomize labels
import skimage.measure
import skimage.util
import numpy as np

# Assuming you have a label image called 'labeled_image'
# Example:
labeled_image = masks_Tc

# Get the unique labels
unique_labels = np.unique(labeled_image)
unique_labels = unique_labels[unique_labels != 0]  # Exclude background label (0)

# Create a shuffled array of the labels
shuffled_labels = np.random.permutation(unique_labels)

# Create a mapping from old labels to shuffled labels
label_mapping = dict(zip(unique_labels, shuffled_labels))

# Apply the mapping to the label image
randomized_labeled_image = np.zeros_like(labeled_image)
for old_label, new_label in label_mapping.items():
    randomized_labeled_image[labeled_image == old_label] = new_label

fig, ax = plt.subplots()
ax.set_facecolor('black')
ax.invert_yaxis()
plt.imshow(image, cmap='gray')
plt.axis('off') # Hide axes ticks and labels
plt.savefig('DAPI.svg', format='svg')