#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:17:06 2017

@author: Sascha Hofmann
"""

from __future__ import division
import os
from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
from scipy.optimize import fsolve
from mpl_toolkits import axes_grid1
from PIL import Image
import pandas as pd
#################### Functions ############################

def gamma(w, nr, n1, phi):
    return 2*np.pi/w*np.sqrt(nr**2 - n1**2* np.sin(phi)**2)

def beta(w, nr,n1, phi):
    return 2*np.pi/w*np.sqrt(n1**2* np.sin(phi)**2 - nr**2 )

def int_inf(gamma, beta):
    return 2*gamma**2/(beta*(beta**2 + gamma**2))

def int_t(t, y, b2, b3):
    a = (2*y**2*((b2**2+b3**2)/(2*b2)*np.sinh(2*b2*t) + b3*np.cosh(2*b2*t)
                - b3 + t*(b2**2-b3**2)))
    b = (y**2*(b2*np.cosh(b2*t) +b3*np.sinh(b2*t))**2 
         + b2**2*(b2*np.sinh(b2*t) + b3*np.cosh(b2*t))**2)
    return a/b

def fluorescence(t, w, n1, n2, n3, n4, phi, d_m):
    crit = np.arcsin(n2/n1)
    phi13 = np.arcsin(n3/n1)
    phi14 =  np.arcsin(n4/n1)
    if phi < crit:
        raise Exception("TIRF angle too small, no evanescent wave!" )
        
    if phi >= phi13 and phi >= phi14 :
        # case a.) evanescent waves in medium 3 and 4
        gamma1 = n1*2*np.pi/w *np.cos(phi)
        beta2 = beta(w, n2, n1, phi)
        beta3 = beta(w, n3, n1, phi)
        beta4 = beta(w, n4, n1, phi)
        delta = beta3 * d_m
        
        a1 = (beta3**2 + beta2*beta4)*np.sinh(delta) + beta3*(beta2+beta4)*np.cosh(delta)
        a2 = - (beta3**2 - beta2*beta4)*np.sinh(delta) - beta3*(beta4- beta2)*np.cosh(delta)
        a3 = (beta3*(beta4*np.cosh(beta2*t) + beta2*np.cosh(beta2*t))*np.cosh(delta) +
              (beta2*beta4*np.cosh(beta2*t) + beta3**2*np.sinh(beta2*t))*np.sinh(delta))
        a4 = (beta3*(beta4*np.cosh(beta2*t) + beta2*np.sinh(beta2*t))*np.cosh(delta) +
              (beta2*beta4*np.sinh(beta2*t) + beta3**2*np.cosh(beta2*t))*np.sinh(delta))
        # Eq. 19 divided by Eg. 22
        F = (gamma1**2 +beta2**2) /4* (-a1**2*(1-np.exp(2*beta2*t))+ a2**2*(1-np.exp(-2*beta2*t))
                + 4*beta2*a1*a2*t)/(gamma1**2*a3**2 +beta2**2*a4**2)
        # Unnormed Intensitiy
        # I = (gamma1**2* (-a1**2*(1-np.exp(2*beta2*t))+ a2**2*(1-np.exp(-2*beta2*t))
        #     + 4*beta2*a1*a2*t)/(2*beta2*(gamma1**2*a3**2 +beta2**2*a4**2)))
    elif phi <= phi13 and phi >= phi14:
        # case b.) evanescent wave in medium 4, continous in 3
        gamma1 = n1*2*np.pi/w *np.cos(phi)
        beta2 = beta(w, n2, n1, phi)
        gamma3 = gamma(w, n3, n1, phi)
        beta4 = beta(w, n4, n1, phi)
        delta = gamma3 * d_m
        
        b1 = -(gamma3**2 - beta2*beta4)*np.sin(delta) + gamma3*(beta2+beta4)*np.cos(delta)
        b2 = (gamma3**2 - beta2 * beta4)*np.sin(delta) + gamma3*(beta2 - beta4)*np.cos(delta)
        b3 = (gamma3*(beta4 * np.sinh(beta2*t) + beta2*np.cosh(beta2*t)*np.cos(delta)) +
              (beta2*beta4*np.cosh(beta2*t) +gamma3**2*np.sinh(beta2*t))*np.sin(delta))
        b4 = (gamma3*(beta4 * np.cosh(beta2*t) + beta2*np.sinh(beta2*t)*np.cos(delta)) +
              (beta2*beta4*np.sinh(beta2*t) +gamma3**2*np.cosh(beta2*t))*np.sin(delta))
        # Eq. 28; numerator not squared as pointed out in Vigeant et al.
        # divided by Eq. 22
        F = (gamma1**2 + beta2**2)/4*(-b1**2*(1-np.exp(2*beta2*t)) + b2**2*(1 - np.exp(-beta2*t)) 
             + 4*beta2*b1*b2*t)/(gamma1**2*b3**2 + beta2**2 *b4**2   )
        # Unnormed Intensitiy
        #        I = gamma1**2*(-b1**2*(1-np.exp(2*beta2*t)) + b2**2*(1 - np.exp(-beta2*t)) 
        #             + 4*beta2*b1*b2*t)/(2*beta2*(gamma1**2*b3**2 + beta2**2 *b4**2   ))
    elif phi >= phi13 and phi <= phi14:
        # case c.) evanescent wave in medium 3, continous in 4
        # TODO: working?
        gamma1 = n1*2*np.pi/w *np.cos(phi)
        beta2 = beta(w, n2, n1, phi)
        beta3 = beta(w, n3, n1, phi)
        gamma4 = gamma(w, n4, n1, phi)
        delta = beta3 * d_m    

        c1 = beta2*gamma4*np.sinh(delta)
        c2 = beta3*gamma4*np.cosh(delta)
        c3 = beta3**2 * np.sinh(delta)
        c4 = beta2*beta3*np.cosh(delta)
        # Eq. 22 for background intensity
        I_norm = 2*gamma1**2/(beta2*(gamma1**2+beta2**2))
        # From Eq. 29
        denominator = ((beta3*(gamma1*gamma4 - beta2**2)*np.sinh(beta2*t)*np.cosh(delta)
        + beta2*(gamma1*gamma4 - beta3**2)*np.cosh(beta2*t)*np.sinh(delta))**2 +
        (beta2*beta3*(gamma1 + gamma4)*np.cosh(beta2*t)*np.cosh(delta) + 
         (gamma1*beta3**2 + beta2**2*gamma4)*np.sinh(beta2*t)*np.sinh(delta))**2)
        #print gamma4
        # Eq. 31
        I = gamma1**2/denominator*((c1**2 + c2**2 + c3**2 + c4**2)/beta2*np.sinh(2*beta2*t) 
        + 2*(c1**2 - c2**2 - c3**2 - c4**2)*t + 2/beta2*(c1*c2 + c3*c4)*(np.cosh(2*beta2*t) -1))
        F = I/I_norm
    else:
        raise Exception('No fitting condition: n3: '+str(n3)+ 'n4: ' +str(n4))
    return F

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
##################### Main ###############################

# Input
phi = np.radians(70.62) #np.radians(70.82)
n1 = 1.525
n2 = 1.337
n3 = 1.45
n4 = 1.37 #1.37
w = 647
d_m = 4

directory = 'Images/'
bg_filename = "6x8_BSA_171017_009_bg.tif"
cell_filename = '6x8_BSA_171017_009.tif'
save_dir = directory+cell_filename.replace('.tif', '/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
background = io.imread(directory + bg_filename)
cell = io.imread(directory + cell_filename)
normed = cell/background
table = []
n3_list = np.arange(1.35, 1.51, 0.01)
n4_list =  np.arange(1.35, 1.45, 0.01)
for n3 in n3_list:
    crit = np.sin(phi)*n1
    if crit > n2 and crit < n3 and crit > n4:
        case = 'a'
    elif crit > n2 and crit > n3 and crit > n4:
        case = 'b'
    else:
        case = 'c'
    height_img = np.zeros_like(normed)
    for i in range(normed.shape[0]):
        print i
        for j in range(normed.shape[1]):
            
            func = lambda t: normed[i,j] - fluorescence(t, w, n1, n2, n3, n4, phi, d_m)
            height_img[i,j] = fsolve(func, 0)
    #height[height >120] =120
    
    
    img = Image.fromarray(height_img)   # Creates a PIL-Image object
    save_name = cell_filename.replace('.tif', '') +'_n3_'+str(n3) + '_height.tif'
    img.save(save_dir + save_name)
    # Calculating cell average
    x1 = 51
    y1 = 41
    x2 = 257
    y2 = 135
    av = np.mean(height_img[y1:y2, x1:x2])
    std = np.std(height_img[y1:y2, x1:x2])
    row = [av,std, case]
    table.append(row)
    
# Save table
df = pd.DataFrame(table, index = n3_list, columns = ['Mean', 'Std', 'Case'] )
df.to_csv(save_dir+cell_filename.replace('.tif', '')+'_n3_height_refr')
'''
# Display 4 images: BG, Cell, Normed, Heightmap with colorbar
fig, ax = plt.subplots(4,1, figsize=(16, 16))
ax[0].imshow(background, cmap = 'gray' )    
ax[0].axis( 'off')
ax[1].imshow(cell, cmap = 'gray')
ax[1].axis( 'off')
ax[2].imshow(cell/background, cmap = 'gray')
ax[2].axis( 'off')
im = ax[3].imshow(height, cmap = 'inferno' )
ax[3].axis( 'off')
add_colorbar(im)
plt.savefig('tiraf.pdf', format = 'pdf', dpi = 1000)



#Wave vectors
plt.figure(2)
n4 = [1.35, 1.36,1.37, 1.39]
t = np.linspace(0, 200)
linestyle = [ 'dashed', 'dashdot', 'solid', 'dotted']

for i in xrange(len(n4)):
    F = fluorescence(t, w, n1, n2, n3, n4[i], phi, d_m)
    plt.plot(t,F, c ='black', ls = linestyle[i], label= r'$n_4 = $'+ str(n4[i]))

    
plt.legend(loc =4)
plt.xlabel('Distance t [nm]')
plt.ylabel('Normed Fluorescence Intensity')
plt.savefig('tiraf_different_n2.pdf', format ='pdf', dpi= 1000)

# Wave vectors
plt.figure(2)
n3 = [1.41, 1.43,1.45, 1.47]
t = np.linspace(0, 200)
linestyle = [ 'dashed', 'dashdot', 'solid', 'dotted']

for i in xrange(len(n3)):
    F = fluorescence(t, w, n1, n2, n3[i], n4, phi, d_m)
    plt.plot(t,F, c ='black', ls = linestyle[i], label= r'$n_3 = $'+ str(n3[i]))

    
plt.legend(loc =4)
plt.xlabel('Distance t [nm]')
plt.ylabel('Normed Fluorescence Intensity')
plt.savefig('tiraf_different_n3.pdf', format ='pdf', dpi= 1000)
# Wave vectors
plt.figure(2)
n2 = [1.3, 1.32,1.337, 1.37]
t = np.linspace(0, 200)
linestyle = [ 'dashed', 'dashdot', 'solid', 'dotted']

for i in xrange(len(n2)):
    F = fluorescence(t, w, n1, n2[i], n3, n4, phi, d_m)
    plt.plot(t,F, c ='black', ls = linestyle[i], label= r'$n_2 = $'+ str(n2[i]))

    
plt.legend(loc =4)
plt.xlabel('Distance t [nm]')
plt.ylabel('Normed Fluorescence Intensity')
plt.savefig('tiraf_different_n2.pdf', format ='pdf', dpi= 1000)

# Wave vectors
plt.figure(2)
phi = [np.radians(70), np.radians(75),  np.radians(80), np.radians(85)]
t = np.linspace(0, 200)
linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
degree_sign= u'\N{DEGREE SIGN}'
for i in xrange(len(phi)):
    F = fluorescence(t, w, n1, n2, n3, n4, phi[i], d_m)
    plt.plot(t,F, c ='black', ls = linestyle[i], label= str(int(np.rad2deg(phi[i]))) + r'$^{\circ}$')

    
plt.legend(loc =4)
plt.xlabel('Distance t [nm]')
plt.ylabel('Normed Fluorescence Intensity')
plt.savefig('tiraf_different_phi.pdf', format ='pdf', dpi= 1000)

plt.figure(1)
t = np.linspace(0, 200)
F = fluorescence(t, w, n1, n2, n3, n4, np.radians(72), d_m)

plt.xlim((0,200))
plt.ylim(0,1)
plt.plot(t,F, c ='black',)
plt.savefig('tiraf.pdf', format ='pdf', dpi= 1000)
'''
#t = np.linspace(np.min(height),250)
#F = fluorescence(t, w, n1, n2, n3, n4, np.radians(75), d_m)
#plt.figure(2)
#plt.plot(height, corrected, ls = '', marker = '.', c = 'r')
#plt.plot(t, F)
#plt.axis([np.min(height),250,0,1])

