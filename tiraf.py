#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:17:06 2017

@author: Sascha Hofmann
"""

from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
from scipy.optimize import fsolve
from mpl_toolkits import axes_grid1
from PIL import Image
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
    crit = np.sin(phi)*n1
    if crit > n2 and crit < n3 and crit > n4:
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
         
        F = (gamma1**2 + beta2**2)/4*(-b1**2*(1-np.exp(2*beta2*t)) + b2**2*(1 - np.exp(-beta2*t)) 
             + 4*beta2*b1*b2*t)/(gamma1**2*b3**2 + beta2**2 *b4**2   )
        # Unnormed Intensitiy
#        I = gamma1**2*(-b1**2*(1-np.exp(2*beta2*t)) + b2**2*(1 - np.exp(-beta2*t)) 
#             + 4*beta2*b1*b2*t)/(2*beta2*(gamma1**2*b3**2 + beta2**2 *b4**2   ))
    elif crit > n2 and crit > n3 and crit > n4:
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
        
        F = (gamma1**2 +beta2**2) /4* (-a1**2*(1-np.exp(2*beta2*t))+ a2**2*(1-np.exp(-2*beta2*t))
                + 4*beta2*a1*a2*t)/(gamma1**2*a3**2 +beta2**2*a4**2)
        # Unnormed Intensitiy
#        I = (gamma1**2* (-a1**2*(1-np.exp(2*beta2*t))+ a2**2*(1-np.exp(-2*beta2*t))
#                + 4*beta2*a1*a2*t)/(2*beta2*(gamma1**2*a3**2 +beta2**2*a4**2)))
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
phi = np.radians(70)
n1 = 1.515
n2 = 1.337
n3 = 1.45
n4 = 1.37
w = 647
d_m = 4
# Wave vectors
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

## Wave vectors
#plt.figure(2)
#n3 = [1.41, 1.43,1.45, 1.47]
#t = np.linspace(0, 200)
#linestyle = [ 'dashed', 'dashdot', 'solid', 'dotted']
#
#for i in xrange(len(n3)):
#    F = fluorescence(t, w, n1, n2, n3[i], n4, phi, d_m)
#    plt.plot(t,F, c ='black', ls = linestyle[i], label= r'$n_3 = $'+ str(n3[i]))
#
#    
#plt.legend(loc =4)
#plt.xlabel('Distance t [nm]')
#plt.ylabel('Normed Fluorescence Intensity')
#plt.savefig('tiraf_different_n3.pdf', format ='pdf', dpi= 1000)
## Wave vectors
#plt.figure(2)
#n2 = [1.3, 1.32,1.337, 1.37]
#t = np.linspace(0, 200)
#linestyle = [ 'dashed', 'dashdot', 'solid', 'dotted']
#
#for i in xrange(len(n2)):
#    F = fluorescence(t, w, n1, n2[i], n3, n4, phi, d_m)
#    plt.plot(t,F, c ='black', ls = linestyle[i], label= r'$n_2 = $'+ str(n2[i]))
#
#    
#plt.legend(loc =4)
#plt.xlabel('Distance t [nm]')
#plt.ylabel('Normed Fluorescence Intensity')
#plt.savefig('tiraf_different_n2.pdf', format ='pdf', dpi= 1000)

## Wave vectors
#plt.figure(2)
#phi = [np.radians(70), np.radians(75),  np.radians(80), np.radians(85)]
#t = np.linspace(0, 200)
#linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
#degree_sign= u'\N{DEGREE SIGN}'
#for i in xrange(len(phi)):
#    F = fluorescence(t, w, n1, n2, n3, n4, phi[i], d_m)
#    plt.plot(t,F, c ='black', ls = linestyle[i], label= str(int(np.rad2deg(phi[i]))) + r'$^{\circ}$')
#
#    
#plt.legend(loc =4)
#plt.xlabel('Distance t [nm]')
#plt.ylabel('Normed Fluorescence Intensity')
#plt.savefig('tiraf_different_phi.pdf', format ='pdf', dpi= 1000)

#plt.figure(1)
#t = np.linspace(0, 200)
#F = fluorescence(t, w, n1, n2, n3, n4, np.radians(72), d_m)
#
#plt.xlim((0,200))
#plt.ylim(0,1)
#plt.plot(t,F, c ='black',)
#plt.savefig('tiraf.pdf', format ='pdf', dpi= 1000)

'''
directory = 'Images/'
bg_filename = "plm_background.tif"
cell_filename = 'plm_cell.tif'
background = io.imread(directory + bg_filename)
cell = io.imread(directory + cell_filename)
corrected = cell/background
height = np.zeros_like(corrected)
for i in range(corrected.shape[0]):
    for j in range(corrected.shape[1]):
        func = lambda t: corrected[i,j] - fluorescence(t, w, n1, n2, n3, n4, np.radians(75), d_m)
        height[i,j] = fsolve(func, 0)
height[height >120] = 34  
fig, ax = plt.subplots(4,1, figsize=(16, 16))

img = Image.fromarray(height)   # Creates a PIL-Image object
save_name = cell_filename.replace('.tif', '') + '_height.tif'
img.save(directory + save_name)

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
'''
#t = np.linspace(np.min(height),250)
#F = fluorescence(t, w, n1, n2, n3, n4, np.radians(75), d_m)
#plt.figure(2)
#plt.plot(height, corrected, ls = '', marker = '.', c = 'r')
#plt.plot(t, F)
#plt.axis([np.min(height),250,0,1])

