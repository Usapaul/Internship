#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np

from astropy.io import ascii
from astropy.table import Table

from scipy.signal import convolve

import matplotlib.pyplot as plt
import matplotlib.colors

#------------------------------------------------
parser = argparse.ArgumentParser(description=
        ''' Make 2d-histogram from datafile with coordinates of
            points of 2d-graphics [x,y] 
            (or astropy.Table with specified names of X- and Y- columns). 
        ''')

parser.add_argument('datafile',help='Name of file containing [x,y] coords')

parser.add_argument('--columnX','-x',help='Name of the X column',
                    metavar='column_name',default='col0')

parser.add_argument('--columnY','-y',help='Name of the Y column',
                    metavar='column_name',default='col1')

parser.add_argument('--rangeX',help='Bounds of the range of X',nargs=2,
                    metavar=('x1','x2'),type=float,default=None)

parser.add_argument('--rangeY',help='Bounds of the range of Y',nargs=2,
                    metavar=('y1','y2'),type=float,default=None)

parser.add_argument('--stepX',help='Bin size along X-axis',
                    type=float,default=None)

parser.add_argument('--stepY',help='Bin size along Y-axis',metavar='size',
                    type=float,default=None)

parser.add_argument('--fixed','-f',dest='fixedFlag',action='store_true',
                    help='Flag to make X- and Y-size of bins equal')

parser.add_argument('--log',dest='logFlag',action='store_true',
                    help='Flag to write log10(counts) in every bin')

args = parser.parse_args()

#------------------------------------------------

dataTable = ascii.read(args.datafile)
ndata = len(dataTable)

arrayX = dataTable[args.columnX]
arrayY = dataTable[args.columnY]

#------------------------------------------------

# Specifing parameters for histogram: user's values or internal initialization
# (in the case they are not specified in the command line).
# N bins is choosen by user or from Sturges' formula

rangeX = args.rangeX if args.rangeX is not None \
         else [np.min(arrayX),np.max(arrayX)]

stepX  = args.stepX if args.stepX is not None \
         else (rangeX[1] - rangeX[0]) / (1 + int(np.log2(ndata)))

rangeY = args.rangeY if args.rangeY is not None \
         else [np.min(arrayY),np.max(arrayY)]

stepY  = args.stepY if args.stepY is not None \
         else (rangeY[1] - rangeY[0]) / (1 + int(np.log2(ndata)))

if args.fixedFlag is True:
    stepX = stepY = np.max(stepX,stepY)

gridX = np.arange(rangeX[0],rangeX[1]+stepX,stepX)
gridY = np.arange(rangeY[0],rangeY[1]+stepY,stepY)

numX, numY = len(gridX), len(gridY)

#------------------------------------------------

Hist, Hxedges, Hyedges = np.histogram2d(arrayX,arrayY,bins=[gridX,gridY])

Hist = Hist.T

if args.logFlag is True:
    Hist_not_log = Hist.copy()
    Hist = np.where(Hist > 0, np.log10(Hist), 0.)

#------------------------------------------------
# Creating edge-detection maps

# Gx and Gy -- Sobel's edge detection kernels [3x3]
# along x- and y-axis correspondingly.
# Gx1 and Gy1 -- the same but [1x7] and [7x1]

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])


Gy1 = np.array([[ 1],
                [-2],
                [ 1],
                [ 0],
                [-1],
                [ 2],
                [-1]])

# Gx1 = np.array([[-1,-2,-1,0,1,2,1]])
Gx1 = -Gy.T

edgeMapX  = convolve(Hist, Gx, mode='same')
edgeMapY  = convolve(Hist, Gy, mode='same')

edgeMapY1  = convolve(Hist, Gy1, mode='same')
edgeMapX1  = convolve(Hist, Gx1, mode='same')

edgeMapX, edgeMapY = np.abs(edgeMapX), np.abs(edgeMapY)
edgeMapX1, edgeMapY1 = np.abs(edgeMapX1), np.abs(edgeMapY1)


edgeMapXY = np.sqrt(edgeMapX**2 + edgeMapY**2)
edgeMapX1Y1 = np.sqrt(edgeMapX1**2 + edgeMapY1**2)

#------------------------------------------------

fig, ax = plt.subplots(2,3,figsize=(14,8))

plt.grid()

# ax[0,0] -- just Hist
ax[0,0].set(aspect='equal',
            title='Histogram'+str(' (log10)' if args.logFlag is True else ''))
ax[0,0].set_xlim(rangeX[0], rangeX[1])
ax[0,0].set_ylim(rangeY[1], rangeY[0])
pcolor00 = ax[0,0].pcolormesh(gridX,gridY,Hist,cmap='rainbow',
                              vmin=None,vmax=None)

fig.colorbar(pcolor00, ax=ax[0,0], fraction=0.046, pad=0.04)

#-----------------------------------------------
# ax[0,1] -- Hist x Gx = HGx
ax[0,1].set(aspect='equal',title='Historgam x Gx')
ax[0,1].set_xlim(rangeX[0], rangeX[1])
ax[0,1].set_ylim(rangeY[1], rangeY[0])
pcolor01 = ax[0,1].pcolormesh(gridX,gridY,edgeMapX,cmap='rainbow',
                              vmin=0,vmax=3)

fig.colorbar(pcolor01, ax=ax[0,1], fraction=0.046, pad=0.04)

#-----------------------------------------------
# ax[1,0] -- Hist x Gy = HGy
ax[1,0].set(aspect='equal',title='Historgam x Gy')
ax[1,0].set_xlim(rangeX[0], rangeX[1])
ax[1,0].set_ylim(rangeY[1], rangeY[0])
pcolor10 = ax[1,0].pcolormesh(gridX,gridY,edgeMapY,cmap='rainbow',
                              vmin=0,vmax=3)

fig.colorbar(pcolor10, ax=ax[1,0], fraction=0.046, pad=0.04)

#-----------------------------------------------
# ax[1,1] -- sqrt(HGx**2 + HGy**2)
ax[1,1].set(aspect='equal',title='sqrt(HGx**2 + HGy**2)')
ax[1,1].set_xlim(rangeX[0], rangeX[1])
ax[1,1].set_ylim(rangeY[1], rangeY[0])
pcolor11 = ax[1,1].pcolormesh(gridX,gridY,edgeMapXY,cmap='rainbow',
                              vmin=1.2,vmax=3)

fig.colorbar(pcolor11, ax=ax[1,1], fraction=0.046, pad=0.04)

#-----------------------------------------------
# ax[0,2] -- plotting stars
ax[0,2].set(aspect='equal',title='HRD')
ax[0,2].set_xlim(rangeX[0], rangeX[1])
ax[0,2].set_ylim(rangeY[1], rangeY[0])
pcolor02 = ax[0,2].plot(arrayX,arrayY,',')

#-----------------------------------------------
# ax[1,2] -- Hist x Gy1
ax[1,2].set(aspect='equal',title='Histogram x Gy1')
ax[1,2].set_xlim(rangeX[0], rangeX[1])
ax[1,2].set_ylim(rangeY[1], rangeY[0])
pcolor12 = ax[1,2].pcolormesh(gridX,gridY,edgeMapX1Y1,cmap='rainbow',
                              vmin=0,vmax=3)

fig.colorbar(pcolor12, ax=ax[1,2], fraction=0.046, pad=0.04)

#-----------------------------------------------

plt.tight_layout()
plt.show()


#------------------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def write_color_slice(filename,hist_map,colour):
    file = open(filename,'w')
    arr = hist_map[:,gridX[:-1] == find_nearest(gridX,colour)][:,0]
    for i in range(len(gridX)-1):
        file.write(str(gridY[i])+' '+str(arr[i])+'\n')
    file.close()


colourV_I = 2.

write_color_slice('fileHist_not_log.dat',Hist_not_log,colourV_I)
write_color_slice('fileEdgeGy.dat',edgeMapY,colourV_I)
write_color_slice('fileEdgeGy1.dat',edgeMapY1,colourV_I)

