#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from astropy.io import ascii
from astropy.table import Table
from scipy.signal import convolve

import matplotlib.colors
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf


import seaborn as sb
sb.set_style('whitegrid', {'grid.linestyle': '-'})

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

#------------------------------------------------
parser = argparse.ArgumentParser(description=
         ''' Creating a 2d histogram for the given coordinates
             of the points [x, y] from the data file
             (or astropy.Table with the specified column names X and Y).
         ''')

parser.add_argument('--datafile',help='Name of file containing [x,y] coords')

parser.add_argument('--columnX','-x',help='Name of the X column',
                    metavar='column_name',default='col0')

parser.add_argument('--columnY','-y',help='Name of the Y column',
                    metavar='column_name',default='col1')

parser.add_argument('--rangeX',help='X range bounds',nargs=2,
                    metavar=('x1','x2'),type=float,default=None)

parser.add_argument('--rangeY',help='Y range bounds',nargs=2,
                    metavar=('y1','y2'),type=float,default=None)

parser.add_argument('--stepX',help='The size of the bin along the X axis',
                    metavar='size',type=float,default=None)

parser.add_argument('--stepY',help='The size of the bin along the Y axis',
                    metavar='size',type=float,default=None)

parser.add_argument('--fixed','-f',dest='fixedFlag',action='store_true',
                    help='Flag to make bin sizes along X and Y axes equal')

parser.add_argument('--log',dest='logFlag',action='store_true',
                    help='A flag for writing logarithm values of the quantity \
                          [log10(quantity)] in each bin')

args = parser.parse_args()

#------------------------------------------------

#dataTable = ascii.read(args.datafile)
coordsRX = np.random.random((3000000)) * (4.5-0.5) + 0.5
coordsRY = np.random.random((3000000)) * (-1-(-5)) - 5.0
coordsRY += (-4-(-5)) + 0.25 * (coordsRX-2)**2
#dataTable = ascii.read(args.datafile)
ndata = len(coordsRX)

#arrayX = dataTable[args.columnX]
arrayX = coordsRX + np.random.normal(0., 0.1, coordsRX.shape)
#arrayY = dataTable[args.columnY]
arrayY = coordsRY + np.random.normal(0., 0.1, coordsRY.shape)

# np.abs(np.random((coordsRX.shape))*0.2)
# np.abs(np.random((coordsRY.shape))*0.2)

#------------------------------------------------
''' Specifying parameters for histogram: user's values (or internal 
    initialization in the case they are not specified in the command line).
    N bins is chosen by user or from Sturges' formula
'''
# rangeX = args.rangeX if args.rangeX is not None \
#          else [np.min(arrayX),np.max(arrayX)]

rangeX = [0.5,4.5]

stepX  = args.stepX if args.stepX is not None \
         else (rangeX[1] - rangeX[0]) / (1 + int(np.log2(ndata)))

# rangeY = args.rangeY if args.rangeY is not None \
#          else [np.min(arrayY),np.max(arrayY)]

rangeY = [-5,-1]


stepY  = args.stepY if args.stepY is not None \
         else (rangeY[1] - rangeY[0]) / (1 + int(np.log2(ndata)))

if args.fixedFlag is True:
    stepX = stepY = np.maximum(stepX,stepY)

gridX = np.arange(rangeX[0],rangeX[1]+stepX,stepX)
gridY = np.arange(rangeY[0],rangeY[1]+stepY,stepY)

# len(grid) minus 1 because grid is the edges of bins
numX, numY = len(gridX)-1, len(gridY)-1

#------------------------------------------------

Hist = np.histogram2d(arrayX,arrayY,bins=[gridX,gridY])[0]
Hist = Hist.T

Hist_not_log = Hist.copy()
if args.logFlag is True:
    np.log10(Hist,out=Hist,where=Hist>0)

#------------------------------------------------
''' Creating edge-detection maps.
    Gx and Gy -- edge detection 2d-kernels along x- and y-axis correspondingly.

    The filter is first derivative. 
    The simple difference formula delta_f = f(x+h) - f(x-h)
    correspond to (-1,0,1) kernel.

    As a variant of smoothing in the window of size 2 the following 
    formula applied: f(x+2h) - f(x-2h) (i.e. (0,-1, 0, 0, 0, 1, 0)).
    But f(x+2h) and f(x-2h) bins was smoothing by 
    Hann filter: (0, 0.5, 1, 0.5, 0). 

    As the result we have smoothed kernel (-0.5, -1, -0.5, 0, 0.5, 1, 0.5).
    Then the Hann filtering was applied in the perpendicular direction:
            [[-0.25, -0.5, -0.25, 0, 0.25, 0.5, 0.25],
             [-0.5 ,  -1,  -0.5,  0, 0.5,   1,  0.5 ],
             [-0.25, -0.5, -0.25, 0, 0.25, 0.5, 0.25]]
    For convenience this kernel array is multiplied by 4.
'''

Gx = np.array([[-1,-2,-1, 0, 1, 2, 1],
               [-2,-4,-2, 0, 2, 4, 2],
               [-1,-2,-1, 0, 1, 2, 1]])

Gy = -Gx.T


# Gy = np.array([[ 15,   69,  114,   69,  15],
#                [ 35,  155,  255,  155,  35],
#                [  0,    0,    0,    0,   0],
#                [-35, -155, -255, -155, -35],
#                [-15,  -69, -114,  -69, -15]])

# Gx = -Gy.T










#------------------------------------------------

def edgeDetection_filtering(histogram,kernel):
    ''' This function does convolution edgeMap with edgeDetection filter
    which is given as 'kernel' here.
    After a convolution with mode='same' border effects are present.
    This function sets to zero pixel (bin) values on the border of map.
    N -- number of bins to reset, equal to 
    int(len(kernel)-along filtering axis).
    '''
    edgeMap = convolve(histogram, kernel, mode='same')
    height, width = edgeMap.shape
    borderY, borderX = np.array(kernel.shape) // 2
    mask = np.ones(edgeMap.shape,dtype=bool)
    mask[borderY:height-borderY,borderX:width-borderX] = False
    return np.where(mask, 0., np.abs(edgeMap))

edgeMapX  = edgeDetection_filtering(Hist, Gx)
edgeMapY  = edgeDetection_filtering(Hist, Gy)

edgeMapXY = np.sqrt(edgeMapX**2 + edgeMapY**2)

#------------------------------------------------
#------------------------------------------------
''' In this block, the approximating TRGB polynomial is calculated.
    To construct the polynomial, peak points are used for Gaussians,
    approximating edgeMap slices in the TRGB region along the Y axis.
'''
binsY = (gridY[:-1] + gridY[1:]) / 2 # binsY/X -- central points of bins
binsX = (gridX[:-1] + gridX[1:]) / 2 # in the edgeDetection map

# boundsTRGB is used to constrain the TRGB-region in the edgeDetection map
boundsTRGB = [[-5.0,-2.0], [2.0,4.01]]
maskTRGB = ((binsY >= boundsTRGB[0][0]) & (binsY <= boundsTRGB[0][1]), \
            (binsX >= boundsTRGB[1][0]) & (binsX <= boundsTRGB[1][1]))

binsY_TRGB = binsY[maskTRGB[0]] # binsX/Y_TRGB -- central points of bins
binsX_TRGB = binsX[maskTRGB[1]] # in the map of TRGB region

regionTRGB = edgeMapXY[maskTRGB[0],:][:,maskTRGB[1]]
numY_TRGB, numX_TRGB = regionTRGB.shape


# We use the following functions to constrain the TRGB region 
# by specifying y_upper[lower]Bounds arrays,
# and y_initvalues is the initial approximation
# if args.datafile.find('SDSS') >= 0:
#     y_initvalues  = 0.3 * (binsX_TRGB-2)**2 - 3.9
#     y_upperBounds = 0.3 * (binsX_TRGB-2)**2 - 3.6
#     y_lowerBounds = 0.3 * (binsX_TRGB-2)**2 - 4.3
# elif args.datafile.find('PS1') >= 0:
#     y_initvalues  = 0.2 * (binsX_TRGB-2)**2 - 4.0
#     y_upperBounds = 0.2 * (binsX_TRGB-2)**2 - 3.6
#     y_lowerBounds = 0.2 * (binsX_TRGB-2)**2 - 4.3
# elif args.datafile.find('GAIA') >= 0:
#     y_initvalues  = 0.3 * (binsX_TRGB-2)**2 - 3.9
#     y_upperBounds = 0.3 * (binsX_TRGB-2)**2 - 3.5
#     y_lowerBounds = 0.3 * (binsX_TRGB-2)**2 - 4.4
# else:
#     y_initvalues  = np.ones_like(binsX_TRGB) + (-4.0)
#     y_upperBounds = np.ones_like(binsX_TRGB) + (-2.0)
#     y_lowerBounds = np.ones_like(binsX_TRGB) + (-6.0)

# y_initvalues  = np.ones_like(binsX_TRGB) * 1.0
# y_upperBounds = np.ones_like(binsX_TRGB) * 1.1
# y_lowerBounds = np.ones_like(binsX_TRGB) * 0.9

y_initvalues  = 0.31 * (binsX_TRGB-2.1)**2 - 3.9
y_upperBounds = 0.25 * (binsX_TRGB-2)**2 - 3.0
y_lowerBounds = 0.25 * (binsX_TRGB-2)**2 - 5.0

gauss_mean = np.zeros((numX_TRGB))
gauss_amplitude = np.zeros((numX_TRGB))
gauss_stddev = np.zeros((numX_TRGB))

p_degree = 1 # max polynom degree in the approximating function
fit = fitting.LevMarLSQFitter()
or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip,niter=10,sigma=3.0)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


for i in range(numX_TRGB):
    # print(i,'of',numX_TRGB)
    # g_init -- Gauss model with initial values
    sliceTRGB = regionTRGB[:,i]/np.max(regionTRGB[:,i])
    y_bounds = (y_lowerBounds[i],y_upperBounds[i])
    mapPeak = np.max(sliceTRGB)
    g_init = models.Gaussian1D(amplitude=mapPeak,
                    mean=binsY_TRGB[np.abs(sliceTRGB - mapPeak).argmin()],
                    stddev=5*stepX,
                    bounds={'mean':y_bounds})
                    # 'stddev':(stepX*0.0005,stepX*5000),
                    # 'amplitude':(mapPeak*0.0003,mapPeak*3000)})
    p_init = models.Polynomial1D(p_degree)
    gp_init = g_init + p_init
    # filtered_data, or_fitted_model = or_fit(g_init,binsY_TRGB,sliceTRGB)
    # or_fitted_model = fit(g_init,binsY_TRGB,sliceTRGB)
    # gauss_mean[i] = or_fitted_model.mean*1
    fitted_model = fit(g_init,binsY_TRGB,sliceTRGB)
    gauss_amplitude[i] = fitted_model.amplitude*1
    gauss_mean[i] = fitted_model.mean*1
    gauss_stddev[i] = fitted_model.stddev*1


curveTRGB = np.r_['0,2',binsX_TRGB, gauss_mean]

#------------------------------------------------
''' We find a polynomial approximation of the TRGB curve in this block
    using robust regression with the window specified in 'methodName'
'''
robust_degree = 2 # max degree of approximating polynom
robustXarray = np.zeros((len(curveTRGB[0,:]),robust_degree))
for i in range(robust_degree):
    robustXarray[:,i] = curveTRGB[0,:]**(i+1)

robustXarray = sm.add_constant(robustXarray,prepend = True)
robustYarray = curveTRGB[1,:]

methodName = 'TukeyBiweight'
rlm_model = sm.RLM(robustYarray, robustXarray, 
                   M=getattr(sm.robust.norms,methodName)())
results = rlm_model.fit()

stdout_curveTRGB = False
if stdout_curveTRGB is True:
    print('Approximating TRGB curve')
    print('Parameters:')
    print(results.params)
    print('Standard error:')
    print(results.bse)
    pvalues = np.round(results.pvalues,3)
    print('Pvalues:')
    print(pvalues)


def robust_fun(x,parameters):
    return np.sum(parameters * [x**i for i in range(len(parameters))])


y_robust_curveTRGB = np.array([robust_fun(binsX_TRGB[i],results.params) \
                               for i in range(numX_TRGB)])

robust_curveTRGB = np.r_['0,2',binsX_TRGB, y_robust_curveTRGB]

#================================================
''' Plotting block
'''
def plot_hist(histogram,place,colorbar,Title):
    ''' The function draws an edgeMap (histogram) in the graphics window.
        'place' is a tuple (row number, column number)
        (it is understood that the histograms in the graphics window 
        are arranged in tabular order).
        'colorbar' - these are color intensity borders that you want to control
        to create images with the most contrasting borders.
    '''
    ax[place].set(aspect='equal',title=Title)
    ax[place].set_xlim(rangeX[0], rangeX[1])
    ax[place].set_ylim(rangeY[1], rangeY[0])
    pcolor = ax[place].pcolormesh(gridX,gridY,histogram,cmap='rainbow',
                                vmin=colorbar[0],vmax=colorbar[1])
    fig.colorbar(pcolor, ax=ax[place], fraction=0.046, pad=0.04)


def plot_curve(x,y,place,marker,Title,**kwargs):
    ''' This function draws a curve in the graphics window.
        'place' is a tuple (row number, column number) (see def plot_hist())
        'marker' is a type of points/lines (points,squares,../solid,dashed,..)
    '''
    ax[place].set(aspect='equal',title=Title)
    ax[place].set_xlim(rangeX[0], rangeX[1])
    ax[place].set_ylim(rangeY[1], rangeY[0])
    ax[place].plot(x,y,marker,**kwargs)


fig, ax = plt.subplots(2,3,figsize=(14,8))
plt.grid()

plot_curve(arrayX,arrayY,(0,0),',','RGB $V-I$ vs $M_I$')
plot_curve(curveTRGB[0,:],curveTRGB[1,:],(1,2),'-','',c='black')
plot_curve(curveTRGB[0,:],curveTRGB[1,:],(1,2),'-','',c='black')
plot_curve(curveTRGB[0,:],curveTRGB[1,:],(1,0),'+','',c='black',markersize=2)
# plot_curve(curveTRGB[0,:],curveTRGB[1,:],(0,1),'-','',c='green')
plot_curve(robust_curveTRGB[0,:],robust_curveTRGB[1,:],(1,0),'-','TRGB curve',
           c='blue')
plot_curve(robust_curveTRGB[0,:],robust_curveTRGB[1,:],(0,1),'-','TRGB curve',
           c='blue')

# plot_hist(histogram=Hist,place=(0,1),colorbar=[None,None],
          # Title='Histogram'+' (log10)' if args.logFlag is True else '') 
plot_hist(histogram=Hist_not_log,place=(0,1),colorbar=[0,np.max(Hist_not_log)],
          Title='Histogram'+' (log10)' if args.logFlag is True else '') 

# np.log10(edgeMapX,out=Hist,where=Hist>0)
# np.log10(edgeMapY,out=Hist,where=Hist>0)
# np.log10(edgeMapXY,out=Hist,where=Hist>0)

edgeMapXY = edgeMapXY / np.max(edgeMapXY)
edgeMapX = edgeMapX / np.max(edgeMapX)
edgeMapY = edgeMapY / np.max(edgeMapY)


plot_hist(edgeMapX, (0,2),[0,None],'Histogram x Gx')
# plot_hist(edgeMapY, (1,1),[0,None],'Histogram x Gy')
plot_hist(edgeMapXY,(1,2),[0,None],'$\sqrt{(HGx)^2+(HGy)^2}$')


ax[1,1].set(title='Discrepance')
ax[1,1].set_xlim(binsX_TRGB[0], binsX_TRGB[-1])
ax[1,1].set_ylim(-0.05, 0.05)
xr = curveTRGB[0,:]
yr0 = -4 + 0.25 * (xr-2)**2
yr1 = curveTRGB[1,:] - yr0
ax[1,1].plot(xr,yr1,'+',c='black',markersize=4)
# ax[1,1].plot(curveTRGB[0,:],,'+',c='black',markersize=4)


# print('---------------------------------------')
# xaver = (xr[0] + xr[-1]) / 2 + 

for i in range(len(yr1)):
    print(np.round(stepX,3), np.round(yr1[i],5),np.round(binsX_TRGB[i],2))

# print('stepX : ', stepX)
# print('xr    : ', xr)
# print('yr1   : ', yr1)


plt.tight_layout()

FUN10PATH = '/home/pavel/general/TRGB/work/edgeDetection/chern/check_binning/'
FUN10File = 'fun10_step0p' + str(np.round(stepX,3)).ljust(5,'0')[2:] +'.png'
plt.savefig(FUN10PATH+FUN10File)
plt.close()

Hist_in_TRGBregion = Hist_not_log[maskTRGB[0],:][:,maskTRGB[1]]
pltXarray = binsY_TRGB
for i in range(numX_TRGB):
    plt.figure(figsize=(14,6))
    pltHarray = Hist_in_TRGBregion[:,i]
    pltYarray = regionTRGB[:,i]
    pltYarray /= np.max(pltYarray)
    pltYarray *= np.max(pltHarray)
    plt.bar(pltXarray,pltHarray,width=stepX,color='gray')
    plt.plot(pltXarray,pltYarray,'-',color='black',linewidth=2)
    nY = len(pltYarray)
    plt.plot([yr0[i]]*nY, np.linspace(0,np.max(pltHarray),nY),
             '|', markersize=8, c='red')
    yGauss = np.exp(-0.5 * (pltXarray - gauss_mean[i])**2 / gauss_stddev[i]**2)
    # yGauss *= gauss_amplitude[i] / np.max(pltYarray) * np.max(pltHarray)
    yGauss *= np.max(pltHarray)
    plt.plot(pltXarray,yGauss,'-',color='blue',linewidth=2)
    plt.plot([gauss_mean[i]]*nY, np.linspace(0,np.max(pltHarray),nY),
             '|', markersize=5, c='blue')
    plt.title('V-I: '+str(np.round(binsX_TRGB[i],2)))
    filename = 'edge_step0p'+ str(np.round(stepX,3)).ljust(5,'0')[2:] + \
               '_' + str(i).zfill(2) + '.png'
    plt.savefig(filename)
    plt.close()






#================================================
''' This block is intended for saving TRGB curve data
    for the following work with that.
'''
write_curveTRGB_or_not = False

if write_curveTRGB_or_not is True:
    curveTable = Table(np.r_['1,2,0',curveTRGB[0],curveTRGB[1], \
                                      robust_curveTRGB[1,:]],
                   names=('binsX_TRGB','gauss_mean','robust_curve'))
    phot_system = '_'
    if args.datafile.find('SDSS') >= 0:
        phot_system = 'SDSS'
    elif args.datafile.find('PS1') >= 0:
        phot_system = 'PS1'
    elif args.datafile.find('GAIA') >= 0:
        phot_system = 'GAIA'
    ascii.write(curveTable,'curveTRGB_'+phot_system+'.dat',
                format='commented_header',overwrite=True)


#================================================
#================================================
''' This block is intended only for saving
    one slice (column) of the histogram in files.
    Then I needed to draw graphs to compare the slices from the results
    of applying two different kernels of the Sobel filter
    (smoothed 3x3 and smoothed 1x7) 
    only for the initial viewing of the values and shape of the peaks,
    evaluating them for adequacy
'''
write_slices_or_not = False

if write_slices_or_not is True:
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

    # colourV_I -- chosen value of the colour to draw correspond slice
    colourV_I = 2.5
    
    write_color_slice('fileHist_not_log.dat',Hist_not_log,colourV_I)
    write_color_slice('fileEdgeGy.dat',edgeMapY,colourV_I)

#------------------------------------------------
''' This block is intended for saving histogram in a file only for the
    initial viewing of the values evaluating them for adequacy.
'''
write_hist_or_not = False

if write_hist_or_not is True:
    file = open('hist2d_edgeMapXY.dat','w')
    for i in range(numY):
        for j in range(numX):
            if edgeMapXY[i,j] > 7:
                file.write( str( (gridX[j+1]+gridX[j])/2 ) + ' ' + \
                            str( (gridY[i+1]+gridY[i])/2 ) + ' ' + '\n')
    file.close()
