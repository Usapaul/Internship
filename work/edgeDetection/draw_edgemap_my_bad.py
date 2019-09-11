#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table

#------------------------------------------------

dataFileName = 'theTipRGB_HRD.dat'
dataTable = ascii.read(dataFileName)
ndata = len(dataTable)

#------------------------------------------------
# Построение таблицы центральных точек бинов по заданным
# левой и правой, а также верхней и нижней границ 
# и шагу (размеру бина). Чтобы получить центральные точки,
# нужно от границ отойти на половину шага.

# SDSS parameters for histogram
colourBinsizeSDSS = 0.01
colourBoundsSDSS = [0.,6.]

absMagBinsizeSDSS = 0.01
absMagBoundsSDSS = [-6.,0.]

# PanSTARRS1 parameters for histogram
colourBinsizePS1 = 0.01
colourBoundsPS1 = [0.,6.]

absMagBinsizePS1 = 0.01
absMagBoundsPS1 = [-6.,0.]

#------------------------------------------------
# Построение одномерных массивов со смещением на полшага,
# для того, чтобы в дальнейшем из одномерных массивов
# сделать таблицу

colourListSDSS = np.arange(colourBoundsSDSS[0],colourBoundsSDSS[1],
                           colourBinsizeSDSS) + colourBinsizeSDSS/2
absMagListSDSS = np.arange(absMagBoundsSDSS[0],absMagBoundsSDSS[1],
                           absMagBinsizeSDSS) + absMagBinsizeSDSS/2

colourListPS1 = np.arange(colourBoundsPS1[0],colourBoundsPS1[1],
                          colourBinsizePS1) + colourBinsizePS1/2
absMagListPS1 = np.arange(absMagBoundsPS1[0],absMagBoundsPS1[1],
                          absMagBinsizePS1) + absMagBinsizePS1/2
 

mapSDSS_nrows = (absMagBoundsSDSS[1]-absMagBoundsSDSS[0])/absMagBinsizeSDSS



np.arange(absMagBoundsSDSS[0],absMagBoundsSDSS[1],absMagBinsizeSDSS)



absMagListSDSS = [ (i+0.5) * absMagBinsizeSDSS for i in 
                   range(absMagBoundsSDSS[0],absMagBoundsSDSS[1]) ]

binTabSDSS = np.linspace(absMagBoundsSDSS[0] + absMagBinsizeSDSS/2,
                         absMagBoundsSDSS[1] - absMagBinsizeSDSS/2,
                         len())

# PanSTARRS1 parameters for histogram
colourBinsizePS1 = 0.01
absMagBinsizePS1 = 0.01
absMagBoundsPS1 = [-6.,0.]
colourBoundsPS1 = [0.,6.]


colourBoundsSDSS = np.linspace(0,6,60)
absMBoundsSDSS = np.linspace(-6,0,60)



colorBoundsPS1 = np.linspace(0,6,60)




sdssTable = (np.zeros(()),names=('colour','m_i','count'),dtype=('f4','f4','i4'))
